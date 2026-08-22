"""Policy server: observations in, action chunks out (§5.3).

Async inference with Real-Time Chunking, ported from LeRobot's
`async_inference.policy_server` (thread/queue structure, observation filtering) and
`rollout.inference.rtc.RTCInferenceEngine` (the RTC prefix bookkeeping) onto this
repo's ZMQ transport, so the Pi never has to import lerobot or torch.

Two threads:

  socket loop (main)  ROUTER. Receives observations, drops the ones not worth running,
                      sends finished chunks back. It owns the socket outright — ZMQ
                      sockets are not thread-safe — and never blocks on the GPU.
  inference thread    Takes the newest observation, runs the policy with the previous
                      chunk's unexecuted tail as the RTC prefix, hands the result back.

Why the RTC state lives here and not on the Pi: the guidance prefix has to be in the
policy's own normalised action space, so the server keeps the *original* chunk while
the Pi executes the *postprocessed* one. Both are indexed by the same absolute
timestep, so the two copies cannot drift apart.

Decoding is unchanged — common/preprocess, the same spatial transform convert.py used
to build the dataset — as is the pre/postprocessor pair loaded from the checkpoint, so
the actions on the wire stay in the normalised joint units servo.write_positions wants.

The task string must match the one the dataset was recorded with, so it is read back
from that dataset (via the checkpoint's train_config.json) rather than typed in.

Desktop-only. torch + lerobot live here where torch is fine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import torch
import zmq
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc import LatencyTracker, RTCConfig
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.processor import RelativeActionsProcessorStep

from common.preprocess import decode_frame
from common.protocol import TimedChunk, TimedObservation, pack_reply, unpack_request
from common.schema import CONTROL_HZ, STATE, image_key

ROBOT_TYPE = "so101"

# Socket loop poll timeout: short enough that finished chunks go out promptly.
POLL_MS = 5
# How long the inference thread waits on an empty observation queue before looping.
OBS_QUEUE_TIMEOUT = 2.0
# Joint-space distance below which two observations count as the same one (lerobot's
# `observations_similar`, joint units).
STATE_ATOL = 1.0


def dataset_tasks(checkpoint: str) -> list[str]:
    """Task strings of the dataset this checkpoint was trained on; [] if unreadable."""
    config = Path(checkpoint) / "train_config.json"  # written by lerobot-train
    if not config.is_file():
        return []
    dataset = json.loads(config.read_text()).get("dataset", {})
    try:
        meta = LeRobotDatasetMetadata(dataset["repo_id"], root=dataset.get("root"))
    except Exception as e:  # dataset moved, offline, format drift — not fatal
        print(f"[serve] could not read tasks from {dataset.get('repo_id')}: {e}")
        return []
    return list(meta.tasks.index)


def resolve_task(requested: str | None, tasks: list[str]) -> str:
    """Pick the task to deploy: the flag if given, else the dataset's, asking if ambiguous."""
    if requested is not None:
        if tasks and requested not in tasks:
            print(f"[warn] --task is not one the dataset was recorded with: {tasks}")
        return requested
    if not tasks:
        raise SystemExit(
            "[serve] no task found in the checkpoint's dataset — pass --task explicitly"
        )
    if len(tasks) == 1:
        print(f"[serve] task: {tasks[0]!r}")
        return tasks[0]
    print("[serve] the dataset was trained on several tasks:")
    for i, task in enumerate(tasks):
        print(f"  [{i}] {task}")
    if not sys.stdin.isatty():
        raise SystemExit("[serve] several tasks and no terminal to ask — pass --task explicitly")
    return tasks[int(input("Select a task to deploy: "))]


def normalize_prev_actions_length(prev_actions: torch.Tensor, target_steps: int) -> torch.Tensor:
    """Pad or truncate the RTC prefix to a fixed length (lerobot rollout/inference/rtc.py)."""
    steps, action_dim = prev_actions.shape
    if steps == target_steps:
        return prev_actions
    if steps > target_steps:
        return prev_actions[:target_steps]
    padded = torch.zeros((target_steps, action_dim), dtype=prev_actions.dtype, device=prev_actions.device)
    padded[:steps] = prev_actions
    return padded


@dataclass
class PrevChunk:
    """The last chunk produced, in the policy's own action space — the RTC prefix source."""

    actions: torch.Tensor  # (T, action_dim), pre-postprocessor
    start: int             # absolute timestep of actions[0]


class PolicyServer:
    def __init__(self, policy, preprocessor, postprocessor, task, device, args):
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.task = task
        self.device = device
        self.actions_per_chunk = args.actions_per_chunk
        self.execution_horizon = args.execution_horizon
        self.rtc = not args.no_rtc

        self.observation_queue: Queue = Queue(maxsize=1)  # only the newest obs is worth running
        self.outbound: Queue = Queue()

        self._prev_chunk: PrevChunk | None = None
        self._last_processed_obs: TimedObservation | None = None
        self._predicted_timesteps: set[int] = set()
        self._lock = threading.Lock()
        self._latency = LatencyTracker()
        # Held across a whole inference and across reset(), so a new client cannot reset
        # the policy out from under a forward pass. Only ever contended on reconnect.
        self._policy_lock = threading.Lock()
        self._client: bytes | None = None

    # ── observation filtering (lerobot async_inference/policy_server.py) ────────

    def enqueue_observation(self, identity: bytes, obs: TimedObservation) -> bool:
        """Queue an observation if it is worth running; drop it otherwise.

        Dropping is silent, as in lerobot: the Pi streams a fresh observation every tick,
        so the cheapest way to say no is to say nothing and answer the next one.
        """
        if not (obs.must_go or self._last_processed_obs is None or self._worth_running(obs)):
            return False
        if self.observation_queue.full():
            self.observation_queue.get_nowait()  # never queue a backlog, only the newest
        self.observation_queue.put((identity, obs))
        return True

    def _worth_running(self, obs: TimedObservation) -> bool:
        with self._lock:
            already_predicted = obs.timestep in self._predicted_timesteps
        if already_predicted:
            return False
        # Same joint configuration as the last one we ran — the chunk would be the same.
        return bool(np.linalg.norm(obs.state - self._last_processed_obs.state) >= STATE_ATOL)

    def reset(self, client: bytes) -> None:
        """Adopt `client` and drop all per-run state. Called when a new client connects.

        The policy and both processors are stateful — smolvla caches observation queues,
        the normaliser and relative-action steps cache the last state — so lerobot's
        `RTCInferenceEngine.reset` resets all three, and so do we. Taking the policy lock
        blocks until any inference still running for the old client has finished, which is
        what keeps `policy.reset()` off a live forward pass and stops that inference from
        writing `_prev_chunk` back after we clear it.
        """
        with self._policy_lock:
            self._client = client
            self.policy.reset()
            self.preprocessor.reset()
            self.postprocessor.reset()
            with self._lock:
                self._predicted_timesteps = set()
            self._prev_chunk = None
            self._last_processed_obs = None
            self._latency.reset()
            while not self.observation_queue.empty():
                self.observation_queue.get_nowait()

    # ── RTC ────────────────────────────────────────────────────────────────────

    def rtc_prefix(self, timestep: int) -> torch.Tensor | None:
        """The previous chunk's actions from `timestep` on, in policy action space.

        Index j of the returned prefix and index j of the chunk about to be generated
        are the same absolute timestep, which is what makes the guidance meaningful.
        """
        if self._prev_chunk is None:
            return None
        offset = timestep - self._prev_chunk.start
        if offset < 0:  # observation predates the last chunk; nothing to blend against
            return None
        leftover = self._prev_chunk.actions[offset:]
        if leftover.numel() == 0:
            return None
        return normalize_prev_actions_length(leftover, self.execution_horizon)

    def inference_delay(self, obs: TimedObservation) -> int:
        """How many actions will already be executed by the time this chunk lands.

        The Pi measures its own round trip and reports it, which is the honest number
        over a network; lerobot infers it from inference latency alone because there
        the policy runs in the robot's own process. Its own inference latency is the
        fallback for the first chunk, before the Pi has an estimate.
        """
        if obs.delay > 0:
            return min(obs.delay, self.actions_per_chunk)
        return min(math.ceil(self._latency.p95() * CONTROL_HZ), self.actions_per_chunk)

    # ── inference ──────────────────────────────────────────────────────────────

    def predict(self, obs: TimedObservation) -> np.ndarray:
        start = time.perf_counter()

        observation = {STATE: obs.state.copy()}  # copy: frombuffer is read-only
        for name, jpeg in obs.frames.items():
            observation[image_key(name)] = decode_frame(jpeg)
        observation = prepare_observation_for_inference(observation, self.device, self.task, ROBOT_TYPE)

        delay = self.inference_delay(obs)
        prefix = self.rtc_prefix(obs.timestep) if self.rtc else None

        # no_grad, not inference_mode: RTC's guidance term is an autograd.grad through
        # the denoiser, and inference_mode tensors cannot be re-enabled for grad.
        with torch.no_grad():
            batch = self.preprocessor(observation)
            if self.rtc:
                chunk = self.policy.predict_action_chunk(
                    batch, inference_delay=delay, prev_chunk_left_over=prefix
                )
            else:
                chunk = self.policy.predict_action_chunk(batch)
            chunk = chunk[:, : self.actions_per_chunk]
            original = chunk.squeeze(0).clone()  # keep the un-postprocessed copy for RTC
            processed = self.postprocessor(chunk).squeeze(0)

        self._prev_chunk = PrevChunk(actions=original, start=obs.timestep)
        self._last_processed_obs = obs
        self._latency.add(time.perf_counter() - start)

        prefix_len = 0 if prefix is None else int(prefix.shape[0])
        print(
            f"[serve] obs #{obs.timestep} -> chunk {tuple(processed.shape)} | "
            f"{(time.perf_counter() - start) * 1000:.0f}ms | delay {delay} | prefix {prefix_len}"
        )
        return processed.float().cpu().numpy()

    def inference_loop(self) -> None:
        while True:
            try:
                identity, obs = self.observation_queue.get(timeout=OBS_QUEUE_TIMEOUT)
            except Empty:
                continue
            with self._policy_lock:
                if identity != self._client:
                    continue  # queued before a reconnect; that client is gone
                with self._lock:
                    self._predicted_timesteps.add(obs.timestep)
                try:
                    actions = self.predict(obs)
                except Exception as e:  # one bad frame must not take the server down
                    print(f"[serve] inference failed for obs #{obs.timestep}: {e}")
                    continue
            self.outbound.put(
                (identity, TimedChunk(seq=obs.seq, timestep=obs.timestep, actions=actions, rtc=self.rtc))
            )

    # ── socket loop ────────────────────────────────────────────────────────────

    def serve_forever(self, bind: str) -> None:
        sock = zmq.Context.instance().socket(zmq.ROUTER)
        sock.setsockopt(zmq.LINGER, 0)
        sock.bind(bind)
        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        threading.Thread(target=self.inference_loop, daemon=True).start()
        print(f"[serve] listening on {bind} (rtc={self.rtc})")

        while True:
            if poller.poll(POLL_MS):
                parts = sock.recv_multipart()
                identity, obs = parts[0], unpack_request(parts[1:])
                if identity != self._client:  # new client: nothing from the old run applies
                    print(f"[serve] client {identity.hex()} connected")
                    self.reset(identity)
                self.enqueue_observation(identity, obs)

            while not self.outbound.empty():
                identity, chunk = self.outbound.get_nowait()
                sock.send_multipart([identity, *pack_reply(chunk)])


def load_policy(args, task: str):
    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint)

    if not args.no_rtc:
        if not hasattr(cfg, "rtc_config") or not hasattr(policy, "init_rtc_processor"):
            raise SystemExit(
                f"[serve] policy type {cfg.type!r} has no RTC support (it is not a flow-matching "
                f"policy) — rerun with --no-rtc for plain async chunking"
            )
        # lerobot/rollout/context.py: attach the config, then build the processor.
        policy.config.rtc_config = RTCConfig(
            enabled=True,
            execution_horizon=args.execution_horizon,
            max_guidance_weight=args.max_guidance_weight,
        )
        policy.init_rtc_processor()

    policy = policy.to(args.device).eval()

    # Stats and steps come from the checkpoint; no dataset needed at serve time.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.checkpoint,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    relative = next(
        (s for s in preprocessor.steps if isinstance(s, RelativeActionsProcessorStep) and s.enabled),
        None,
    )
    if relative is not None and not args.no_rtc:
        # lerobot re-anchors the prefix against the current joint state for these; we do
        # not, and a silently mis-anchored prefix is worse than no RTC.
        raise SystemExit("[serve] relative-action checkpoints are not supported with RTC — use --no-rtc")
    return policy, preprocessor, postprocessor


def main():
    ap = argparse.ArgumentParser(description="SO-101 policy server.")
    ap.add_argument("--checkpoint", required=True, help="Path or repo id of a trained policy")
    ap.add_argument("--bind", default="tcp://*:5555")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", default=None,
                    help="Language task. Defaults to the task the dataset was recorded with.")
    ap.add_argument("--actions-per-chunk", type=int, default=50,
                    help="Actions kept per chunk; the rest of the policy's horizon is dropped")
    ap.add_argument("--execution-horizon", type=int, default=10,
                    help="RTC: steps over which the prefix guidance decays to zero")
    ap.add_argument("--max-guidance-weight", type=float, default=10.0, help="RTC: guidance clamp")
    ap.add_argument("--no-rtc", action="store_true",
                    help="Async chunking without RTC; the Pi blends overlapping actions instead")
    args = ap.parse_args()

    task = resolve_task(args.task, dataset_tasks(args.checkpoint))
    policy, preprocessor, postprocessor = load_policy(args, task)
    print(f"[serve] {args.checkpoint} on {args.device}")

    server = PolicyServer(policy, preprocessor, postprocessor, task, args.device, args)
    server.serve_forever(args.bind)


if __name__ == "__main__":
    main()
