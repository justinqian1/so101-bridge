"""Policy server: observations in, action chunks out (§5.3).

ZMQ REP lockstep with the Pi client. Decodes JPEGs through common/preprocess (the same
spatial transform convert.py used to build the dataset), runs a LeRobot policy's
action-chunk prediction, and echoes the sequence number back.

This is our own minimal REQ/REP (§5.3 recommendation) — we own the timeout semantics,
which is exactly the part worth owning. Revisit LeRobot's async server only if its
chunk aggregation turns out to do something we want.

Desktop-only. torch + lerobot live here where torch is fine.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import zmq
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class

from common.preprocess import decode_frame, to_policy_tensor
from common.protocol import pack_reply, unpack_request
from common.schema import STATE, image_key


def load_policy(checkpoint: str, device: str):
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint)
    policy.to(device).eval()
    return policy


def build_batch(state: np.ndarray, frames: dict[str, bytes], task: str | None, device: str) -> dict:
    batch = {STATE: torch.from_numpy(state.copy())[None].to(device)}
    for name, jpeg in frames.items():
        tensor = to_policy_tensor(decode_frame(jpeg))
        batch[image_key(name)] = tensor[None].to(device)
    if task is not None:
        batch["task"] = [task]
    return batch


def main():
    ap = argparse.ArgumentParser(description="SO-101 policy server.")
    ap.add_argument("--checkpoint", required=True, help="Path or repo id of a trained policy")
    ap.add_argument("--bind", default="tcp://*:5555")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--task", default=None, help="Language task, for VLA policies")
    args = ap.parse_args()

    policy = load_policy(args.checkpoint, args.device)
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(args.bind)
    print(f"[serve] {args.checkpoint} on {args.device}, listening on {args.bind}")

    while True:
        seq, _t, state, frames = unpack_request(sock.recv_multipart())
        batch = build_batch(state, frames, args.task, args.device)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(batch)  # [1, T, action_dim]
        actions = chunk[0].float().cpu().numpy()
        sock.send_multipart(pack_reply(seq, actions))


if __name__ == "__main__":
    main()
