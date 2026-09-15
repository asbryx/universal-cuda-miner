"""Installed and checkout CLI for finite-range CPU proof verification."""
from __future__ import annotations

import argparse
import json

from .layouts import digest_candidate, get_layout, verify_candidate


def parse_fields(raw: str) -> dict[str, str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("--fields must be a JSON object") from exc
    if not isinstance(data, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in data.items()
    ):
        raise ValueError("--fields must contain string keys and values")
    return data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, help="hashbroker84, hashcats116, minerpotatos116, or prspct84")
    parser.add_argument("--fields", required=True, help="JSON object containing public protocol fields")
    parser.add_argument("--target", required=True, type=lambda value: int(value, 0), help="strict uint256 target")
    parser.add_argument("--start", type=lambda value: int(value, 0), default=0)
    parser.add_argument("--count", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--step", type=lambda value: int(value, 0), default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 0 or args.start < 0 or args.step <= 0:
        raise SystemExit("start/count/step must be nonnegative, nonnegative, and positive")
    layout = get_layout(args.protocol)
    fields = parse_fields(args.fields)
    layout.pack(fields, args.start)
    if not 0 <= args.target < 1 << 256:
        raise ValueError("target must be an unsigned uint256")
    if args.count and args.start + (args.count - 1) * args.step >= 1 << (8 * layout.nonce_width):
        raise ValueError("job nonce range overflows the layout nonce field")
    checked = 0
    for index in range(args.count):
        nonce = args.start + index * args.step
        checked += 1
        digest = digest_candidate(layout, fields, nonce)
        if int.from_bytes(digest, "big") < args.target:
            if not verify_candidate(layout, fields, nonce, args.target):
                raise RuntimeError("internal verification mismatch")
            print(json.dumps({
                "protocol": layout.name,
                "nonce": nonce,
                "digest": "0x" + digest.hex(),
                "checked": checked,
                "verified": True,
            }, sort_keys=True))
            return 0
    print(json.dumps({"protocol": layout.name, "checked": checked, "found": False}, sort_keys=True))
    return 1
