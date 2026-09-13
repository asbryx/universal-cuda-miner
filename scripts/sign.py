#!/usr/bin/env python3
"""Offline transaction signer; reads one strict local private key and never prints it."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eth_account import Account


def read_key(path: Path) -> str:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("keyfile must contain exactly one private key line")
    key = lines[0]
    if not (key.startswith("0x") and len(key) == 66):
        raise ValueError("keyfile must contain one 0x-prefixed 32-byte private key")
    try:
        account = Account.from_key(key)
    except ValueError as exc:
        raise ValueError("keyfile contains an invalid private key") from exc
    if not account.address:
        raise ValueError("keyfile did not yield an address")
    return key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keyfile", type=Path, required=True)
    parser.add_argument("--chain-id", type=int, required=True)
    parser.add_argument("--nonce", type=int, required=True, help="uint64 proof nonce")
    parser.add_argument("--to", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--value", type=int, default=0)
    parser.add_argument("--gas", type=int, default=300000)
    parser.add_argument("--gas-price", type=int, required=True)
    parser.add_argument("--output", type=Path, help="Write signed JSON locally instead of stdout")
    args = parser.parse_args()
    key = read_key(args.keyfile)
    if not (args.to.startswith("0x") and len(args.to) == 42):
        raise SystemExit("--to must be a 20-byte 0x-prefixed address")
    if not args.data.startswith("0x"):
        raise SystemExit("--data must be 0x-prefixed calldata")
    if min(args.chain_id, args.nonce, args.value, args.gas, args.gas_price) < 0:
        raise SystemExit("numeric transaction fields must not be negative")
    signed = Account.from_key(key).sign_transaction(
        {
            "chainId": args.chain_id,
            "nonce": args.nonce,
            "to": args.to,
            "data": args.data,
            "value": args.value,
            "gas": args.gas,
            "gasPrice": args.gas_price,
        }
    )
    payload = {
        "sender": Account.from_key(key).address,
        "tx_hash": "0x" + signed.hash.hex(),
        "raw_transaction": "0x" + signed.raw_transaction.hex(),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
