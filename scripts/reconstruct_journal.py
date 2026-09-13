#!/usr/bin/env python3
"""Rebuild token and spend accounting from receipts without network access."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support direct execution from a source checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hashbroker.receipts import reconstruct_journal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    journal = json.loads(args.journal.read_text(encoding="utf-8"))
    rebuilt = reconstruct_journal(journal, args.contract)
    args.output.write_text(json.dumps(rebuilt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"transactions": len(rebuilt.get("transactions", [])), "token_ids": len(rebuilt["token_ids"]), "mint_spent_wei": rebuilt["mint_spent_wei"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
