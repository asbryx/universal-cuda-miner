from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from eth_account import Account
from eth_utils import keccak

from hashbroker.pow import build_message, leading_zero_bits, verify_candidate
from hashbroker.receipts import TRANSFER_TOPIC, reconstruct_journal

KEY = "0x" + "11" * 32
ADDRESS = Account.from_key(KEY).address
CHALLENGE = "0x" + "22" * 32
CONTRACT = "0x" + "33" * 20


def test_canonical_84_byte_message_and_hash():
    message = build_message(ADDRESS, 0x0102030405060708, CHALLENGE)
    assert len(message) == 84
    assert message == bytes.fromhex(ADDRESS[2:]) + b"\0" * 24 + (0x0102030405060708).to_bytes(8, "big") + bytes.fromhex(CHALLENGE[2:])
    assert hashlib.sha256(message).hexdigest() == hashlib.sha256(message).hexdigest()


def test_candidate_verification_offline():
    nonce = 0
    while leading_zero_bits(hashlib.sha256(build_message(ADDRESS, nonce, CHALLENGE)).digest()) < 8:
        nonce += 1
    digest, bits = verify_candidate(ADDRESS, nonce, CHALLENGE, 8)
    assert digest.startswith("0x") and bits >= 8


def _mint_log(token_id: int, recipient: str = ADDRESS, contract: str = CONTRACT):
    return {
        "address": contract,
        "topics": [TRANSFER_TOPIC, "0x" + "00" * 32, "0x" + recipient[2:].rjust(64, "0"), hex(token_id)],
    }


def test_receipt_reconstruction_repairs_confirmed_accounting_idempotently():
    journal = {
        "transactions": [
            {"tx_hash": "0x" + "aa" * 32, "status": "confirmed", "address": ADDRESS, "mint_price_wei": 7, "receipt": {"status": "0x1", "logs": [_mint_log(41)]}},
            # Duplicate journal row must not double-count spend or token.
            {"tx_hash": "0x" + "aa" * 32, "status": "confirmed", "address": ADDRESS, "mint_price_wei": 7, "receipt": {"status": "0x1", "logs": [_mint_log(41)]}},
        ],
        "token_ids": [],
        "mint_spent_wei": 999,
    }
    rebuilt = reconstruct_journal(journal, CONTRACT)
    assert rebuilt["token_ids"] == [41]
    assert rebuilt["mint_spent_wei"] == 7
    assert reconstruct_journal(rebuilt, CONTRACT) == rebuilt


def test_signer_works_with_synthetic_key_and_never_prints_key(tmp_path: Path):
    keyfile = tmp_path / "one.key"
    keyfile.write_text(KEY + chr(10), encoding="utf-8")
    command = [
        sys.executable, "scripts/sign.py", "--keyfile", str(keyfile), "--chain-id", "1",
        "--nonce", "0", "--to", "0x" + "44" * 20, "--data", "0x", "--gas-price", "1",
    ]
    result = subprocess.run(command, cwd=Path(__file__).parents[1], check=True, capture_output=True, text=True)
    assert KEY not in result.stdout
    payload = json.loads(result.stdout)
    assert payload["sender"].lower() == ADDRESS.lower()
    assert payload["raw_transaction"].startswith("0x")


def test_dry_run_uses_disjoint_two_host_lane_offsets(tmp_path: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    cmd = [sys.executable, "scripts/mine.py", "--address", ADDRESS, "--hosts", "host-a:22:0,host-b:22:4", "--start", "100", "--local-gpus", "4"]
    result = subprocess.run(cmd, cwd=Path(__file__).parents[1], env=env, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["host_base_offsets"] == [0, 4]
    assert payload["nonce_starts"] == [100, 101, 102, 103, 104, 105, 106, 107]
    assert payload["network_actions"] is False
