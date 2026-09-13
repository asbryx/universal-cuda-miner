"""Deterministic, idempotent receipt reconstruction.

Recovery derives accounting from confirmed receipts every time. A journal's
old ``confirmed`` marker is not trusted as proof that token IDs or spend were
accounted for.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from eth_utils import keccak

TRANSFER_TOPIC = "0x" + keccak(text="Transfer(address,address,uint256)").hex()


@dataclass(frozen=True)
class Reconstructed:
    status: str
    token_ids: tuple[int, ...]
    spend_wei: int


def _hex_address(value: str) -> str:
    return "0x" + value[-40:].lower()


def token_ids_from_receipt(receipt: dict[str, Any], contract: str, recipient: str) -> list[int]:
    """Return only ERC-721 Transfer mints for this contract and recipient."""
    if str(receipt.get("status", "0x0")).lower() != "0x1":
        return []
    contract = contract.lower()
    recipient = recipient.lower()
    found: list[int] = []
    for log in receipt.get("logs", []):
        topics = log.get("topics", [])
        if (
            str(log.get("address", "")).lower() == contract
            and len(topics) == 4
            and str(topics[0]).lower() == TRANSFER_TOPIC.lower()
            and int(str(topics[1]), 16) == 0
            and _hex_address(str(topics[2])) == recipient
        ):
            found.append(int(str(topics[3]), 16))
    return found


def reconstruct_record(record: dict[str, Any], contract: str) -> Reconstructed:
    """Recompute one record's accounting from its receipt, safely and idempotently."""
    receipt = record.get("receipt") or {}
    status = "confirmed" if str(receipt.get("status", "")).lower() == "0x1" else "reverted"
    if status != "confirmed":
        return Reconstructed(status, (), 0)
    ids = token_ids_from_receipt(receipt, contract, str(record["address"]))
    # A mint receipt must account for exactly one token; reject ambiguous logs.
    if len(ids) != 1:
        raise ValueError("confirmed receipt does not contain exactly one matching mint")
    return Reconstructed(status, tuple(ids), int(record.get("mint_price_wei", 0)))


def reconstruct_journal(journal: dict[str, Any], contract: str) -> dict[str, Any]:
    """Rebuild tokens and spend from all confirmed transaction receipts.

    Calling this repeatedly produces the same result: token IDs are deduped and
    spend is summed once per unique transaction hash.
    """
    result = deepcopy(journal)
    transactions = result.get("transactions", [])
    seen_hashes: set[str] = set()
    token_ids: set[int] = set()
    spend = 0
    for record in transactions:
        if str(record.get("status", "")).lower() != "confirmed":
            continue
        tx_hash = str(record.get("tx_hash", ""))
        if tx_hash in seen_hashes:
            continue
        seen_hashes.add(tx_hash)
        item = reconstruct_record(record, contract)
        token_ids.update(item.token_ids)
        spend += item.spend_wei
    result["token_ids"] = sorted(token_ids)
    result["mint_spent_wei"] = spend
    return result
