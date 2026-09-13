"""HashBroker adapter: SHA-256(address || 24 zero bytes || nonce64 || challenge)."""
from __future__ import annotations

from collections.abc import Mapping

from ..layouts import HASHBROKER84, digest_candidate, verify_candidate


def build_message(address: str, nonce: int, challenge: str) -> bytes:
    return HASHBROKER84.pack({"address": address, "challenge": challenge}, nonce)


def digest(address: str, nonce: int, challenge: str) -> bytes:
    return digest_candidate(HASHBROKER84, {"address": address, "challenge": challenge}, nonce)


def verify(address: str, nonce: int, challenge: str, target: int) -> bool:
    return verify_candidate(HASHBROKER84, {"address": address, "challenge": challenge}, nonce, target)


LAYOUT = HASHBROKER84
__all__ = ["LAYOUT", "build_message", "digest", "verify"]
