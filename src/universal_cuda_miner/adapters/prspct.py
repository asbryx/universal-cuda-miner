"""PRSPCT adapter: Ethereum Keccak-256(seed || address || nonce256)."""
from __future__ import annotations

from ..layouts import PRSPCT84, digest_candidate, verify_candidate


def build_message(seed: str, address: str, nonce: int) -> bytes:
    return PRSPCT84.pack({"seed": seed, "address": address}, nonce)


def digest(seed: str, address: str, nonce: int) -> bytes:
    return digest_candidate(PRSPCT84, {"seed": seed, "address": address}, nonce)


def verify(seed: str, address: str, nonce: int, target: int) -> bool:
    return verify_candidate(PRSPCT84, {"seed": seed, "address": address}, nonce, target)


LAYOUT = PRSPCT84
__all__ = ["LAYOUT", "build_message", "digest", "verify"]
