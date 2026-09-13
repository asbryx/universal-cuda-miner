"""Hashcats adapter: Ethereum Keccak-256(address || nonce256 || prev || anchor)."""
from __future__ import annotations

from ..layouts import HASHCATS116, digest_candidate, verify_candidate


def build_message(address: str, nonce: int, prev: str, anchor: str) -> bytes:
    return HASHCATS116.pack({"address": address, "prev": prev, "anchor": anchor}, nonce)


def digest(address: str, nonce: int, prev: str, anchor: str) -> bytes:
    return digest_candidate(HASHCATS116, {"address": address, "prev": prev, "anchor": anchor}, nonce)


def verify(address: str, nonce: int, prev: str, anchor: str, target: int) -> bool:
    return verify_candidate(HASHCATS116, {"address": address, "prev": prev, "anchor": anchor}, nonce, target)


LAYOUT = HASHCATS116
__all__ = ["LAYOUT", "build_message", "digest", "verify"]
