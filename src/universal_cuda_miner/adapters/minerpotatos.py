"""MinerPotatos: Keccak-256(address || prevWork || anchor || nonce)."""
from __future__ import annotations

from ..layouts import MINERPOTATOS116, digest_candidate, verify_candidate


def build_message(address: str, prev_work: str, anchor: str, nonce: int) -> bytes:
    return MINERPOTATOS116.pack(
        {"address": address, "prevWork": prev_work, "anchor": anchor}, nonce
    )


def digest(address: str, prev_work: str, anchor: str, nonce: int) -> bytes:
    return digest_candidate(
        MINERPOTATOS116,
        {"address": address, "prevWork": prev_work, "anchor": anchor},
        nonce,
    )


def verify(address: str, prev_work: str, anchor: str, nonce: int, target: int) -> bool:
    return verify_candidate(
        MINERPOTATOS116,
        {"address": address, "prevWork": prev_work, "anchor": anchor},
        nonce,
        target,
    )


LAYOUT = MINERPOTATOS116
__all__ = ["LAYOUT", "build_message", "digest", "verify"]