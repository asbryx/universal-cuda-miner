"""HashBroker proof-of-work message and verification primitives.

The public protocol hashes exactly 84 bytes:
20-byte address + 24 zero bytes + uint64 nonce + 32-byte challenge.
"""
from __future__ import annotations

import hashlib
import re

_HEX_20 = re.compile(r"^0x[0-9a-fA-F]{40}$")
_HEX_32 = re.compile(r"^0x[0-9a-fA-F]{64}$")


def _hex_bytes(value: str, pattern: re.Pattern[str], name: str) -> bytes:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{name} must be a 0x-prefixed fixed-width hex value")
    return bytes.fromhex(value[2:])


def build_message(address: str, nonce: int, challenge: str) -> bytes:
    """Build the canonical 84-byte message without network or wallet access."""
    address_bytes = _hex_bytes(address, _HEX_20, "address")
    challenge_bytes = _hex_bytes(challenge, _HEX_32, "challenge")
    if not isinstance(nonce, int) or nonce < 0 or nonce >= 1 << 64:
        raise ValueError("nonce must be an unsigned uint64")
    message = address_bytes + (b"\0" * 24) + nonce.to_bytes(8, "big") + challenge_bytes
    assert len(message) == 84
    return message


def digest(address: str, nonce: int, challenge: str) -> bytes:
    return hashlib.sha256(build_message(address, nonce, challenge)).digest()


def leading_zero_bits(value: bytes) -> int:
    """Count leading zero bits in a byte string."""
    count = 0
    for byte in value:
        if byte == 0:
            count += 8
            continue
        count += 8 - byte.bit_length()
        break
    return count


def verify_candidate(address: str, nonce: int, challenge: str, difficulty: int) -> tuple[str, int]:
    if not isinstance(difficulty, int) or not 0 <= difficulty <= 256:
        raise ValueError("difficulty must be between 0 and 256")
    raw = digest(address, nonce, challenge)
    bits = leading_zero_bits(raw)
    if bits < difficulty:
        raise ValueError("candidate does not meet difficulty")
    return "0x" + raw.hex(), bits
