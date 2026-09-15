from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, Mapping

from eth_hash.auto import keccak

_HEX = re.compile(r"^(?:0x)?[0-9a-fA-F]*$")


@dataclass(frozen=True)
class Field:
    name: str
    width: int
    fixed: bytes | None = None


@dataclass(frozen=True)
class ProtocolLayout:
    """A fixed packed-message protocol with a runtime nonce field.

    ``nonce_offset`` and ``nonce_width`` describe bytes in the template.  The
    template has zero bytes in that field; a worker may replace it for every
    candidate without regenerating the kernel.  All integer fields are encoded
    unsigned big-endian, matching the wire layouts used by the protocols.
    """

    name: str
    algorithm: str
    message_width: int
    nonce_offset: int
    nonce_width: int
    fields: tuple[Field, ...]
    _digest: Callable[[bytes], bytes]
    target_kind: str = "uint256_lt"

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(
            field.name for field in self.fields
            if field.name != "nonce" and field.fixed is None
        )

    def pack(self, values: Mapping[str, str], nonce: int) -> bytes:
        unknown = set(values) - set(self.field_names)
        if unknown:
            raise ValueError(f"unknown field(s): {', '.join(sorted(unknown))}")
        if not isinstance(nonce, int) or isinstance(nonce, bool):
            raise ValueError("nonce must be an integer")
        if not 0 <= nonce < 1 << (8 * self.nonce_width):
            raise ValueError(f"nonce must fit unsigned {self.nonce_width * 8}-bit field")
        message = bytearray(self.message_width)
        cursor = 0
        for field in self.fields:
            if field.name == "nonce":
                raw = nonce.to_bytes(field.width, "big")
            elif field.fixed is not None:
                if len(field.fixed) != field.width:
                    raise AssertionError("fixed field width does not match layout")
                raw = field.fixed
            else:
                if field.name not in values:
                    raise ValueError(f"missing field: {field.name}")
                raw = _parse_hex(values[field.name], field.width, field.name)
            message[cursor : cursor + field.width] = raw
            cursor += field.width
        if cursor != self.message_width:
            raise AssertionError("layout field widths do not cover message")
        return bytes(message)

    def digest(self, message: bytes) -> bytes:
        if len(message) != self.message_width:
            raise ValueError(f"message must be exactly {self.message_width} bytes")
        return self._digest(message)


def _parse_hex(value: str, width: int, name: str) -> bytes:
    if not isinstance(value, str) or not _HEX.fullmatch(value):
        raise ValueError(f"{name} must be hexadecimal")
    digits = value[2:] if value[:2].lower() == "0x" else value
    if len(digits) != width * 2:
        raise ValueError(f"{name} must be exactly {width} bytes")
    return bytes.fromhex(digits)


def _sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


HASHBROKER84 = ProtocolLayout(
    name="hashbroker84",
    algorithm="sha256",
    message_width=84,
    nonce_offset=44,
    nonce_width=8,
    fields=(Field("address", 20), Field("padding", 24, bytes(24)), Field("nonce", 8), Field("challenge", 32)),
    _digest=_sha256,
)

HASHCATS116 = ProtocolLayout(
    name="hashcats116",
    algorithm="keccak256",
    message_width=116,
    nonce_offset=20,
    nonce_width=32,
    fields=(Field("address", 20), Field("nonce", 32), Field("prev", 32), Field("anchor", 32)),
    _digest=keccak,
)

MINERPOTATOS116 = ProtocolLayout(
    name="minerpotatos116",
    algorithm="keccak256",
    message_width=116,
    nonce_offset=84,
    nonce_width=32,
    fields=(Field("address", 20), Field("prevWork", 32), Field("anchor", 32), Field("nonce", 32)),
    _digest=keccak,
)

PRSPCT84 = ProtocolLayout(
    name="prspct84",
    algorithm="keccak256",
    message_width=84,
    nonce_offset=52,
    nonce_width=32,
    fields=(Field("seed", 32), Field("address", 20), Field("nonce", 32)),
    _digest=keccak,
)

_LAYOUTS = {layout.name: layout for layout in (HASHBROKER84, HASHCATS116, MINERPOTATOS116, PRSPCT84)}


def get_layout(name: str) -> ProtocolLayout:
    try:
        return _LAYOUTS[name.lower()]
    except (KeyError, AttributeError) as exc:
        raise ValueError(f"unknown protocol layout: {name}") from exc


def digest_candidate(layout: ProtocolLayout, values: Mapping[str, str], nonce: int) -> bytes:
    return layout.digest(layout.pack(values, nonce))


def leading_zero_bits(value: bytes) -> int:
    count = 0
    for byte in value:
        if byte == 0:
            count += 8
        else:
            count += 8 - byte.bit_length()
            break
    return count


def verify_candidate(layout: ProtocolLayout, values: Mapping[str, str], nonce: int, target: int) -> bool:
    if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < 1 << 256:
        raise ValueError("target must be an unsigned uint256")
    return int.from_bytes(digest_candidate(layout, values, nonce), "big") < target


@dataclass(frozen=True)
class Job:
    layout: ProtocolLayout
    template: bytes
    nonce_offset: int
    nonce_width: int
    target: int
    start: int
    count: int
    step: int

    @property
    def target_kind(self) -> str:
        return self.layout.target_kind


def build_job(
    layout: ProtocolLayout,
    values: Mapping[str, str],
    *,
    target: int,
    start: int = 0,
    count: int = 0,
    step: int = 1,
) -> Job:
    if not isinstance(target, int) or isinstance(target, bool) or not 0 <= target < 1 << 256:
        raise ValueError("target must be an unsigned uint256")
    if not isinstance(start, int) or start < 0 or start >= 1 << (8 * layout.nonce_width):
        raise ValueError("start must fit the layout nonce field")
    if not isinstance(count, int) or count < 0:
        raise ValueError("count must be nonnegative")
    if not isinstance(step, int) or step <= 0:
        raise ValueError("step must be positive")
    if count and start + (count - 1) * step >= 1 << (8 * layout.nonce_width):
        raise ValueError("job nonce range overflows the layout nonce field")
    template = layout.pack(values, 0)
    return Job(layout, template, layout.nonce_offset, layout.nonce_width, target, start, count, step)


def render_candidate(job: Job, nonce: int) -> bytes:
    if not 0 <= nonce < 1 << (8 * job.nonce_width):
        raise ValueError("nonce outside layout field")
    rendered = bytearray(job.template)
    rendered[job.nonce_offset : job.nonce_offset + job.nonce_width] = nonce.to_bytes(job.nonce_width, "big")
    return bytes(rendered)


def serialize_job(job: Job, job_id: str = "job") -> str:
    """Serialize a :class:`Job` for the persistent CUDA stdin protocol."""
    if not job_id or any(character.isspace() for character in job_id):
        raise ValueError("job_id must be a nonempty token")
    target = f"{job.target:064x}"
    start = f"{job.start:064x}"
    return (
        f"job {job_id} {job.layout.name} {job.layout.algorithm} "
        f"{job.nonce_offset} {job.nonce_width} {job.template.hex()} "
        f"{target} {start} {job.count} {job.step}"
    )


__all__ = [
    "HASHBROKER84",
    "HASHCATS116",
    "MINERPOTATOS116",
    "PRSPCT84",
    "Job",
    "ProtocolLayout",
    "build_job",
    "digest_candidate",
    "get_layout",
    "leading_zero_bits",
    "render_candidate",
    "serialize_job",
    "verify_candidate",
]
