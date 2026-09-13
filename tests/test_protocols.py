from __future__ import annotations

import hashlib

import pytest
from eth_hash.auto import keccak

from universal_cuda_miner.layouts import (
    HASHBROKER84,
    HASHCATS116,
    PRSPCT84,
    build_job,
    digest_candidate,
    render_candidate,
    verify_candidate,
)

ADDRESS = "0x" + "12" * 20
CHALLENGE = "0x" + "34" * 32
PREV = "0x" + "56" * 32
ANCHOR = "0x" + "78" * 32
SEED = "0x" + "9a" * 32


def test_hashbroker84_packs_documented_offsets_and_uses_sha256() -> None:
    nonce = 0x0102030405060708
    message = HASHBROKER84.pack(
        {"address": ADDRESS, "challenge": CHALLENGE}, nonce
    )
    expected = (
        bytes.fromhex(ADDRESS[2:])
        + bytes(24)
        + nonce.to_bytes(8, "big")
        + bytes.fromhex(CHALLENGE[2:])
    )
    assert message == expected
    assert len(message) == 84
    assert digest_candidate(HASHBROKER84, {"address": ADDRESS, "challenge": CHALLENGE}, nonce) == hashlib.sha256(expected).digest()


def test_hashcats116_packs_full_width_nonce_and_uses_ethereum_keccak() -> None:
    nonce = int("01" + "23" * 31, 16)
    fields = {"address": ADDRESS, "prev": PREV, "anchor": ANCHOR}
    message = HASHCATS116.pack(fields, nonce)
    expected = bytes.fromhex(ADDRESS[2:]) + nonce.to_bytes(32, "big") + bytes.fromhex(PREV[2:] + ANCHOR[2:])
    assert message == expected
    assert len(message) == 116
    assert digest_candidate(HASHCATS116, fields, nonce) == keccak(expected)


def test_prspct84_packs_seed_address_and_high_nonce_bytes() -> None:
    nonce = int("deadbeef" * 8, 16)
    fields = {"seed": SEED, "address": ADDRESS}
    message = PRSPCT84.pack(fields, nonce)
    expected = bytes.fromhex(SEED[2:] + ADDRESS[2:]) + nonce.to_bytes(32, "big")
    assert message == expected
    assert len(message) == 84
    assert digest_candidate(PRSPCT84, fields, nonce) == keccak(expected)


def test_strict_uint256_target_rejects_equality() -> None:
    fields = {"address": ADDRESS, "prev": PREV, "anchor": ANCHOR}
    digest = digest_candidate(HASHCATS116, fields, 0)
    value = int.from_bytes(digest, "big")
    assert verify_candidate(HASHCATS116, fields, 0, value) is False
    assert verify_candidate(HASHCATS116, fields, 0, value + 1) is True


def test_template_job_carries_runtime_nonce_layout_without_regeneration() -> None:
    fields = {"address": ADDRESS, "prev": PREV, "anchor": ANCHOR}
    job = build_job(HASHCATS116, fields, target=2**256 - 1, start=7, count=32, step=4)
    assert job.template[20:52] == bytes(32)
    assert job.nonce_offset == 20
    assert job.nonce_width == 32
    assert job.start == 7 and job.count == 32 and job.step == 4
    assert job.target_kind == "uint256_lt"


@pytest.mark.parametrize(
    ("layout", "fields", "nonce"),
    [
        (HASHBROKER84, {"address": ADDRESS, "challenge": CHALLENGE}, 0x0102030405060708),
        (HASHCATS116, {"address": ADDRESS, "prev": PREV, "anchor": ANCHOR}, int("23" * 32, 16)),
        (PRSPCT84, {"seed": SEED, "address": ADDRESS}, int("45" * 32, 16)),
    ],
)
def test_runtime_template_render_matches_independent_protocol_pack(
    layout, fields, nonce
) -> None:
    job = build_job(layout, fields, target=2**256 - 1, start=0, count=1)
    # This is the parity boundary consumed by the CUDA worker: the worker gets
    # template/offset/width, not a protocol-specific generated kernel.
    assert render_candidate(job, nonce) == layout.pack(fields, nonce)


@pytest.mark.parametrize("layout", [HASHBROKER84, HASHCATS116, PRSPCT84])
def test_nonce_bounds_are_checked(layout) -> None:
    fields = {
        "address": ADDRESS,
        "challenge": CHALLENGE,
        "prev": PREV,
        "anchor": ANCHOR,
        "seed": SEED,
    }
    required = {name: fields[name] for name in layout.field_names}
    with pytest.raises(ValueError):
        layout.pack(required, -1)
    with pytest.raises(ValueError):
        layout.pack(required, 1 << (8 * layout.nonce_width))


def test_unknown_or_malformed_fields_fail_closed() -> None:
    with pytest.raises(ValueError, match="address"):
        HASHBROKER84.pack({"address": "0x00", "challenge": CHALLENGE}, 0)
    with pytest.raises(ValueError, match="unknown field"):
        HASHBROKER84.pack({"address": ADDRESS, "challenge": CHALLENGE, "extra": "0x00"}, 0)
    with pytest.raises(ValueError, match="target"):
        build_job(HASHCATS116, {"address": ADDRESS, "prev": PREV, "anchor": ANCHOR}, target=2**256)
