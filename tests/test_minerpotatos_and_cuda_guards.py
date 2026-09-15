from __future__ import annotations

from pathlib import Path

from eth_hash.auto import keccak

from universal_cuda_miner.layouts import build_job, digest_candidate, get_layout

ROOT = Path(__file__).parents[1]


def test_minerpotatos_layout_matches_site_preimage_and_keccak() -> None:
    layout = get_layout("minerpotatos116")
    fields = {
        "address": "0x" + "11" * 20,
        "prevWork": "0x" + "22" * 32,
        "anchor": "0x" + "33" * 32,
    }
    nonce = 0x0102030405060708
    message = layout.pack(fields, nonce)
    expected = bytes.fromhex("11" * 20 + "22" * 32 + "33" * 32) + nonce.to_bytes(32, "big")
    assert message == expected
    # Independent vector captured from the deployed contract's pure workFor().
    assert digest_candidate(layout, fields, nonce).hex() == (
        "70b06f5a778a7cfbc8d88e2a2eb6eb0f88f308993a2cede2f523cdc808cd41ec"
    )
    assert digest_candidate(layout, fields, nonce) == keccak(expected)


def test_minerpotatos_layout_has_nonce_at_byte_84() -> None:
    layout = get_layout("minerpotatos116")
    job = build_job(
        layout,
        {"address": "0x" + "11" * 20, "prevWork": "0x" + "22" * 32, "anchor": "0x" + "33" * 32},
        target=(1 << 256) - 1,
        start=0,
        count=1,
    )
    assert layout.message_width == 116
    assert layout.nonce_offset == 84
    assert job.template[84:] == bytes(32)


def test_cuda_source_contains_runtime_self_test_and_no_undefined_nonce_namespace() -> None:
    source = (ROOT / "cuda" / "universal_cuda_miner.cu").read_text(encoding="utf-8")
    assert "--self-test" in source
    assert "nonce_range::" not in source
    assert "self_test" in source


def test_cuda_compile_workflow_is_runtime_gated() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "--self-test" in workflow
    assert "cuda-compile" in workflow
    assert "nvidia-smi" in workflow


def test_documented_minerpotatos_protocol() -> None:
    docs = (ROOT / "docs" / "protocols.md").read_text(encoding="utf-8")
    assert "minerpotatos116" in docs
    assert "previous work 32" in docs and "nonce uint256" in docs
