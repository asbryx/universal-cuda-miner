#!/usr/bin/env python3
"""Remote CUDA correctness and throughput test for hashcats-cuda.

This file deliberately uses Paramiko for both upload and remote execution.  It
never signs, broadcasts, or calls an RPC; it only exercises the leased GPUs.
The worker protocol tested here is one line per job:

    address prevWork anchor target startDEC countDEC

and one response:

    result FOUND NONCE DIGEST MS

"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import paramiko


REMOTE_HOST = os.environ.get("SSH_HOST", "gpu.example.invalid")
REMOTE_PORT = int(os.environ.get("SSH_PORT", "22"))
REMOTE_USER = os.environ.get("SSH_USER", "gpu")
REMOTE_ROOT = os.environ.get("REMOTE_DIR", "/opt/hashcats")
REMOTE_SOURCE = f"{REMOTE_ROOT}/hashcats-cuda.cu"
REMOTE_BINARY = f"{REMOTE_ROOT}/hashcats-cuda"

# The wallet is public work input, not a signing key.
ADDRESS = "0000000000000000000000000000000000000001"
PREV = "".join(f"{i:02x}" for i in range(32))
ANCHOR = "".join(f"{255 - i:02x}" for i in range(32))
KNOWN_NONCE = int("0102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f", 16)
BENCH_COUNT = 33_554_432

MASK64 = (1 << 64) - 1
RC = (
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
)
# R[x][y], matching CUDA lane index x + 5*y.
ROT = (
    (0, 36, 3, 41, 18),
    (1, 44, 10, 45, 2),
    (62, 6, 43, 15, 61),
    (28, 55, 25, 21, 56),
    (27, 20, 39, 8, 14),
)


def rol(x: int, n: int) -> int:
    return ((x << n) | (x >> (64 - n))) & MASK64 if n else x


def keccak_f(state: list[int]) -> None:
    for rc in RC:
        c = [
            state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20]
            for x in range(5)
        ]
        d = [c[(x - 1) % 5] ^ rol(c[(x + 1) % 5], 1) for x in range(5)]
        for x in range(5):
            for y in range(5):
                state[x + 5 * y] ^= d[x]
        b = [0] * 25
        for x in range(5):
            for y in range(5):
                b[y + 5 * ((2 * x + 3 * y) % 5)] = rol(state[x + 5 * y], ROT[x][y])
        for x in range(5):
            for y in range(5):
                state[x + 5 * y] = b[x + 5 * y] ^ (
                    (~b[(x + 1) % 5 + 5 * y]) & b[(x + 2) % 5 + 5 * y]
                )
                state[x + 5 * y] &= MASK64
        state[0] ^= rc


def keccak256(message: bytes) -> bytes:
    rate = 136
    padded = bytearray(message)
    padded.append(0x01)
    padded.extend(b"\x00" * ((rate - (len(padded) % rate) - 1) % rate))
    padded.append(0x80)
    state = [0] * 25
    for block_start in range(0, len(padded), rate):
        block = padded[block_start : block_start + rate]
        for i in range(rate // 8):
            state[i] ^= int.from_bytes(block[8 * i : 8 * i + 8], "little")
        keccak_f(state)
    return b"".join(x.to_bytes(8, "little") for x in state[:4])


def work_digest(address: str, nonce: int, prev: str, anchor: str) -> bytes:
    message = (
        bytes.fromhex(address)
        + nonce.to_bytes(32, "big")
        + bytes.fromhex(prev)
        + bytes.fromhex(anchor)
    )
    assert len(message) == 116
    return keccak256(message)


def make_target_above(digest: bytes) -> str:
    value = int.from_bytes(digest, "big") + 1
    assert value < (1 << 256)
    return f"{value:064x}"


def connect() -> paramiko.SSHClient:
    key_path = Path(os.environ["SSH_KEY_FILE"])
    key = paramiko.Ed25519Key.from_private_key_file(str(key_path))
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    if os.environ.get("SSH_KNOWN_HOSTS"):
        client.load_host_keys(os.environ["SSH_KNOWN_HOSTS"])
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    client.connect(
        REMOTE_HOST,
        port=REMOTE_PORT,
        username=REMOTE_USER,
        pkey=key,
        timeout=30,
        banner_timeout=30,
        auth_timeout=30,
    )
    return client


def remote_command(
    client: paramiko.SSHClient, command: str, timeout: int = 180
) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    channel = stdout.channel
    channel.settimeout(timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return channel.recv_exit_status(), out, err


def upload_and_build(client: paramiko.SSHClient, source: Path) -> None:
    code, out, err = remote_command(
        client, f"mkdir -p {REMOTE_ROOT} && rm -f {REMOTE_BINARY}"
    )
    if code:
        raise RuntimeError(f"remote setup failed: {out}{err}")
    client.open_sftp().put(str(source), REMOTE_SOURCE)
    build = (
        f"cd {REMOTE_ROOT} && "
        "nvcc -O3 -std=c++17 -arch=sm_120 "
        "-Xcompiler=-fno-strict-aliasing hashcats-cuda.cu -o hashcats-cuda"
    )
    code, out, err = remote_command(client, build, timeout=300)
    if code:
        raise RuntimeError(f"nvcc failed (exit {code})\n{out}\n{err}")
    print("remote build: OK")


def run_worker(
    client: paramiko.SSHClient,
    device: int,
    line: str,
    extra_env: dict[str, str] | None = None,
) -> str:
    env = {"CUDA_DEVICE": str(device)}
    if extra_env:
        env.update(extra_env)
    env_text = " ".join(f"{k}={v}" for k, v in env.items())
    command = f"cd {REMOTE_ROOT} && {env_text} {REMOTE_BINARY}"
    stdin, stdout, stderr = client.exec_command(command, timeout=180)
    stdin.write(line + "\n")
    stdin.flush()

    def text(value: object) -> str:
        return (
            value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)
        )

    response = text(stdout.readline()).strip()
    stdin.close()
    out_rest = text(stdout.read())
    err = text(stderr.read())
    exit_code = stdout.channel.recv_exit_status()
    if exit_code:
        raise RuntimeError(
            f"device {device} worker exit {exit_code}: {response} {out_rest} {err}"
        )
    if not response:
        raise RuntimeError(f"device {device} worker returned no response: {err}")
    return response


def parse_result(response: str) -> tuple[int, int | None, str | None, float]:
    fields = response.split()
    if len(fields) != 5 or fields[0] != "result":
        raise AssertionError(f"unexpected worker response: {response!r}")
    found = int(fields[1])
    nonce = None if fields[2] == "-" else int(fields[2], 10)
    digest = None if fields[3] == "-" else fields[3].lower().removeprefix("0x")
    ms = float(fields[4])
    return found, nonce, digest, ms


def run_correctness(client: paramiko.SSHClient) -> list[dict[str, object]]:
    digest = work_digest(ADDRESS, KNOWN_NONCE, PREV, ANCHOR)
    digest_hex = digest.hex()
    target = make_target_above(digest)
    line = f"{ADDRESS} {PREV} {ANCHOR} {target} {KNOWN_NONCE} 1"
    results = []
    for device in range(4):
        response = run_worker(client, device, line)
        found, nonce, returned_digest, ms = parse_result(response)
        if found != 1 or nonce != KNOWN_NONCE or returned_digest != digest_hex:
            raise AssertionError(
                f"device {device} known vector mismatch: {response}; "
                f"expected nonce={KNOWN_NONCE} digest={digest_hex}"
            )
        results.append({"device": device, "response": response, "digest": digest_hex})
    # Equality must fail: the contract and worker both require digest < target.
    equality_line = f"{ADDRESS} {PREV} {ANCHOR} {digest_hex} {KNOWN_NONCE} 1"
    for device in range(4):
        response = run_worker(client, device, equality_line)
        found, nonce, returned_digest, _ = parse_result(response)
        if found != 0 or nonce is not None or returned_digest is not None:
            raise AssertionError(
                f"device {device} accepted digest == target: {response}"
            )
    print(
        "CUDA correctness: 4/4 cards matched CPU Keccak vector; strict equality rejected"
    )
    return results


def benchmark(
    client: paramiko.SSHClient, env: dict[str, str]
) -> list[dict[str, object]]:
    # Target zero is intentionally impossible, so the full batch is always run.
    target = "0" * 64
    line = f"{ADDRESS} {PREV} {ANCHOR} {target} 0 {BENCH_COUNT}"
    rows = []
    for device in range(4):
        response = run_worker(client, device, line, env)
        found, nonce, digest, ms = parse_result(response)
        if found != 0 or nonce is not None or digest is not None or ms <= 0:
            raise AssertionError(
                f"device {device} hard-target benchmark invalid: {response}"
            )
        rate = BENCH_COUNT / (ms / 1000.0)
        row = {
            "device": device,
            "ms": ms,
            "hashes": BENCH_COUNT,
            "hash_per_second": rate,
            "response": response,
        }
        rows.append(row)
        print(
            f"device {device}: {rate / 1e6:.3f} MH/s ({ms:.3f} ms, {BENCH_COUNT:,} hashes)"
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).with_name("hashcats-cuda.cu")
    )
    parser.add_argument("--no-build", action="store_true")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="benchmark a small block/thread setting matrix",
    )
    args = parser.parse_args()

    # Known standard vector guards the CPU verifier itself against SHA3 padding
    # or lane-order mistakes before any remote result is accepted.
    if (
        keccak256(b"").hex()
        != "c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"
    ):
        raise AssertionError(
            "CPU Keccak implementation failed the empty-message vector"
        )

    client = connect()
    try:
        if not args.no_build:
            upload_and_build(client, args.source)
        correctness = run_correctness(client)
        settings = [{"CUDA_THREADS": "256", "CUDA_BLOCKS_PER_SM": "4"}]
        if args.tune:
            settings = [
                {"CUDA_THREADS": "128", "CUDA_BLOCKS_PER_SM": "2"},
                {"CUDA_THREADS": "256", "CUDA_BLOCKS_PER_SM": "2"},
                {"CUDA_THREADS": "256", "CUDA_BLOCKS_PER_SM": "4"},
                {"CUDA_THREADS": "512", "CUDA_BLOCKS_PER_SM": "2"},
            ]
        all_benchmarks = []
        for setting in settings:
            print("benchmark settings:", setting)
            rows = benchmark(client, setting)
            all_benchmarks.append({"settings": setting, "devices": rows})
        report = {"correctness": correctness, "benchmarks": all_benchmarks}
        print("REPORT_JSON " + json.dumps(report, sort_keys=True))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
