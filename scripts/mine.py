#!/usr/bin/env python3
"""Configurable HashBroker CUDA supervisor.

Dry-run is the default. Only ``--live`` may start SSH workers or broadcast a
transaction. The script never logs URLs, key material, raw environments, or
private transaction fields.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Support both ``python -m`` and direct script execution from a checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import requests
from eth_utils import keccak

from hashbroker.config import Settings, load_settings
from hashbroker.pow import verify_candidate
from hashbroker.receipts import reconstruct_journal

CHALLENGE_SELECTOR = "0xd2ef7398"
DIFFICULTY_SELECTOR = "0x5c062d6c"
PRICE_SELECTOR = "0x6817c76c"
MINE_SELECTOR = "0x" + keccak(text="mine(uint256,bytes32)")[:4].hex()


@dataclass(frozen=True)
class Host:
    hostname: str
    port: int
    base_offset: int


def validate_lane_layout(hosts: tuple[Host, ...], local_gpus: int) -> None:
    """Require each host to own one contiguous, non-overlapping lane block."""
    if local_gpus < 1:
        raise ValueError("local_gpus must be positive")
    expected = 0
    for host in sorted(hosts, key=lambda item: item.base_offset):
        if host.base_offset < expected:
            raise ValueError("host base offsets overlap local GPU lanes")
        if host.base_offset > expected:
            raise ValueError("host base offsets must form contiguous lane blocks")
        expected += local_gpus


def parse_hosts(value: tuple[str, ...]) -> tuple[Host, ...]:
    result = []
    for item in value:
        fields = item.rsplit(":", 2)
        if len(fields) != 3:
            raise ValueError("HOSTS entries must be host:port:base_offset")
        host, port, base = fields
        if not host or int(port) < 1 or int(base) < 0:
            raise ValueError("HOSTS entry has invalid host, port, or base_offset")
        result.append(Host(host, int(port), int(base)))
    if not result:
        raise ValueError("HOSTS is empty")
    return tuple(result)


def redacted_error(exc: BaseException) -> str:
    # requests exceptions can echo full key-bearing URLs; expose only a type.
    return type(exc).__name__


def rpc(urls: tuple[str, ...], method: str, params: list[Any]) -> Any:
    if not urls:
        raise RuntimeError("RPC_URLS is required")
    last_type = "RuntimeError"
    for url in urls:
        try:
            response = requests.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("error"):
                last_type = "RPCError"
                continue
            return payload["result"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_type = redacted_error(exc)
    raise RuntimeError(f"all configured RPC endpoints failed ({last_type})")


def read_state(settings: Settings) -> dict[str, Any]:
    address = settings.contract_address
    return {
        "challenge": rpc(settings.rpc_urls, "eth_call", [{"to": address, "data": CHALLENGE_SELECTOR}, "latest"]),
        "difficulty": int(rpc(settings.rpc_urls, "eth_call", [{"to": address, "data": DIFFICULTY_SELECTOR}, "latest"]), 16),
        "price": int(rpc(settings.rpc_urls, "eth_call", [{"to": address, "data": PRICE_SELECTOR}, "latest"]), 16),
    }


def job_line(job_id: str, address: str, state: dict[str, Any], start: int, workers: int) -> str:
    return f"job {job_id} {address} {state['challenge']} {state['difficulty']} {start} {workers}"


def build_commands(settings: Settings, hosts: tuple[Host, ...], job: str, address: str, state: dict[str, Any], start: int, local_gpus: int) -> list[tuple[list[str], str]]:
    commands = []
    total_workers = len(hosts) * local_gpus
    for host in hosts:
        line = job_line(job, address, state, start + host.base_offset, total_workers)
        for gpu in range(local_gpus):
            remote = f"{settings.cuda_binary} --gpu {gpu} --workers {total_workers}"
            commands.append(([
                "ssh", "-i", str(settings.ssh_keyfile), "-o", "IdentitiesOnly=yes",
                "-o", "BatchMode=yes", "-p", str(host.port), f"{settings.ssh_user}@{host.hostname}", remote,
            ], line))
    return commands


def validate_solution(address: str, state: dict[str, Any], nonce: int) -> tuple[str, int]:
    return verify_candidate(address, nonce, state["challenge"], state["difficulty"])


def dry_run(settings: Settings, args: argparse.Namespace) -> int:
    hosts = parse_hosts(tuple(args.hosts.split(",")) if args.hosts else settings.hosts)
    validate_lane_layout(hosts, args.local_gpus)
    if not args.address:
        raise ValueError("--address is required")
    # Dry-run checks only formatting and lane partitioning; no DNS, SSH, RPC, or signing.
    commands = build_commands(settings, hosts, "dry-run", args.address, {"challenge": "0x" + "00" * 32, "difficulty": 0}, args.start, args.local_gpus)
    print(json.dumps({
        "mode": "dry-run",
        "workers": len(commands),
        "host_count": len(hosts),
        "host_base_offsets": [host.base_offset for host in hosts],
        "nonce_starts": [args.start + host.base_offset + gpu for host in hosts for gpu in range(args.local_gpus)],
        "commands_built": len(commands),
        "network_actions": False,
    }, indent=2))
    return 0


def live(settings: Settings, args: argparse.Namespace) -> int:
    hosts = parse_hosts(tuple(args.hosts.split(",")) if args.hosts else settings.hosts)
    if not settings.ssh_keyfile:
        raise ValueError("SSH_KEYFILE is required for --live")
    if not settings.contract_address or not args.address:
        raise ValueError("CONTRACT_ADDRESS and --address are required for --live")
    state = read_state(settings)
    if state["price"] > settings.max_mint_price_wei:
        raise RuntimeError("chain price exceeds configured cap")
    job = f"job-{int(time.time())}"
    commands = build_commands(settings, hosts, job, args.address, state, args.start, args.local_gpus)
    processes = []
    try:
        for command, _ in commands:
            # Remote execution is intentionally opt-in via --live.
            processes.append(subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True))
        for process, (_, line) in zip(processes, commands):
            assert process.stdin is not None
            process.stdin.write(line + "\n")
            process.stdin.flush()
        for process in processes:
            assert process.stdout is not None
            for output in process.stdout:
                fields = output.strip().split()
                if fields and fields[0] == "solution":
                    values = dict(field.split("=", 1) for field in fields[1:] if "=" in field)
                    nonce = int(values["nonce"], 0)
                    digest, bits = validate_solution(args.address, state, nonce)
                    if digest.removeprefix("0x") != values["hash"].removeprefix("0x") or bits < state["difficulty"]:
                        raise RuntimeError("worker solution failed independent verification")
                    print(json.dumps({"proof": {"address": args.address, "challenge": state["challenge"], "nonce": nonce, "digest": digest, "bits": bits}}, indent=2))
                    return 0
        raise RuntimeError("workers exited without a solution")
    finally:
        for process in processes:
            if process.stdin:
                try:
                    process.stdin.write("quit\n")
                    process.stdin.flush()
                except OSError:
                    pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="allow SSH worker execution; never enabled by default")
    parser.add_argument("--address", help="public wallet address receiving a proof")
    parser.add_argument("--hosts", help="comma-separated host:port:base_offset entries")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--local-gpus", type=int, default=4)
    args = parser.parse_args()
    if args.start < 0 or args.local_gpus < 1:
        raise SystemExit("--start must be nonnegative and --local-gpus must be positive")
    settings = load_settings()
    return live(settings, args) if args.live else dry_run(settings, args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
