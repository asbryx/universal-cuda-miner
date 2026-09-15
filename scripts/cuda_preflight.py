#!/usr/bin/env python3
"""Fail-closed CUDA host and worker preflight.

This script accepts only a local worker path and public runtime options. It does
not read wallets, RPC URLs, credentials, or deployment configuration.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def run_checked(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run one preflight command and convert failures into a short hard stop."""
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )
    except FileNotFoundError as exc:
        raise SystemExit(f"preflight failed: command not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise SystemExit(f"preflight failed ({exc.returncode}): {' '.join(command)}\n{detail}") from exc


def gpu_indices() -> list[str]:
    listing = run_checked(["nvidia-smi", "-L"])
    if not listing.stdout.strip():
        raise SystemExit("preflight failed: nvidia-smi returned no visible GPUs")
    queried = run_checked(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]
    )
    indices = [line.strip() for line in queried.stdout.splitlines() if line.strip().isdigit()]
    if not indices:
        raise SystemExit("preflight failed: could not enumerate GPU indices")
    print(listing.stdout.strip())
    return indices


def check_uvm() -> None:
    """Validate the Linux UVM device instead of trusting nvidia-smi alone."""
    if os.name == "nt":
        return
    device = Path("/dev/nvidia-uvm")
    if not device.exists():
        raise SystemExit("preflight failed: /dev/nvidia-uvm is missing")
    try:
        descriptor = os.open(device, os.O_RDWR)
    except OSError as exc:
        raise SystemExit(f"preflight failed: cannot open /dev/nvidia-uvm: {exc}") from exc
    else:
        os.close(descriptor)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, help="local compiled universal_cuda_miner path")
    args = parser.parse_args(argv)
    worker = Path(args.worker)
    if not worker.is_file():
        raise SystemExit(f"preflight failed: worker does not exist: {worker}")

    check_uvm()
    indices = gpu_indices()
    for index in indices:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = index
        result = run_checked(
            [str(worker), "--gpu", "0", "--self-test"],
            env=environment,
        )
        output = (result.stdout or "") + (result.stderr or "")
        print(output, end="")
        if "self_test_passed=true" not in output:
            raise SystemExit(
                f"preflight failed: GPU {index} self-test did not report self_test_passed=true"
            )
        print(f"cuda_preflight_passed gpu={index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
