# Universal CUDA PoW Miner

Reusable, sanitized CUDA proof-of-work toolkit for launching new miners without rebuilding the operational stack from zero.

## What it provides

- Optimized CUDA worker templates for Ethereum Keccak and SHA-256-style lanes.
- Full-width uint256 nonce ranges with disjoint multi-GPU work allocation.
- Local-only signing and receipt verification; private keys never reach GPU hosts.
- Configurable RPC, SSH, fee, retry, price, target, and cumulative mint limits.
- Dry-run defaults, deterministic offline tests, GPU self-test, benchmark, journal recovery, and secret scanning.
- Persistent worker protocol suitable for a remote supervisor.

## Layout

- `cuda/` — CUDA worker templates.
- `scripts/` — configurable mining, signing, remote-runner, preflight, recovery, and secret-scan tools.
- `tests/` — offline tests only.
- `docs/` — configuration, CUDA protocol, and operations.
- `SECURITY.md` — publication and credential boundary.

## Quick start

```text
python scripts/preflight.py
python -m pytest -q
python scripts/scan_public.py
```

Copy `.env.example` to a local `.env` and fill it only locally. Never commit `.env`, wallet keys, SSH private keys, RPC URLs containing credentials, runtime journals, or compiled binaries.

For a new protocol, adapt a worker's public input schema and the verifier/coordinator interface. The default is configuration-only. Live network access and transaction broadcast require an explicit live flag in the relevant runner.

## Security model

Hashing hosts receive public challenge data only. Signing happens on the operator machine. Every candidate is recomputed locally, simulated before broadcast, and counted only after a successful receipt plus protocol-specific ownership/event verification.

This repository contains no operational wallet, host, RPC credential, proxy, runtime state, or user-specific destination.

## License

MIT
