# HashBroker CUDA tools

A small public reference implementation for HashBroker proof-of-work lanes, an offline-safe transaction signer, receipt-accounting recovery, and a CUDA worker. The Python tools are configurable and default to **dry-run**. No credentials, machine inventories, wallet profiles, host addresses, or runtime journals are included.

## What is included

- `cuda/hashbroker_cuda.cu`: standalone SHA-256 worker. It has no RPC, wallet, signing, or transaction code.
- `scripts/mine.py`: configurable supervisor with an explicit `--live` gate. The default builds and validates lane assignments without DNS, RPC, SSH, signing, or broadcast.
- `scripts/sign.py`: signs a caller-supplied transaction using one strict local key file; it does not send transactions.
- `scripts/reconstruct_journal.py`: deterministically rebuilds token IDs and spend from receipts, deduping transaction hashes so stale `confirmed` markers cannot silently omit accounting.
- `src/hashbroker/`: offline protocol, configuration, and receipt primitives.
- `tests/`: real offline tests only; there are no controller or remote-host fixtures.

## Quick start

```sh
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# POSIX shell: . .venv/bin/activate
python -m pip install -e '.[test]'
python scripts/mine.py --address 0x1111111111111111111111111111111111111111 \
  --hosts host-a:22:0,host-b:22:4 --start 0 --local-gpus 4
python -m pytest -q
python scripts/scan_public.py .
```

The dry-run output must show disjoint nonce starts `0..7`, two host offsets `[0, 4]`, and `network_actions: false`.

## Configuration and live boundary

Copy `.env.example` to `.env` and replace placeholders locally. The `.env` file is ignored. See `docs/configuration.md` for all settings.

`--live` is never implied. It is the only mode that may resolve configured RPC endpoints, start configured SSH workers, or process a live proof. Review the source and local configuration before using it. The public repository does not contain any operational hosts, SSH key, wallet key, RPC credential, or fund-management configuration.

The supervisor's host syntax is `host:port:base_offset`. With four local GPUs per host, host A uses base `0` and host B uses base `4`. The CUDA worker adds its local GPU index, yielding global lanes `0..7`. This explicitly avoids the known duplicate-lane failure where both hosts used the same start and step.

## Signing without broadcasting

The signer requires exactly one local line in `WALLET_KEYFILE` and writes a signed payload locally or to stdout. It never calls RPC and never broadcasts:

```sh
python scripts/sign.py --keyfile /path/to/local/key \
  --chain-id 1 --nonce 0 --to 0x1111111111111111111111111111111111111111 \
  --data 0x --value 0 --gas 300000 --gas-price 1
```

Use a dedicated test key for development. Never paste a private key into a command, issue, log, CI variable, or repository.

## Receipt recovery

A journal is not accounting evidence by itself. For each confirmed record, `reconstruct_journal` checks the receipt's success status, exact contract, ERC-721 `Transfer` topic, zero sender, and recipient. It deduplicates transaction hashes and recomputes both token IDs and spend. Re-running it is idempotent.

```sh
python scripts/reconstruct_journal.py local-journal.json \
  --contract 0x1111111111111111111111111111111111111111 \
  --output repaired-journal.json
```

The command is offline: supply a local copy of receipts. A future network adapter must fetch receipts through the configured RPC layer and then use the same reconstruction function; it must not trust a stale status marker.

## CUDA worker

See `docs/cuda-interface.md` for build flags, stdin protocol, output records, self-test, and multi-host lane offsets. CUDA compilation is optional for the Python test suite.

## Security model

- No secret-bearing defaults; placeholders are synthetic and clearly labeled.
- RPC errors expose only exception types, never full key-bearing URLs.
- Wallet keys are read from a strict one-line file and are never printed.
- Dry-run is the default; live side effects require an explicit flag.
- `.gitignore` covers environment files, keys, wallets, addresses, hosts, proxies, inventories, logs, journals, results, caches, and binaries.
- `scripts/scan_public.py` is deterministic and prints file names/counts only.

Read `SECURITY.md`, `docs/configuration.md`, and `docs/cuda-interface.md` before adapting the tooling.

## License

MIT. See `LICENSE`.
