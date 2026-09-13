# Hashcats CUDA miner

Native Ethereum Keccak-256 proof-of-work worker for Hashcats, with a local signing coordinator and four remote NVIDIA GPU workers. This is an independently developed community tool, not an official Hashcats release.

## Architecture

The CUDA worker searches `keccak256(address[20] || nonce[32] || previousWork[32] || anchor[32])`, a 116-byte packed message. Nonce values are full big-endian uint256; acceptance is strictly `digest < target`. This is Ethereum Keccak, not standardized SHA3-256.

`remote_runner.py` reads challenge, anchor, personal target and price in one Multicall read, distributes distinct nonce ranges to four persistent CUDA processes, and verifies candidates on CPU. `coordinator.py` runs on the machine holding the wallet key, verifies again, signs `mine(uint256,uint256)`, and verifies receipt Transfer logs and `ownerOf`. Signing keys never go to the GPU host.

## Requirements

- Local Python 3.11+, dependencies in requirements.txt.
- Remote Linux, four NVIDIA CUDA GPUs, Python with requests/eth-utils/eth-abi and a Keccak backend.
- CUDA development toolkit supporting the GPU. Build/test script currently targets `sm_120` (RTX 5090); change the build architecture for other GPUs.
- SSH public-key access with an independently verified host key in known_hosts.
- ETH for both entry price and gas on Robinhood Chain (4663).

## Setup

```sh
python -m venv .venv
# Activate the virtual environment for your shell.
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` locally. It is not loaded automatically: export variables in your shell, or on POSIX use `set -a; . ./.env; set +a`. In PowerShell set `$env:NAME = 'value'`. Never commit the real file.

Set `MINER_ADDRESS` to your address. Put strict hexadecimal private keys, one per nonempty line, in `WALLET_KEY_FILE`; the coordinator selects the key whose derived address matches. Prefer a single dedicated key. File stays local. Do not put a key in CLI arguments, documentation or remote config.

Set SSH_HOST, SSH_PORT, SSH_USER, SSH_KEY_FILE and SSH_KNOWN_HOSTS. Both SSH entry points reject unknown hosts. Prepare REMOTE_DIR on the remote machine and install remote Python dependencies in REMOTE_PYTHON's environment:

```sh
python3 -m pip install requests eth-utils eth-abi 'eth-hash[pycryptodome]'
```

The public example uses a non-routable placeholder host and a zero wallet. `MAX_PRICE_WEI=0` must be deliberately raised to your chosen entry-price ceiling before live mining. `TARGET_MINTS` is the cumulative target in the selected state directory, not an additional amount on every restart.

## Dry-run and tests

```sh
python coordinator.py
python -m pytest -q
python scripts/scan_secrets.py
```

Without `--live`, the coordinator only loads configuration and exits: no SSH, signing, or broadcast. This is a configuration-only dry-run, not a chain simulation.

GPU correctness and benchmark (explicit remote operation, no wallet keys or transactions):

```sh
python gpu_test.py --tune
```

This uploads the CUDA source, builds the binary and checks a deterministic full-width nonce against a CPU Keccak implementation on four GPUs, including equality rejection. The benchmark uses an impossible target so early hits cannot inflate throughput. Use `--no-build` to test the existing remote binary. Do not run a deployment/benchmark over an active mining installation; use a separate directory.

## Live operation

```sh
python -u coordinator.py --live
```

The coordinator uploads only remote_runner.py. The binary must already exist under REMOTE_DIR. Use one coordinator per signing wallet. A local file lock prevents duplicate coordinators sharing STATE_DIR. For background operation use your process manager; keep the local signer machine online.

Outputs under STATE_DIR: mining.jsonl, status.json, receipts.json, individual receipts, pending.json and a lock. A broadcast hash is not a mint. Only receipt success, mint Transfer log and owner verification count toward TARGET_MINTS. Stop conditions include target achieved, insufficient balance, price above cap, ten reverts, unresolved pending transaction, fatal worker failure, and a two-hour session limit.

A pending.json on startup requires manual transaction reconciliation before restarting; do not delete it blindly or send replacements without checking chain state. Gas limit is a conservative 400000; legacy gas price is refreshed before broadcast. These defaults are reference choices, not universal optimal settings.

## Performance and limitations

The original kernel was checked on four RTX 5090 cards. Per-card kernel measurements were approximately 6.3–6.6 GH/s; original live effective aggregate throughput was approximately 25.8 GH/s. These are historical measurements, not promises for your configuration. The sanitized orchestration copy is covered by offline tests; it was not deployed over the original live miner.

For a fixed target T, ideal expected hashes are `2**256 / T`; divide by effective hash/s for expected solution time. Dynamic targets, competing mints, stale work, and submission latency change actual accepted-mint yield. High GPU utilization does not guarantee profitability or rare NFTs.

The remote controller currently polls state and pauses hashing while a candidate is submitted. A lost SSH connection can leave workers alive on some platforms; confirm remote process state and stop only your own orphan workers before restarting. This version does not claim automatic disconnect recovery or production-grade supervision. GPU process count is currently four.

## Configuration

| Variable | Purpose |
|---|---|
| MINER_ADDRESS | Public mining and receiving address |
| WALLET_KEY_FILE | Local key file; never uploaded |
| RPC_URL | HTTPS RPC; avoid logging credential-bearing URLs |
| COLLECTION_ADDRESS | Public collection contract |
| SSH_HOST / SSH_PORT / SSH_USER | Remote SSH endpoint |
| SSH_KEY_FILE / SSH_KNOWN_HOSTS | Local SSH credentials and trusted host keys |
| REMOTE_DIR / REMOTE_PYTHON | Remote deployment directory and Python executable |
| STATE_DIR | Local runtime records and singleton lock |
| TARGET_MINTS | Cumulative verified mint target |
| MAX_PRICE_WEI | Maximum entry payment, excluding gas |

Worker tuning variables CUDA_THREADS and CUDA_BLOCKS_PER_SM are exercised by gpu_test.py. Worker stdin is six fields, **without** a `work` prefix:

```text
address prevWork anchor target startDEC countDEC
result FOUND NONCE DIGEST MILLISECONDS
```

Official mechanism reference: https://hashcats.fun/docs

See SECURITY.md for publication boundaries and credential handling.
