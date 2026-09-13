# universal-cuda-miner

A small, protocol-configurable proof-of-work toolkit. It separates **message
layout**, **CPU verification**, and the **persistent CUDA worker** so a new
protocol does not require copying a miner tree or regenerating a kernel for
every job.

This repository is an offline hashing/verification worker. It does not contain
RPC, wallet, signing, SSH, transaction, or deployment automation.

## Supported layouts

| Name | Algorithm | Packed message | Nonce field |
|---|---|---|---|
| `hashbroker84` | SHA-256 | `address[20] || zero[24] || nonce[8] || challenge[32]` | bytes 44..51, big-endian uint64 |
| `hashcats116` | Ethereum Keccak-256 | `address[20] || nonce[32] || prev[32] || anchor[32]` | bytes 20..51, big-endian uint256 |
| `prspct84` | Ethereum Keccak-256 | `seed[32] || address[20] || nonce[32]` | bytes 52..83, big-endian uint256 |

`keccak256` here means Ethereum Keccak-256 (domain suffix `0x01`), not
standardized SHA3-256. All targets use the strict comparison `digest < target`;
equality is rejected.

The Python adapters are namespaced as:

```text
universal_cuda_miner.adapters.hashbroker
universal_cuda_miner.adapters.hashcats
universal_cuda_miner.adapters.prspct
```

## Install and verify

Python 3.10+ is required. The only runtime dependency is `eth-hash` for the
Ethereum Keccak primitive.

```sh
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/scan_public.py .
```

The package exposes `universal-cuda-mine` after installation. It searches a
finite CPU range and emits a JSON result only after independent CPU
verification:

```sh
universal-cuda-mine \
  --protocol hashbroker84 \
  --fields '{"address":"0x1212121212121212121212121212121212121212","challenge":"0x3434343434343434343434343434343434343434343434343434343434343434"}' \
  --target 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff \
  --start 0 --count 1
```

The CLI is deterministic and offline. A nonzero exit status means that the
finite range had no candidate or input validation failed.

## CUDA worker

`cuda/universal_cuda_miner.cu` is one persistent worker entrypoint. It compiles
once and receives layout metadata at runtime: algorithm, message width, nonce
offset, nonce width, zeroed template, target, and nonce range. It supports the
three layouts above without generating or compiling a protocol-specific kernel.
Every GPU result is recomputed by an independent host implementation before it
is emitted.

Compile-only example (choose an architecture supported by the installed toolkit
and target GPU):

```sh
nvcc -O3 -std=c++17 -arch=sm_89 cuda/universal_cuda_miner.cu -o universal_cuda_miner
```

The worker remains resident and accepts one job per line:

```text
job ID PROTOCOL ALGORITHM NONCE_OFFSET NONCE_WIDTH TEMPLATE_HEX TARGET_HEX START_HEX COUNT STEP
stop
quit
```

For example, the Python `Job` object provides the exact template, offset, and
width needed by this protocol. The public source does not claim a particular
hash rate: throughput depends on GPU architecture, toolkit, launch settings,
and protocol. Use a dedicated, non-production device and report measured
hashes/second with the architecture and command line.

No GPU benchmark is run by the default test suite. Do not point a compile or
benchmark run at an occupied mining installation.

## Repository layout

```text
src/universal_cuda_miner/       one package and common layout engine
src/universal_cuda_miner/adapters/ named protocol adapters
scripts/mine.py                 checkout CLI wrapper
cuda/universal_cuda_miner.cu   persistent runtime-configurable CUDA worker
tests/                          non-tautological CPU and package tests
docs/                           protocol and operations details
```

There are intentionally no copied root package trees and no protocol-specific
root-script duplicates.

## Scope and limitations

- The worker is a proof-of-work engine, not a complete chain integration.
- Protocol field semantics, target acquisition, submission, and transaction
  handling are intentionally outside this repository.
- The CUDA path is compile-tested in CI when the toolkit job is available; a
  CUDA compile does not prove a particular GPU's runtime correctness.
- CUDA runtime parity requires an NVIDIA device and is opt-in. CPU tests cover
  the packed bytes, algorithm choice, target boundary, nonce placement, and
  adapter behavior.

## Security

See [SECURITY.md](SECURITY.md). Never add keys, credentials, live RPC URLs,
wallet inventories, runtime logs, or machine-specific paths. The public scan is
intentionally conservative and prints finding names, never matching values.

## License

MIT
