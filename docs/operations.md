# Operations

## CPU-only verification

Install the package and run `python -m pytest -q`. The command-line tool accepts
only explicit public fields, target, and a finite range. It performs no network
calls and does not broadcast transactions.

## CUDA compile and runtime

Compile `cuda/universal_cuda_miner.cu` once with the architecture matching the
GPU. Before sending a job, run the generic local gate:

```sh
python scripts/cuda_preflight.py --worker ./universal_cuda_miner
```

It checks GPU enumeration, `/dev/nvidia-uvm` where applicable, and the worker
`--self-test` on every visible GPU. Any missing device, CUDA initialization
failure, or missing `self_test_passed=true` is a hard stop. Only after this gate
passes should you run one persistent process per GPU. Send jobs through stdin and send `stop`
before a new job or `quit` at shutdown. The worker reports only verified
solutions. A host CPU recomputation protects the output path from a GPU or
packing error.

Use an isolated build/output directory for compile-only verification. Never
replace or restart an occupied miner to test this repository. Runtime GPU
parity is opt-in and should use synthetic public fields and a dedicated device.

## Honest performance reporting

The project does not publish a universal hash-rate number. Any benchmark
report must include GPU model, driver/toolkit, architecture flag, worker
options, protocol, template width, target mode, and whether host transfer time
was included. A compile-only CI result is not a benchmark and is not a runtime
parity result.

## External integration boundary

A caller owns challenge retrieval, authorization, transaction submission, and
receipt handling. Keep credentials and live RPC configuration outside this
repository. If a caller receives a worker candidate, recompute the exact packed
message with the same adapter before any external action.
