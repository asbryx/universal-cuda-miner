# CUDA worker interface

`cuda/hashbroker_cuda.cu` is the GPU proof-of-work worker. It performs SHA-256 only. It has no RPC, wallet, signing, transaction, or credential handling.

## Build

```sh
nvcc -O3 -arch=sm_89 -std=c++17 -lineinfo -o hashbroker_cuda cuda/hashbroker_cuda.cu
```

Run one process per local GPU. The worker accepts a persistent stdin protocol:

```text
job JOB_ID ADDRESS40HEX CHALLENGE64HEX DIFFICULTY START_NONCE [RANGE_STEP]
stop
quit
```

`RANGE_STEP` defaults to `--workers`. Local GPU `g` tests `START_NONCE + g` and advances by `RANGE_STEP`; this makes lanes disjoint when all workers receive the same job start.

## Multi-host lane rule

The controller's `HOSTS` format is `host:port:base_offset`. For two hosts with four local GPUs each, use synthetic configuration such as:

```text
HOSTS=host-a:22:0,host-b:22:4
```

Host A therefore owns global lanes 0..3 and host B owns lanes 4..7. The worker still adds its local GPU ID. The base offset is part of the public controller's job line and is not a change to any live machine.

This explicit offset corrects the known operational failure mode in which both hosts were given the same start and step and consequently duplicated lanes 0..3.

## Output

The worker emits flushed records such as `ready`, `protocol_ready`, `job_started`, `stats`, `solution`, `job_stopped`, and `error`. A solution is recomputed by the worker's independent CPU SHA-256 implementation before output. The controller verifies it again before downstream actions.

Run `--self-test` for standard SHA-256 vectors and the fixed 84-byte protocol message. CUDA compilation is optional for the offline Python test suite.
