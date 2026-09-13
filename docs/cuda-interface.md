# CUDA interface

The single source is `cuda/universal_cuda_miner.cu`.

```sh
nvcc -O3 -std=c++17 -arch=sm_89 cuda/universal_cuda_miner.cu -o universal_cuda_miner
```

Use an architecture appropriate to the target GPU. The worker is persistent:

```text
ready gpu=0 ...
protocol_ready commands=job,stop,quit
job ID PROTOCOL ALGORITHM NONCE_OFFSET NONCE_WIDTH TEMPLATE_HEX TARGET_HEX START_HEX COUNT STEP
solution nonce=0x... digest=0x... verified=true
stop
quit
```

`TEMPLATE_HEX` is the complete packed message with zero bytes in the nonce
field. `NONCE_OFFSET` and `NONCE_WIDTH` are byte offsets, not 32-bit word
indexes. `START_HEX` is a uint256; `COUNT` and `STEP` are decimal uint64 values.
The candidate sequence is `start + i*step` for `0 <= i < count`.

`ALGORITHM=sha256` uses SHA-256. `ALGORITHM=keccak256` uses Ethereum
Keccak-256. The target comparison is strict digest `< TARGET_HEX`; equality is
not accepted.

The source allocates the CUDA worker once and passes layout metadata with every
job. It does not rebuild a kernel for a new protocol or job. It also performs an
independent host hash and target check before printing a solution.
