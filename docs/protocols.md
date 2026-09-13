# Protocol adapter contract

A `ProtocolLayout` is the single source of truth for CPU packing and GPU job
metadata. It defines:

- algorithm (`sha256` or Ethereum `keccak256`);
- complete packed message width;
- ordered public fields;
- byte offset and width of the big-endian nonce field;
- strict uint256 target comparison.

The template returned by `build_job()` is packed with nonce zero. The CUDA
worker receives that template and substitutes a runtime nonce at
`nonce_offset`, so changing jobs does not generate or compile a kernel.

## Reference layouts

- `hashbroker84`: address 20, 24 fixed zero bytes, nonce uint64, challenge 32;
  SHA-256 of exactly 84 bytes.
- `hashcats116`: address 20, nonce uint256, previous work 32, anchor 32;
  Ethereum Keccak-256 of exactly 116 bytes.
- `prspct84`: seed 32, address 20, nonce uint256; Ethereum Keccak-256 of exactly
  84 bytes.

Do not infer a layout from field names or word alignment. Tests assert exact byte
offsets and independent reference digests.

## Adding a layout

1. Add a `ProtocolLayout` with explicit field widths and nonce metadata.
2. Add a namespaced adapter under `src/universal_cuda_miner/adapters/`.
3. Add an independent known-vector test and boundary test.
4. Confirm the CUDA template can represent the message width and algorithm.
5. Update this document and README without making performance or protocol
   integration claims that have not been exercised.

The package has no network or wallet interface by design.
```
