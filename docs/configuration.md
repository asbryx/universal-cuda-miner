# Configuration

This package is intentionally offline. It has no RPC, wallet, SSH, signing,
or deployment configuration. The only inputs are public protocol fields, a
strict target, and an explicit finite nonce range.

The Python API is the configuration boundary:

```python
from universal_cuda_miner import build_job, get_layout

job = build_job(
    get_layout("hashcats116"),
    {"address": "0x...", "prev": "0x...", "anchor": "0x..."},
    target=2**256 - 1,
    start=0,
    count=1_000_000,
    step=1,
)
```

`job.template`, `job.nonce_offset`, and `job.nonce_width` are passed to the
persistent CUDA worker. Keep external chain state and credentials in a separate
caller, and recompute any returned candidate with the same `ProtocolLayout`.
Never commit live keys, RPC URLs, hostnames, or runtime state.

The `.env.example` file is documentation-only; the package does not load it.
See [protocols.md](protocols.md) for exact field layouts.

---

## Generated artifacts

Editable installs may create `src/*.egg-info`, which is ignored and must not be
published. CUDA outputs and benchmark reports are also ignored.
