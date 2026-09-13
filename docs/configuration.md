# Configuration

Configuration is environment-driven. Start from `.env.example`, keep the local `.env` untracked, and use synthetic values for dry runs.

- `RPC_URLS`: comma-separated JSON-RPC endpoints. Required only for `--live`; errors expose endpoint types, not URLs.
- `CONTRACT_ADDRESS`, `CHAIN_ID`: protocol target and chain identifier.
- `WALLET_KEYFILE`: a local file containing exactly one `0x`-prefixed 32-byte key. Never put keys in environment snapshots or source.
- `MAX_MINT_PRICE_WEI`, `MAX_TOTAL_MINT_WEI`, `GAS_LIMIT`, `MAX_GAS_PRICE_WEI`: fail-closed transaction caps.
- `HOSTS`: comma-separated `host:port:base_offset` entries. Use base offsets 0 and 4 for two four-GPU hosts.
- `CUDA_BINARY`, `SSH_USER`, `SSH_KEYFILE`: local operator paths, never committed.

The CLI defaults to dry-run and does not resolve hosts, contact RPC, spawn SSH, sign, or broadcast. Pass `--live` explicitly only after reviewing the configuration and accepting those side effects.
