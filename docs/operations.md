# Operations

## Offline validation

```sh
python -m pytest -q
python scripts/scan_public.py .
python -m compileall -q src scripts tests
```

The test suite uses synthetic keys, addresses, challenges, receipts, and hostnames. It performs no network calls and no SSH process creation.

## Dry-run

```sh
python scripts/mine.py \
  --address 0x1111111111111111111111111111111111111111 \
  --hosts host-a:22:0,host-b:22:4 --start 0 --local-gpus 4
```

Dry-run validates that host base offsets and local GPU indices form disjoint lanes. It does not parse live RPC state, start CUDA, resolve names, open SSH, sign, or broadcast.

## Live mode

Only use after reviewing source and local `.env`:

```sh
python scripts/mine.py --live \
  --address 0x1111111111111111111111111111111111111111 \
  --hosts host-a:22:0,host-b:22:4 --start 0 --local-gpus 4
```

Live mode requires explicit `RPC_URLS`, `CONTRACT_ADDRESS`, `SSH_KEYFILE`, and other caps. It is intentionally not exercised by CI. Stop the run using the worker's normal stdin shutdown path or the operator's process supervisor; never use live mode against an unreviewed configuration.

## Recovery

When a process stops after a broadcast, first obtain transaction receipts through a trusted, separately configured RPC adapter. Store the receipts locally, then run `scripts/reconstruct_journal.py`. The reconstruction checks exact mint logs and is safe to repeat. Do not submit another transaction merely because an old journal says `confirmed` without token IDs or spend.
