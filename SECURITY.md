# Security

This repository contains protocol packing, CPU verification, tests, and a
credential-free CUDA worker only.

Never commit:

- private keys, seed phrases, wallet/profile files, or address inventories;
- API tokens, credential-bearing RPC URLs, SSH keys, or host inventories;
- runtime logs, journals, receipts, benchmark output, or compiled binaries;
- real operator-specific destinations or machine-local paths.

The worker receives public protocol inputs only. It has no RPC, wallet, signing,
SSH, transaction, or network implementation. Keep those concerns in a separate
caller and independently verify a candidate with the exact protocol adapter
before any external action.

Report suspected security issues privately to the repository owner rather than
including secrets in an issue or pull request.
