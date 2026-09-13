# Security and publication boundary

This tree is a separate sanitized copy. No real wallet keys, SSH keys, operator wallet addresses, SSH endpoints, local machine paths, session logs or receipts belong in Git. Only deliberately public protocol addresses and synthetic test vectors are retained.

Store credentials outside the repository or in ignored secrets/. Keep STATE_DIR ignored. Never publish runtime output. The signing key stays local; remote workers receive public work only. Protect the local machine and remote host. SSH host keys must be independently verified before adding to known_hosts.

Default coordinator invocation does not connect or broadcast. --live explicitly enables paid transactions. Set an explicit price cap and keep adequate gas balance. An unknown pending receipt blocks restart. Reverted calls consume gas, but their reverted msg.value is rolled back.

Operational RPC exceptions are reduced to generic errors; do not add raw URLs, request headers, environments or private key values to logs. Avoid supplying secret RPC credentials to an untrusted compute provider: RPC_URL is sent to that host.

The scanner catches common credential formats, not all possible secrets. Pair it with manual staged-tree review and exact known-value comparison. Test vectors are public deterministic data, not funded keys. CI runs offline CPU checks only and never connects to a GPU or wallet.

Report suspected credential exposure privately to the repository owner; do not post secret material in an issue.
