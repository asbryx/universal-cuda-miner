"""Configurable, protocol-aware CUDA proof-of-work primitives.

The package contains no chain, wallet, SSH, or credential handling.  A protocol
adapter defines the exact packed message and the CPU verifier; the CUDA worker
consumes the same template and runtime nonce metadata.
"""

from .layouts import (
    HASHBROKER84,
    HASHCATS116,
    PRSPCT84,
    ProtocolLayout,
    build_job,
    digest_candidate,
    get_layout,
    leading_zero_bits,
    render_candidate,
    serialize_job,
    verify_candidate,
)

__all__ = [
    "HASHBROKER84",
    "HASHCATS116",
    "PRSPCT84",
    "ProtocolLayout",
    "build_job",
    "digest_candidate",
    "get_layout",
    "leading_zero_bits",
    "verify_candidate",
]

__version__ = "1.0.0"
