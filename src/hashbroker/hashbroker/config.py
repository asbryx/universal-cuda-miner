"""Configuration loaded from environment; no credential-bearing defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    rpc_urls: tuple[str, ...]
    contract_address: str
    chain_id: int
    wallet_keyfile: Path | None
    max_mint_price_wei: int
    max_total_mint_wei: int
    gas_limit: int
    max_gas_price_wei: int
    hosts: tuple[str, ...]
    cuda_binary: Path
    ssh_user: str
    ssh_keyfile: Path | None


def _csv(name: str, default: str = "") -> tuple[str, ...]:
    return tuple(item.strip() for item in os.getenv(name, default).split(",") if item.strip())


def _int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw, 0)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < 0:
        raise ValueError(f"{name} must not be negative")
    return value


def load_settings() -> Settings:
    """Load settings, intentionally requiring RPC URLs only for live operation."""
    keyfile = os.getenv("WALLET_KEYFILE", "").strip()
    ssh_keyfile = os.getenv("SSH_KEYFILE", "").strip()
    rpc_urls = _csv("RPC_URLS")
    contract_address = os.getenv("CONTRACT_ADDRESS", "0x0000000000000000000000000000000000000000").strip()
    if not (contract_address.startswith("0x") and len(contract_address) == 42):
        raise ValueError("CONTRACT_ADDRESS must be a 20-byte 0x-prefixed address")
    return Settings(
        rpc_urls=rpc_urls,
        contract_address=contract_address,
        chain_id=_int("CHAIN_ID", 1),
        wallet_keyfile=Path(keyfile) if keyfile else None,
        max_mint_price_wei=_int("MAX_MINT_PRICE_WEI", 100_000_000_000_000),
        max_total_mint_wei=_int("MAX_TOTAL_MINT_WEI", 5_000_000_000_000_000),
        gas_limit=_int("GAS_LIMIT", 300_000),
        max_gas_price_wei=_int("MAX_GAS_PRICE_WEI", 1_000_000_000),
        hosts=_csv("HOSTS"),
        cuda_binary=Path(os.getenv("CUDA_BINARY", "./cuda/hashbroker_cuda")),
        ssh_user=os.getenv("SSH_USER", "miner").strip() or "miner",
        ssh_keyfile=Path(ssh_keyfile) if ssh_keyfile else None,
    )
