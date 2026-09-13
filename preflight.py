import os
from pathlib import Path
import paramiko

ROOT = Path(os.environ.get("STATE_DIR", "state")).resolve()
ADDR = os.environ.get("MINER_ADDRESS", "0x0000000000000000000000000000000000000000")
CONTRACT = os.environ.get(
    "COLLECTION_ADDRESS", "0xCA75DF55Cc9C476DB27a7375D1fc8E794cf80721"
)
RPC = os.environ.get("RPC_URL", "https://rpc.mainnet.chain.robinhood.com")
KEY_FILE = Path(os.environ.get("WALLET_KEY_FILE", "secrets/wallet.key"))
REMOTE_DIR = os.environ.get("REMOTE_DIR", "/opt/hashcats")
REMOTE_PYTHON = os.environ.get("REMOTE_PYTHON", "python3")
TARGET_MINTS = int(os.environ.get("TARGET_MINTS", "1"))
MAX_PRICE_WEI = int(os.environ.get("MAX_PRICE_WEI", "0"))


def ssh():
    c = paramiko.SSHClient()
    c.load_system_host_keys()
    if os.environ.get("SSH_KNOWN_HOSTS"):
        c.load_host_keys(os.environ["SSH_KNOWN_HOSTS"])
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect(
        os.environ["SSH_HOST"],
        port=int(os.environ.get("SSH_PORT", "22")),
        username=os.environ["SSH_USER"],
        key_filename=os.environ["SSH_KEY_FILE"],
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
    )
    c.get_transport().set_keepalive(15)
    return c
