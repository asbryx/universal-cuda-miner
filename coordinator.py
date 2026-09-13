import json, time, sys, threading, requests, os
from pathlib import Path
from eth_account import Account
from eth_utils import keccak
from eth_abi import encode
from preflight import (
    ssh,
    ADDR,
    CONTRACT,
    ROOT,
    RPC,
    KEY_FILE,
    REMOTE_DIR,
    REMOTE_PYTHON,
    TARGET_MINTS,
    MAX_PRICE_WEI,
)

sess = requests.Session()
LOG = ROOT / "mining.jsonl"
STATUS = ROOT / "status.json"


def log(x):
    x = {"at": time.time(), **x}
    print(json.dumps(x), flush=True)
    with LOG.open("a") as f:
        f.write(json.dumps(x) + "\n")
    STATUS.write_text(json.dumps(x, indent=2))


def remote_command():
    import shlex

    env = {
        "MINER_ADDRESS": ADDR,
        "RPC_URL": RPC,
        "COLLECTION_ADDRESS": CONTRACT,
        "REMOTE_DIR": REMOTE_DIR,
    }
    return (
        "env "
        + " ".join(k + "=" + shlex.quote(v) for k, v in env.items())
        + " "
        + shlex.quote(REMOTE_PYTHON)
        + " -u "
        + shlex.quote(REMOTE_DIR + "/remote_runner.py")
    )


def rpc(method, params):
    j = sess.post(
        RPC,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        timeout=12,
    ).json()
    if "error" in j:
        raise RuntimeError("RPC rejected request")
    return j["result"]


def read(sig, types=[], args=[]):
    return rpc(
        "eth_call",
        [
            {
                "to": CONTRACT,
                "data": "0x" + (keccak(text=sig)[:4] + encode(types, args)).hex(),
            },
            "latest",
        ],
    )


def main():
    if "--live" not in sys.argv:
        print(
            "Dry-run: configuration loaded; no SSH, signing or broadcast. Use --live to enable."
        )
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    from filelock import FileLock

    lf = FileLock(str(ROOT / "coordinator.lock"))
    lf.acquire(timeout=0)
    accounts = [
        Account.from_key(x.strip())
        for x in KEY_FILE.read_text().splitlines()
        if x.strip()
    ]
    acct = next(x for x in accounts if x.address.lower() == ADDR.lower())
    del accounts
    assert int(rpc("eth_chainId", []), 16) == 4663
    successes = []
    if (ROOT / "receipts.json").exists():
        successes = json.loads((ROOT / "receipts.json").read_text())
    if len(successes) >= TARGET_MINTS:
        log({"type": "already_done", "mints": successes})
        return
    if (ROOT / "pending.json").exists():
        raise RuntimeError("Unresolved pending transaction: reconcile before restart")
    initial = int(rpc("eth_getBalance", [ADDR, "latest"]), 16)
    gasprice = max(int(rpc("eth_gasPrice", []), 16) * 2, 100000000)
    log(
        {
            "type": "starting",
            "wallet": ADDR,
            "balance": initial,
            "gasPrice": gasprice,
            "target": TARGET_MINTS,
        }
    )
    c = ssh()
    sftp = c.open_sftp()
    sftp.put(str(ROOT / "remote_runner.py"), REMOTE_DIR + "/remote_runner.py")
    sftp.close()
    i, o, e = c.exec_command(remote_command(), get_pty=False)

    def errors():
        for line in e:
            log({"type": "remote_stderr", "text": "Remote diagnostic omitted"})

    threading.Thread(target=errors, daemon=True).start()
    start = time.time()
    reverts = 0
    try:
        for line in o:
            x = json.loads(line)
            log(x)
            if time.time() - start > 7200:
                raise RuntimeError("2-hour session limit reached")
            if x["type"] == "fatal":
                raise RuntimeError(str(x))
            if x["type"] != "candidate":
                continue
            job = x["job"]
            n = x["nonce"]
            digest = keccak(
                bytes.fromhex(ADDR[2:])
                + n.to_bytes(32, "big")
                + bytes.fromhex(job["prev"])
                + bytes.fromhex(job["anchor"])
            )
            assert digest.hex() == x["digest"] and int.from_bytes(digest, "big") < int(
                job["target"], 16
            )
            if int(read("prevWork()"), 16) != int(job["prev"], 16):
                log({"type": "stale_candidate"})
                i.write('{"type":"resume"}\n')
                i.flush()
                continue
            price = int(read("mintPrice()"), 16)
            balance = int(rpc("eth_getBalance", [ADDR, "latest"]), 16)
            gasprice = max(int(rpc("eth_gasPrice", []), 16) * 2, 100000000)
            gas = 400000
            if price > MAX_PRICE_WEI:
                raise RuntimeError("Mint price rose beyond configured price cap")
            if balance < price + gas * gasprice:
                raise RuntimeError("Insufficient funds")
            nonce = int(rpc("eth_getTransactionCount", [ADDR, "pending"]), 16)
            data = (
                "0x"
                + (
                    keccak(text="mine(uint256,uint256)")[:4]
                    + encode(["uint256", "uint256"], [n, job["anchorBlock"]])
                ).hex()
            )
            tx = {
                "chainId": 4663,
                "nonce": nonce,
                "to": CONTRACT,
                "value": price,
                "gas": gas,
                "gasPrice": gasprice,
                "data": data,
            }
            signed = acct.sign_transaction(tx)
            txhash = "0x" + signed.hash.hex().removeprefix("0x")
            pending = {"tx": txhash, "nonce": nonce, "price": price}
            (ROOT / "pending.json").write_text(json.dumps(pending))
            log({"type": "broadcasting", **pending})
            try:
                rpc(
                    "eth_sendRawTransaction",
                    ["0x" + signed.raw_transaction.hex().removeprefix("0x")],
                )
            except Exception as exc:
                log(
                    {
                        "type": "broadcast_uncertain",
                        "error": type(exc).__name__,
                        "tx": txhash,
                    }
                )
            receipt = None
            deadline = time.time() + 120
            while time.time() < deadline:
                try:
                    receipt = rpc("eth_getTransactionReceipt", [txhash])
                except Exception:
                    pass
                if receipt:
                    break
                time.sleep(0.25)
            if not receipt:
                raise RuntimeError(
                    "Receipt unresolved; stopping without retry transaction"
                )
            (ROOT / f"receipt-{txhash}.json").write_text(json.dumps(receipt, indent=2))
            if int(receipt["status"], 16) == 1:
                topic = "0x" + keccak(text="Transfer(address,address,uint256)").hex()
                ids = []
                for event in receipt["logs"]:
                    ts = event["topics"]
                    if (
                        event["address"].lower() == CONTRACT.lower()
                        and len(ts) == 4
                        and ts[0].lower() == topic.lower()
                        and int(ts[1], 16) == 0
                        and ts[2][-40:].lower() == ADDR[2:].lower()
                    ):
                        ids.append(int(ts[3], 16))
                assert len(ids) == 1, ids
                owner = read("ownerOf(uint256)", ["uint256"], [ids[0]])
                assert owner[-40:].lower() == ADDR[2:].lower()
                result = {
                    "tokenId": ids[0],
                    "tx": txhash,
                    "owner": ADDR,
                    "price": price,
                    "gasCost": int(receipt["gasUsed"], 16)
                    * int(receipt["effectiveGasPrice"], 16),
                }
                successes.append(result)
                (ROOT / "receipts.json").write_text(json.dumps(successes, indent=2))
                log({"type": "MINT_VERIFIED", "count": len(successes), **result})
            else:
                reverts += 1
                log(
                    {
                        "type": "reverted",
                        "tx": txhash,
                        "gasUsed": int(receipt["gasUsed"], 16),
                    }
                )
            (ROOT / "pending.json").unlink()
            if len(successes) >= TARGET_MINTS:
                break
            if (
                int(rpc("eth_getBalance", [ADDR, "latest"]), 16)
                < int(read("mintPrice()"), 16) + gas * gasprice
            ):
                log({"type": "BUDGET_EXHAUSTED", "mints": successes})
                break
            if reverts >= 10:
                raise RuntimeError("10 reverted transactions; stop for diagnosis")
            i.write('{"type":"resume"}\n')
            i.flush()
    finally:
        try:
            i.write('{"type":"stop"}\n')
            i.flush()
            i.channel.shutdown_write()
        except Exception:
            pass
        for _ in range(30):
            if o.channel.exit_status_ready():
                break
            time.sleep(0.2)
        c.close()
    log(
        {
            "type": "DONE" if len(successes) == TARGET_MINTS else "STOPPED",
            "mints": successes,
            "initialBalance": initial,
            "finalBalance": int(rpc("eth_getBalance", [ADDR, "latest"]), 16),
        }
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log({"type": "STOPPED_ERROR", "error": type(e).__name__})
        raise
