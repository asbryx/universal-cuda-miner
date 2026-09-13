import sys, os, json, time, threading, subprocess, queue, requests
from eth_utils import keccak
from eth_abi import encode, decode

A = os.environ["MINER_ADDRESS"]
C = os.environ.get("COLLECTION_ADDRESS", "0xCA75DF55Cc9C476DB27a7375D1fc8E794cf80721")
RPC = os.environ["RPC_URL"]
REMOTE_DIR = os.environ.get("REMOTE_DIR", "/opt/hashcats")
stop = threading.Event()
gate = threading.Event()
gate.set()
lock = threading.Lock()
emitlock = threading.Lock()
state = {}
stats = [0] * 4
rates = [0.0] * 4
children = []


def emit(x):
    with emitlock:
        print(json.dumps(x), flush=True)


s = requests.Session()


def rpc(method, params):
    j = s.post(
        RPC,
        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
        timeout=8,
    ).json()
    if "error" in j:
        raise RuntimeError("RPC rejected request")
    return j["result"]


def snapshot():
    specs = [
        ("prevWork()", [], []),
        ("currentAnchor()", [], []),
        ("targetFor(address)", ["address"], [A]),
        ("mintPrice()", [], []),
        ("totalMinted()", [], []),
    ]
    calls = [
        (C, False, keccak(text=sig)[:4] + encode(types, args))
        for sig, types, args in specs
    ]
    data = keccak(text="aggregate3((address,bool,bytes)[])")[:4] + encode(
        ["(address,bool,bytes)[]"], [calls]
    )
    raw = rpc(
        "eth_call",
        [
            {
                "to": "0xcA11bde05977b3631167028862bE2a173976CA11",
                "data": "0x" + data.hex(),
            },
            "latest",
        ],
    )
    out = decode(["(bool,bytes)[]"], bytes.fromhex(raw[2:]))[0]
    assert all(x[0] for x in out)
    prev = int.from_bytes(out[0][1], "big")
    ab, anchor = decode(["uint256", "bytes32"], out[1][1])
    target = int.from_bytes(out[2][1], "big")
    return dict(
        prev=f"{prev:064x}",
        anchor=anchor.hex(),
        anchorBlock=ab,
        target=f"{target:064x}",
        price=int.from_bytes(out[3][1], "big"),
        total=int.from_bytes(out[4][1], "big"),
        at=time.time(),
    )


def watcher():
    global state
    while not stop.is_set():
        try:
            state = snapshot()
        except Exception as e:
            emit({"type": "watch_error", "error": type(e).__name__})
        stop.wait(0.10)


def control():
    for line in sys.stdin:
        try:
            x = json.loads(line)
            if x.get("type") == "stop":
                stop.set()
                break
            if x.get("type") == "resume":
                gate.set()
        except Exception:
            pass
    stop.set()


def worker(idx):
    env = dict(os.environ, CUDA_DEVICE=str(idx))
    p = subprocess.Popen(
        [os.path.join(REMOTE_DIR, "hashcats-cuda")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
        env=env,
    )
    children.append(p)
    start = (idx + 1) << 60
    count = 33554432
    try:
        while not stop.is_set():
            if not gate.wait(0.1):
                continue
            job = state.copy()
            if not job or time.time() - job["at"] > 2:
                stop.wait(0.05)
                continue
            p.stdin.write(
                f"{A} {job['prev']} {job['anchor']} {job['target']} {start} {count}\n"
            )
            p.stdin.flush()
            line = p.stdout.readline().strip()
            t = line.split()
            if len(t) < 5 or t[0] != "result":
                raise RuntimeError("bad CUDA output " + line)
            found = int(t[1])
            ms = float(t[4])
            start += count
            if not found:
                stats[idx] += count
                rates[idx] = count / max(ms, 0.001) * 1000
            if found:
                nonce = int(t[2])
                digest = keccak(
                    bytes.fromhex(A[2:])
                    + nonce.to_bytes(32, "big")
                    + bytes.fromhex(job["prev"])
                    + bytes.fromhex(job["anchor"])
                ).hex()
                if digest != t[3].removeprefix("0x") or int(digest, 16) >= int(
                    job["target"], 16
                ):
                    raise RuntimeError("GPU HASH MISMATCH")
                with lock:
                    if gate.is_set() and state.get("prev") == job["prev"]:
                        gate.clear()
                        emit(
                            dict(
                                type="candidate",
                                gpu=idx,
                                nonce=nonce,
                                digest=digest,
                                job=job,
                            )
                        )
    except Exception as e:
        emit({"type": "fatal", "gpu": idx, "error": type(e).__name__})
        stop.set()
    finally:
        p.terminate()
        p.wait()


threading.Thread(target=control, daemon=True).start()
threading.Thread(target=watcher, daemon=True).start()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
for t in threads:
    t.start()
try:
    while not stop.wait(10):
        emit(
            {
                "type": "stats",
                "hashrates": rates,
                "hashes": stats,
                "state": state,
                "paused": not gate.is_set(),
            }
        )
finally:
    stop.set()
    gate.set()
    for t in threads:
        t.join(5)
    emit({"type": "stopped"})
