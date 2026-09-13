import subprocess, sys
from pathlib import Path
import gpu_test
from eth_utils import keccak

ROOT = Path(__file__).resolve().parents[1]


def test_keccak_reference():
    for data in [b"", b"hello", bytes(range(116))]:
        assert gpu_test.keccak256(data) == keccak(data)


def test_dry_run():
    p = subprocess.run(
        [sys.executable, str(ROOT / "coordinator.py")], capture_output=True, text=True
    )
    assert p.returncode == 0, p.stderr
    assert "no SSH, signing or broadcast" in p.stdout


def test_protocol():
    text = (ROOT / "remote_runner.py").read_text()
    assert 'f"work ' not in text
    assert "ms=float(t[4])" in text.replace(" ", "")


def test_no_identity():
    assert gpu_test.ADDRESS == "0000000000000000000000000000000000000001"
