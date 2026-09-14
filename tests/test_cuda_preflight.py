from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_cuda_preflight_requires_driver_and_runtime_checks() -> None:
    source = (ROOT / "scripts" / "cuda_preflight.py").read_text(encoding="utf-8")
    assert "nvidia-smi" in source
    assert "/dev/nvidia-uvm" in source
    assert "--self-test" in source
    assert "self_test_passed=true" in source


def test_cuda_preflight_fails_closed_on_nonzero_self_test() -> None:
    source = (ROOT / "scripts" / "cuda_preflight.py").read_text(encoding="utf-8")
    assert "check=True" in source
    assert "raise SystemExit" in source


def test_operations_require_preflight_before_jobs() -> None:
    docs = (ROOT / "docs" / "operations.md").read_text(encoding="utf-8")
    assert "cuda_preflight.py" in docs
    assert "Before sending a job" in docs
