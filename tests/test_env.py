import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import check_env


def test_gpu_detected():
    info = check_env.collect()
    assert info["cuda_available"]
    assert info["vram_gb"] >= 7
    assert info["alloc_test"]


def test_python_version():
    assert sys.version_info[:2] == (3, 10)
