import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.runner import run_bench
from worldmodel.adapters.diamond_atari import DiamondAtariAdapter


def test_diamond_atari_smoke(tmp_path):
    r = run_bench(DiamondAtariAdapter(), "smoke", tmp_path / "r.json", tmp_path, video=False)
    assert r["status"] == "ok", r["error"]
    assert r["frames"] == 10


def test_diamond_csgo_smoke(tmp_path):
    from worldmodel.adapters.diamond_csgo import DiamondCsgoAdapter

    r = run_bench(DiamondCsgoAdapter(), "smoke", tmp_path / "r.json", tmp_path, video=False)
    assert r["status"] == "ok", r["error"]
    assert r["frames"] == 10


def test_open_oasis_smoke(tmp_path):
    from worldmodel.adapters.open_oasis import OpenOasisAdapter

    r = run_bench(OpenOasisAdapter(), "smoke", tmp_path / "r.json", tmp_path, video=False)
    assert r["status"] == "ok", r["error"]
    assert r["frames"] == 10
