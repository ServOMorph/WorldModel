import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from bench import scenarios
from bench.metrics import drift_score, fps_stats, similarity
from bench.runner import run_bench
from worldmodel.adapters.base import WorldModelAdapter
from worldmodel.adapters.dummy import DummyAdapter


def test_adapter_is_abstract():
    with pytest.raises(TypeError):
        WorldModelAdapter()


def test_dummy_deterministic():
    outs = []
    for _ in range(2):
        a = DummyAdapter()
        a.load()
        a.reset(seed=5)
        outs.append(a.step("left"))
    assert np.array_equal(outs[0], outs[1])


def test_dummy_requires_load():
    with pytest.raises(RuntimeError):
        DummyAdapter().reset(seed=0)


def test_scenarios_locked():
    assert len(scenarios.get("smoke")["actions"]) == 10
    with pytest.raises(KeyError):
        scenarios.get("nope")


def test_similarity_and_drift():
    a = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert similarity(a, a) == pytest.approx(1.0)
    assert drift_score(a, a) == pytest.approx(0.0, abs=1e-6)
    b = np.random.default_rng(1).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert drift_score(a, b) > 0.5


def test_fps_stats():
    s = fps_stats([0.1] * 10)
    assert s["fps_mean"] == pytest.approx(10.0)
    assert s["fps_p95"] == pytest.approx(10.0)


def test_run_bench_smoke(tmp_path):
    res = tmp_path / "results.json"
    r = run_bench(DummyAdapter(), "smoke", results_path=res, runs_dir=tmp_path)
    assert r["status"] == "ok"
    assert r["frames"] == 10
    assert r["fps_mean"] > 0
    assert r["ram_peak_gb"] > 0
    assert Path(r["video"]).exists() and Path(r["video"]).stat().st_size > 0
    assert json.loads(res.read_text())[0]["adapter"] == "dummy"


def test_run_bench_drift_detected(tmp_path):
    res = tmp_path / "results.json"
    low = run_bench(DummyAdapter(drift=0.0), "short", res, tmp_path, video=False)
    high = run_bench(DummyAdapter(drift=0.01), "short", res, tmp_path, video=False)
    assert high["drift"] > low["drift"]


def test_run_bench_oom_recorded(tmp_path):
    res = tmp_path / "results.json"
    r = run_bench(DummyAdapter(fail_at=3), "smoke", res, tmp_path, video=False)
    assert r["status"] == "oom"
    assert r["frames"] == 3


def test_results_overwrite_same_key(tmp_path):
    res = tmp_path / "results.json"
    run_bench(DummyAdapter(), "smoke", res, tmp_path, video=False)
    run_bench(DummyAdapter(), "smoke", res, tmp_path, video=False)
    assert len(json.loads(res.read_text())) == 1
