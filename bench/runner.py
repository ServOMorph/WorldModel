import json
import time
from pathlib import Path

from . import scenarios
from .metrics import ResourceSampler, drift_score, fps_stats

RESULTS = Path(__file__).resolve().parent / "results.json"
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def write_video(frames, path, fps=10):
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    h2, w2 = h - h % 2, w - w % 2
    with imageio.get_writer(str(path), fps=fps, codec="libx264", macro_block_size=2) as wr:
        for f in frames:
            wr.append_data(f[:h2, :w2])


def save_result(result, results_path=RESULTS):
    results_path = Path(results_path)
    data = []
    if results_path.exists():
        data = json.loads(results_path.read_text(encoding="utf-8"))
    data = [r for r in data if not (r["adapter"] == result["adapter"] and r["scenario"] == result["scenario"])]
    data.append(result)
    results_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_bench(adapter, scenario_name, results_path=RESULTS, runs_dir=RUNS_DIR, video=True):
    sc = scenarios.get(scenario_name)
    result = {
        "adapter": adapter.name,
        "scenario": scenario_name,
        "status": "ok",
        "error": None,
    }
    frames = []
    durations = []
    with ResourceSampler() as sampler:
        t0 = time.perf_counter()
        try:
            adapter.load()
            result["load_s"] = time.perf_counter() - t0
            frames.append(adapter.reset(sc["seed"]))
            for action in sc["actions"]:
                t = time.perf_counter()
                frame = adapter.step(action)
                durations.append(time.perf_counter() - t)
                frames.append(frame)
        except MemoryError as e:
            result["status"] = "oom"
            result["error"] = str(e)
        except Exception as e:
            result["status"] = "crash"
            result["error"] = f"{type(e).__name__}: {e}"
        finally:
            try:
                adapter.unload()
            except Exception:
                pass
    result["frames"] = len(durations)
    result.update(fps_stats(durations))
    result["ram_peak_gb"] = round(sampler.ram_peak_gb, 3)
    result["vram_peak_gb"] = round(sampler.vram_peak_gb, 3)
    result["drift"] = drift_score(frames[0], frames[-1]) if len(frames) > 1 else None
    if video and frames:
        vp = Path(runs_dir) / f"{adapter.name}_{scenario_name}.mp4"
        write_video(frames, vp)
        result["video"] = str(vp)
    save_result(result, results_path)
    return result
