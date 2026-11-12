import threading
import time

import numpy as np
import psutil


def similarity(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / den)


def drift_score(first, last):
    return 1.0 - max(0.0, similarity(first, last))


class ResourceSampler:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.ram_peak_gb = 0.0
        self.vram_peak_gb = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process()

    def _vram_used(self):
        try:
            import torch

            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return (total - free) / 1024**3
        except Exception:
            pass
        return 0.0

    def _sample(self):
        self.ram_peak_gb = max(self.ram_peak_gb, self._proc.memory_info().rss / 1024**3)
        self.vram_peak_gb = max(self.vram_peak_gb, self._vram_used())

    def _loop(self):
        while not self._stop.is_set():
            self._sample()
            time.sleep(self.interval)

    def __enter__(self):
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._sample()


def fps_stats(durations):
    if not durations:
        return {"fps_mean": 0.0, "fps_p95": 0.0}
    d = np.asarray(durations)
    return {
        "fps_mean": float(1.0 / d.mean()),
        "fps_p95": float(1.0 / np.percentile(d, 95)),
    }
