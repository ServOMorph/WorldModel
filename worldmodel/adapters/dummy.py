import numpy as np

from .base import WorldModelAdapter


class DummyAdapter(WorldModelAdapter):
    name = "dummy"
    family = "generic"
    actions = ("noop", "left", "right", "up", "down")

    def __init__(self, size=(64, 64), drift=0.0, fail_at=None):
        self.size = size
        self.drift = drift
        self.fail_at = fail_at
        self.loaded = False
        self._t = 0
        self._rng = None
        self._base = None

    def load(self):
        self.loaded = True

    def reset(self, seed, image=None, prompt=None):
        if not self.loaded:
            raise RuntimeError("modele non charge")
        self._rng = np.random.default_rng(seed)
        self._t = 0
        h, w = self.size
        self._base = self._rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        return self._base.copy()

    def step(self, action):
        if self.fail_at is not None and self._t >= self.fail_at:
            raise MemoryError("OOM simule")
        self._t += 1
        shift = {"left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0)}.get(action, (0, 0))
        self._base = np.roll(self._base, shift, axis=(0, 1))
        frame = self._base.astype(np.float32)
        if self.drift:
            noise = self._rng.normal(0, 255 * self.drift * self._t, frame.shape)
            frame = frame + noise
        return np.clip(frame, 0, 255).astype(np.uint8)

    def unload(self):
        self.loaded = False
        self._base = None
