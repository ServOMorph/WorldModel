from abc import ABC, abstractmethod

import numpy as np


class WorldModelAdapter(ABC):
    name = "base"
    family = "generic"
    actions = ("noop",)

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def reset(self, seed, image=None, prompt=None):
        pass

    @abstractmethod
    def step(self, action) -> np.ndarray:
        pass

    @abstractmethod
    def unload(self):
        pass
