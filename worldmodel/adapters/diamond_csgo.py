import sys
from pathlib import Path

import numpy as np

from .base import WorldModelAdapter

ROOT = Path(__file__).resolve().parents[2]
DIAMOND = ROOT / "third_party" / "diamond_csgo2"
CONFLICTING = ("agent", "envs", "models", "coroutines", "data", "game", "utils", "csgo")


class DiamondCsgoAdapter(WorldModelAdapter):
    family = "diamond"

    def __init__(self, quality="fast"):
        self.quality = quality
        self.name = f"diamond_csgo_{quality}"
        self.actions = ("noop", "up", "down", "left", "right", "fire")
        self.wm_env = None
        self.device = None

    def load(self):
        for m in [m for m in sys.modules if m.split(".")[0] in CONFLICTING]:
            del sys.modules[m]
        sys.path.insert(0, str(DIAMOND / "src"))
        import torch
        from huggingface_hub import snapshot_download
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
        from agent import Agent
        from envs import WorldModelEnv

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        hf = Path(snapshot_download(repo_id="eloialonso/diamond", allow_patterns="csgo/*"))
        agent_cfg = OmegaConf.load(hf / "csgo/config/agent/csgo.yaml")
        env_cfg = OmegaConf.load(hf / "csgo/config/env/csgo.yaml")
        wm_cfg = OmegaConf.load(DIAMOND / "config" / "world_model_env" / f"{self.quality}.yaml")
        root = OmegaConf.create({"agent": agent_cfg, "env": env_cfg, "world_model_env": wm_cfg})
        OmegaConf.resolve(root)

        _orig = torch.load
        torch.load = lambda *a, **k: _orig(*a, **{**k, "weights_only": False})
        try:
            agent = Agent(instantiate(root.agent, num_actions=root.env.num_actions)).to(self.device).eval()
            agent.load(hf / "csgo/model/csgo.pt")
        finally:
            torch.load = _orig

        sl = root.agent.denoiser.inner_model.num_steps_conditioning
        if agent.upsampler is not None:
            sl = max(sl, root.agent.upsampler.inner_model.num_steps_conditioning)
        wm_env_cfg = instantiate(root.world_model_env, num_batches_to_preload=1)
        self.wm_env = WorldModelEnv(
            agent.denoiser, agent.upsampler, agent.rew_end_model, hf / "csgo/spawn", 1, sl, wm_env_cfg
        )
        self._agent = agent

    def _encode(self, action):
        import pygame
        from csgo.action_processing import CSGOAction, encode_csgo_action

        keys = {
            "up": [pygame.K_w],
            "down": [pygame.K_s],
            "left": [pygame.K_LEFT],
            "right": [pygame.K_RIGHT],
        }.get(action, [])
        ca = CSGOAction(keys, 0, 0, action == "fire", False)
        return encode_csgo_action(ca, device=self.device)

    def _frame(self, obs):
        arr = obs[0].detach().float().add(1).div(2).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        return np.ascontiguousarray(arr)

    def reset(self, seed, image=None, prompt=None):
        import torch

        if self.wm_env is None:
            raise RuntimeError("modele non charge")
        torch.manual_seed(seed)
        obs, _ = self.wm_env.reset()
        return self._frame(obs)

    def step(self, action):
        obs, *_ = self.wm_env.step(self._encode(action))
        return self._frame(obs)

    def unload(self):
        import torch

        self.wm_env = None
        self._agent = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
