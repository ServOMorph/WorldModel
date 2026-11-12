import sys
from pathlib import Path

import numpy as np

from .base import WorldModelAdapter

ROOT = Path(__file__).resolve().parents[2]
DIAMOND = ROOT / "third_party" / "diamond"
CACHE = ROOT / "models" / "diamond_atari"


class DiamondAtariAdapter(WorldModelAdapter):
    family = "diamond"

    def __init__(self, game="Breakout", init_steps=1000, horizon=100000, denoising_steps=3):
        self.game = game
        self.name = f"diamond_atari_{game.lower()}"
        self.init_steps = init_steps
        self.horizon = horizon
        self.denoising_steps = denoising_steps
        self.actions = ("noop", "fire", "right", "left")
        self.wm_env = None
        self.device = None

    def _download(self, filename):
        from huggingface_hub import hf_hub_download

        return Path(hf_hub_download(repo_id="eloialonso/diamond", filename=filename))

    def load(self):
        if str(DIAMOND / "src") not in sys.path:
            sys.path.insert(0, str(DIAMOND / "src"))
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from torch.utils.data import DataLoader

        from torch.utils.data import Sampler

        Sampler.__init__ = lambda self, *a, **k: None

        from agent import Agent
        from coroutines.collector import NumToCollect, make_collector
        from data import BatchSampler, Dataset, collate_segments_to_batch
        from envs import WorldModelEnv, make_atari_env

        if not OmegaConf.has_resolver("eval"):
            OmegaConf.register_new_resolver("eval", eval)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        ckpt = self._download(f"atari_100k/models/{self.game}.pt")
        agent_cfg = OmegaConf.load(self._download("atari_100k/config/agent/default.yaml"))
        env_cfg = OmegaConf.load(self._download("atari_100k/config/env/atari.yaml"))
        env_cfg.train.id = env_cfg.test.id = f"{self.game}NoFrameskip-v4"
        wm_cfg = OmegaConf.load(DIAMOND / "config" / "trainer.yaml").world_model_env
        wm_cfg.horizon = self.horizon
        wm_cfg.diffusion_sampler.num_steps_denoising = self.denoising_steps
        root = OmegaConf.create({"agent": agent_cfg, "env": env_cfg, "world_model_env": wm_cfg})
        OmegaConf.resolve(root)

        test_env = make_atari_env(num_envs=1, device=self.device, **root.env.test)
        agent = Agent(instantiate(root.agent, num_actions=test_env.num_actions)).to(self.device).eval()
        agent.load(ckpt)

        CACHE.mkdir(parents=True, exist_ok=True)
        dataset = Dataset(CACHE / f"{self.game}_{self.init_steps}")
        _orig_load = torch.load
        torch.load = lambda *a, **k: _orig_load(*a, **{**k, "weights_only": False})
        try:
            dataset.load_from_default_path()
        finally:
            torch.load = _orig_load
        if len(dataset) == 0:
            collector = make_collector(test_env, agent.actor_critic, dataset, epsilon=0)
            collector.send(NumToCollect(steps=self.init_steps))
            dataset.save_to_default_path()

        bs = BatchSampler(dataset, 0, 1, 1, root.agent.denoiser.inner_model.num_steps_conditioning, None, False)
        dl = DataLoader(dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch)
        wm_env_cfg = instantiate(root.world_model_env, num_batches_to_preload=1)
        self.wm_env = WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, wm_env_cfg)
        self._agent = agent
        self._action_map = self._build_action_map()

    def _build_action_map(self):
        import gymnasium

        names = [x.lower() for x in gymnasium.make(f"{self.game}NoFrameskip-v4").unwrapped.get_action_meanings()]
        return {a: names.index(a) for a in self.actions if a in names}

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
        import torch

        idx = self._action_map.get(action, self._action_map.get("noop", 0))
        act = torch.tensor([idx], device=self.device)
        obs, *_ = self.wm_env.step(act)
        return self._frame(obs)

    def unload(self):
        import torch

        self.wm_env = None
        self._agent = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
