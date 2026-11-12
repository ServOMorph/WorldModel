import sys
from pathlib import Path

import numpy as np

from .base import WorldModelAdapter

ROOT = Path(__file__).resolve().parents[2]
OASIS = ROOT / "third_party" / "open-oasis"
SAMPLE = OASIS / "sample_data" / "treechop-f153ac423f61-20210916-183423.chunk_000.mp4"
SCALING = 0.07843137255
MAX_NOISE = 1000
NOISE_ABS_MAX = 20


class OpenOasisAdapter(WorldModelAdapter):
    family = "oasis"

    def __init__(self, ddim_steps=10, fp16=True):
        self.ddim_steps = ddim_steps
        self.fp16 = fp16
        self.name = f"open_oasis_{ddim_steps}steps"
        self.actions = ("noop", "up", "down", "left", "right", "fire")
        self.model = None
        self.vae = None

    def load(self):
        sys.path.insert(0, str(OASIS))
        import torch
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from dit import DiT_models
        from utils import ACTION_KEYS, sigmoid_beta_schedule
        from vae import VAE_models

        self.torch = torch
        self.keys = ACTION_KEYS
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dit_path = hf_hub_download("Etched/oasis-500m", "oasis500m.safetensors")
        vae_path = hf_hub_download("Etched/oasis-500m", "vit-l-20.safetensors")
        model = DiT_models["DiT-S/2"]()
        model.load_state_dict(load_file(dit_path), strict=False)
        vae = VAE_models["vit-l-20-shallow-encoder"]()
        vae.load_state_dict(load_file(vae_path))
        self.model = model.to(self.device).eval()
        self.vae = vae.to(self.device).eval()
        betas = sigmoid_beta_schedule(MAX_NOISE).to(self.device)
        ac = torch.cumprod(1.0 - betas, dim=0)
        self.alphas_cumprod = ac.reshape(-1, 1, 1, 1)
        self.noise_range = torch.linspace(-1, MAX_NOISE - 1, self.ddim_steps + 1)
        self.ctx_max_noise_idx = self.ddim_steps // 10 * 3

    def _action_vec(self, action):
        torch = self.torch
        v = torch.zeros(len(self.keys))
        name = {"up": "forward", "down": "back", "left": "left", "right": "right", "fire": "attack"}.get(action)
        if name:
            v[self.keys.index(name)] = 1.0
        return v.to(self.device)

    def _decode(self, latent):
        from einops import rearrange

        torch = self.torch
        z = rearrange(latent, "b c h w -> b (h w) c").float()
        with torch.no_grad():
            img = (self.vae.decode(z / SCALING) + 1) / 2
        img = img.clamp(0, 1).mul(255).byte()[0].permute(1, 2, 0).cpu().numpy()
        return np.ascontiguousarray(img)

    def reset(self, seed, image=None, prompt=None):
        from einops import rearrange

        torch = self.torch
        if self.model is None:
            raise RuntimeError("modele non charge")
        torch.manual_seed(seed)
        if image is None:
            import imageio

            frame = torch.from_numpy(np.asarray(imageio.get_reader(str(SAMPLE)).get_data(0)))
        else:
            frame = torch.as_tensor(image)
        x = frame.float().div(255).permute(2, 0, 1).unsqueeze(0).to(self.device)
        H, W = x.shape[-2:]
        with torch.no_grad():
            z = self.vae.encode(x * 2 - 1).mean * SCALING
        z = rearrange(z, "b (h w) c -> b c h w", h=H // self.vae.patch_size, w=W // self.vae.patch_size)
        self.x = z.unsqueeze(1)
        self.acts = [self._action_vec("noop")]
        return self._decode(z)

    def step(self, action):
        torch = self.torch
        max_frames = self.model.max_frames
        if self.x.shape[1] >= max_frames:
            self.x = self.x[:, -(max_frames - 1):]
            self.acts = self.acts[-(max_frames - 1):]
        self.acts.append(self._action_vec(action))
        acts = torch.stack(self.acts).unsqueeze(0)
        chunk = torch.clamp(torch.randn((1, 1, *self.x.shape[-3:]), device=self.device), -NOISE_ABS_MAX, NOISE_ABS_MAX)
        x = torch.cat([self.x, chunk], dim=1)
        i = x.shape[1] - 1
        ac = self.alphas_cumprod
        for noise_idx in reversed(range(1, self.ddim_steps + 1)):
            ctx_idx = min(noise_idx, self.ctx_max_noise_idx)
            t_ctx = torch.full((1, i), self.noise_range[ctx_idx], dtype=torch.long, device=self.device)
            t = torch.full((1, 1), self.noise_range[noise_idx], dtype=torch.long, device=self.device)
            t_next = torch.full((1, 1), self.noise_range[noise_idx - 1], dtype=torch.long, device=self.device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)
            x_curr = x.clone()
            ctx_noise = torch.clamp(torch.randn_like(x_curr[:, :-1]), -NOISE_ABS_MAX, NOISE_ABS_MAX)
            x_curr[:, :-1] = ac[t[:, :-1]].sqrt() * x_curr[:, :-1] + (1 - ac[t[:, :-1]]).sqrt() * ctx_noise
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.half, enabled=self.fp16 and self.device.type == "cuda"):
                v = self.model(x_curr, t, acts).float()
            x_start = ac[t].sqrt() * x_curr - (1 - ac[t]).sqrt() * v
            x_noise = ((1 / ac[t]).sqrt() * x_curr - x_start) / (1 / ac[t] - 1).sqrt()
            x_pred = ac[t_next].sqrt() * x_start + x_noise * (1 - ac[t_next]).sqrt()
            x[:, -1:] = x_pred[:, -1:]
        self.x = x
        return self._decode(x[:, -1])

    def unload(self):
        self.model = None
        self.vae = None
        self.x = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()
