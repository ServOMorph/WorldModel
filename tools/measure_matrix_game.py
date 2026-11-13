import argparse
import json
import os
import sys
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MG = ROOT / "third_party" / "matrix-game" / "Matrix-Game-2"
WEIGHTS = ROOT / "models" / "matrix-game-2"
VRAM_LIMIT_GB = 7.5


def install_flash_attn_shim():
    try:
        import flash_attn
        return False
    except ImportError:
        pass
    import torch

    def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
        return out.transpose(1, 2)

    import importlib.machinery

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                               dropout_p=0.0, softmax_scale=None, causal=False, **kwargs):
        b = cu_seqlens_q.numel() - 1
        for cu, ln in ((cu_seqlens_q, max_seqlen_q), (cu_seqlens_k, max_seqlen_k)):
            assert torch.equal(cu.cpu(), torch.arange(b + 1, dtype=cu.dtype) * ln), "longueurs variables non supportees"
        qq = q.view(b, max_seqlen_q, *q.shape[1:])
        kk = k.view(b, max_seqlen_k, *k.shape[1:])
        vv = v.view(b, max_seqlen_k, *v.shape[1:])
        out = torch.nn.functional.scaled_dot_product_attention(
            qq.transpose(1, 2), kk.transpose(1, 2), vv.transpose(1, 2),
            dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
        return out.transpose(1, 2).reshape(b * max_seqlen_q, *q.shape[1:])

    mod = types.ModuleType("flash_attn")
    mod.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)
    mod.flash_attn_func = flash_attn_func
    mod.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = mod
    return True


def start_guard(torch, limit_gb):
    state = {"peak_card_gb": 0.0, "stop": False, "phase": "init"}

    def run():
        while not state["stop"]:
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / 1024 ** 3
            state["peak_card_gb"] = max(state["peak_card_gb"], used)
            if used > limit_gb:
                print(f"GARDE-FOU: {used:.2f} Go > {limit_gb} Go, phase={state['phase']}, alloue={torch.cuda.memory_allocated()/1024**3:.2f}, reserve={torch.cuda.memory_reserved()/1024**3:.2f}", flush=True)
                os._exit(3)
            time.sleep(0.2)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return state


def load_image(path, torch, dtype):
    from PIL import Image
    from torchvision.transforms import v2

    image = Image.open(path).convert("RGB")
    w, h = image.size
    th, tw = 352, 640
    if h / w > th / tw:
        nh = int(w * th / tw)
        top = (h - nh) // 2
        image = image.crop((0, top, w, top + nh))
    else:
        nw = int(h * tw / th)
        left = (w - nw) // 2
        image = image.crop((left, 0, left + nw, h))
    tf = v2.Compose([
        v2.Resize(size=(th, tw), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    return tf(image)[None, :, None, :, :].to(dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--image", default=str(MG / "demo_images" / "universal" / "0000.png"))
    ap.add_argument("--out", default=str(ROOT / "bench" / "matrix_game_smoke.json"))
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    os.chdir(MG)
    sys.path.insert(0, str(MG))
    import torch

    shimmed = install_flash_attn_shim()

    from omegaconf import OmegaConf
    from safetensors.torch import load_file

    from demo_utils.vae_block3 import VAEDecoderWrapper
    from pipeline import CausalInferenceStreamingPipeline
    from pipeline.causal_inference import cond_current
    from utils.conditions import Bench_actions_universal
    from utils.wan_wrapper import WanDiffusionWrapper
    from wan.vae.wanx_vae import get_wanx_vae_wrapper

    if args.dry:
        print("imports ok, shim:", shimmed)
        return

    import psutil
    from einops import rearrange

    device = torch.device("cuda")
    dtype = torch.bfloat16
    proc = psutil.Process()
    ram_peak = {"gb": 0.0}

    def ram_sample():
        ram_peak["gb"] = max(ram_peak["gb"], proc.memory_info().rss / 1024 ** 3)

    guard = start_guard(torch, VRAM_LIMIT_GB)
    torch.cuda.reset_peak_memory_stats()
    res = {"flash_attn_shim": shimmed, "blocks": args.blocks, "limit_gb": VRAM_LIMIT_GB}

    config = OmegaConf.load("configs/inference_yaml/inference_universal.yaml")
    mode = config.pop("mode")
    n_lat = args.blocks * config.num_frame_per_block
    num_frames = (n_lat - 1) * 4 + 1

    import torch.nn.functional as F

    guard["phase"] = "clip"
    t1 = time.time()
    vae = get_wanx_vae_wrapper(str(WEIGHTS), torch.float16)
    ram_sample()
    image = load_image(args.image, torch, dtype)
    clip = vae.clip
    visual = clip.model.visual.to(device, dtype=torch.bfloat16)
    cv = image.transpose(1, 2).reshape(-1, 3, image.shape[-2], image.shape[-1]).to(device).float()
    cv = F.interpolate(cv, size=(clip.model.image_size,) * 2, mode="bicubic", align_corners=False)
    cv = clip.transforms.transforms[-1](cv.mul_(0.5).add_(0.5))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        visual_context = visual(cv, use_31_block=True).to(dtype)
    res["vram_peak_clip_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    del cv, visual
    clip.model.visual.to("cpu")
    guard["phase"] = "vae_encode"
    vae.vae = vae.vae.to(device, dtype)
    img_gpu = image.to(device)
    pad = torch.zeros_like(img_gpu).repeat(1, 1, 4 * (n_lat - 1), 1, 1)
    img_cond = vae.encode(torch.concat([img_gpu, pad], dim=2), device=device,
                          tiled=True, tile_size=[44, 80], tile_stride=[23, 38]).to(device)
    del img_gpu, pad
    vae.vae = vae.vae.to("cpu")
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    res["reset_s"] = round(time.time() - t1, 1)
    res["vram_peak_reset_gb"] = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    mask = torch.ones_like(img_cond)
    mask[:, :, 1:] = 0
    cond_concat = torch.cat([mask[:, :4], img_cond], dim=1)
    cd = Bench_actions_universal(num_frames)
    conditional_dict = {
        "cond_concat": cond_concat.to(device=device, dtype=dtype),
        "visual_context": visual_context,
        "mouse_cond": cd["mouse_condition"].unsqueeze(0).to(device=device, dtype=dtype),
        "keyboard_cond": cd["keyboard_condition"].unsqueeze(0).to(device=device, dtype=dtype),
    }

    guard["phase"] = "load_dit"
    t0 = time.time()
    generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)
    decoder = VAEDecoderWrapper()
    vae_sd = torch.load(WEIGHTS / "Wan2.1_VAE.pth", map_location="cpu")
    decoder.load_state_dict({k: v for k, v in vae_sd.items() if "decoder." in k or "conv2" in k})
    del vae_sd
    decoder.to(device, torch.float16).requires_grad_(False).eval()
    pipeline = CausalInferenceStreamingPipeline(config, generator=generator, vae_decoder=decoder)
    pipeline.generator.load_state_dict(load_file(str(WEIGHTS / "base_distilled_model" / "base_distill.safetensors")))
    ram_sample()
    pipeline = pipeline.to(device=device, dtype=dtype)
    pipeline.vae_decoder.to(torch.float16)
    ram_sample()
    torch.cuda.synchronize()
    res["load_s"] = round(time.time() - t0, 1)
    res["vram_after_load_gb"] = round(torch.cuda.memory_allocated() / 1024 ** 3, 2)
    noise = torch.randn([1, 16, n_lat, 44, 80], device=device, dtype=dtype)

    guard["phase"] = "kv_init"
    p = pipeline
    p._initialize_kv_cache(1, dtype, device)
    p._initialize_kv_cache_mouse_and_keyboard(1, dtype, device)
    p._initialize_crossattn_cache(1, dtype, device)
    from copy import deepcopy
    from demo_utils.constant import ZERO_VAE_CACHE
    vae_cache = deepcopy(ZERO_VAE_CACHE)
    for j in range(len(vae_cache)):
        vae_cache[j] = None

    actions = [
        {"mouse": [0, 0], "keyboard": [1, 0, 0, 0]},
        {"mouse": [0, 0.1], "keyboard": [1, 0, 0, 0]},
        {"mouse": [0, 0], "keyboard": [0, 0, 1, 0]},
    ]
    guard["phase"] = "blocks"
    block_times, frames_out = [], 0
    start = 0
    nfpb = p.num_frame_per_block
    with torch.no_grad():
        for b in range(args.blocks):
            a = actions[b % len(actions)]
            replace = {"mouse": torch.tensor(a["mouse"], device=device),
                       "keyboard": torch.tensor(a["keyboard"], device=device)}
            torch.cuda.synchronize()
            tb = time.time()
            noisy = noise[:, :, start:start + nfpb]
            new_act, conditional_dict = cond_current(conditional_dict, start, nfpb, replace=replace, mode=mode)
            for index, ts in enumerate(p.denoising_step_list):
                timestep = torch.ones([1, nfpb], device=device, dtype=torch.int64) * ts
                _, pred = p.generator(
                    noisy_image_or_video=noisy, conditional_dict=new_act, timestep=timestep,
                    kv_cache=p.kv_cache1, kv_cache_mouse=p.kv_cache_mouse,
                    kv_cache_keyboard=p.kv_cache_keyboard, crossattn_cache=p.crossattn_cache,
                    current_start=start * p.frame_seq_length)
                if index < len(p.denoising_step_list) - 1:
                    nxt = p.denoising_step_list[index + 1]
                    flat = rearrange(pred, "b c f h w -> (b f) c h w")
                    noisy = p.scheduler.add_noise(flat, torch.randn_like(flat),
                                                  nxt * torch.ones([nfpb], device=device, dtype=torch.long))
                    noisy = rearrange(noisy, "(b f) c h w -> b c f h w", b=pred.shape[0])
            ctx_t = torch.ones_like(timestep) * p.args.context_noise
            p.generator(
                noisy_image_or_video=pred, conditional_dict=new_act, timestep=ctx_t,
                kv_cache=p.kv_cache1, kv_cache_mouse=p.kv_cache_mouse,
                kv_cache_keyboard=p.kv_cache_keyboard, crossattn_cache=p.crossattn_cache,
                current_start=start * p.frame_seq_length)
            video, vae_cache = p.vae_decoder(pred.transpose(1, 2).half(), *vae_cache)
            torch.cuda.synchronize()
            block_times.append(time.time() - tb)
            frames_out += video.shape[1]
            start += nfpb
            ram_sample()
            print(f"bloc {b}: {block_times[-1]:.2f}s alloue_pic={torch.cuda.max_memory_allocated()/1024**3:.2f} reserve_pic={torch.cuda.max_memory_reserved()/1024**3:.2f} carte={guard['peak_card_gb']:.2f}", flush=True)

    guard["stop"] = True
    total = sum(block_times)
    res.update({
        "block_times_s": [round(t, 2) for t in block_times],
        "frames_out": frames_out,
        "fps_mean": round(frames_out / total, 3),
        "fps_excl_first_block": round((frames_out - frames_out // args.blocks) / sum(block_times[1:]), 3)
        if args.blocks > 1 else None,
        "vram_peak_process_gb": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
        "vram_peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2),
        "vram_peak_card_gb": round(guard["peak_card_gb"], 2),
        "ram_peak_gb": round(ram_peak["gb"], 2),
    })
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
