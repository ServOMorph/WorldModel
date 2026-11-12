import os
import sys


def collect():
    import torch

    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "hf_home": os.environ.get("HF_HOME"),
    }
    if info["cuda_available"]:
        props = torch.cuda.get_device_properties(0)
        info["gpu"] = props.name
        info["vram_gb"] = round(props.total_memory / 1024**3, 2)
        x = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
        y = x @ x
        torch.cuda.synchronize()
        info["alloc_test"] = bool(torch.isfinite(y).all().item())
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
    return info


def main():
    info = collect()
    for k, v in info.items():
        print(f"{k}: {v}")
    ok = info["cuda_available"] and info.get("alloc_test", False)
    print("OK" if ok else "ECHEC")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
