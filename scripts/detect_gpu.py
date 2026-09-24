from __future__ import annotations


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch не установлен.")
        return 1
    available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {available}")
    if not available:
        print("Рекомендация: Qwen отключить; FASHN/FLUX будут работать только при корректной CUDA или CPU fallback.")
        return 0
    properties = torch.cuda.get_device_properties(0)
    vram_gb = properties.total_memory / 1024**3
    print(f"GPU: {properties.name}")
    print(f"VRAM: {vram_gb:.1f} GB")
    if vram_gb >= 20:
        print("Рекомендация: Qwen-Image-Edit можно пробовать с CPU offload.")
    else:
        print("Рекомендация: Qwen-Image-Edit не запускать; включить FLUX.1-schnell fallback с CPU offload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
