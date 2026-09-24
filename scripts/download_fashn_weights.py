"""Compatibility launcher for the official FASHN VTON weight downloader."""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-dir", required=True, type=Path)
    args = parser.parse_args()
    weights_dir = args.weights_dir.resolve()
    dwpose_dir = weights_dir / "dwpose"
    weights_dir.mkdir(parents=True, exist_ok=True)
    dwpose_dir.mkdir(exist_ok=True)

    hf_hub_download("fashn-ai/fashn-vton-1.5", "model.safetensors", local_dir=weights_dir)
    for filename in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        hf_hub_download("fashn-ai/DWPose", filename, local_dir=dwpose_dir)

    # Instantiation fetches parser weights required by the FASHN pipeline.
    from fashn_human_parser import FashnHumanParser

    FashnHumanParser(device="cpu")


if __name__ == "__main__":
    main()
