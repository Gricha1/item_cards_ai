"""Run one FASHN VTON inference without Telegram or any API key."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.services.fashn_vton import FashnVtonService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", type=Path, required=True)
    parser.add_argument("--garment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path("weights/fashn-vton-1.5"))
    parser.add_argument("--category", default="tops")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--low-memory", action="store_true")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    service = FashnVtonService(
        args.weights,
        args.category,
        low_memory=args.low_memory,
        num_timesteps=args.steps,
    )
    result = service.generate(args.person, args.garment, args.output)
    print(result)


if __name__ == "__main__":
    main()
