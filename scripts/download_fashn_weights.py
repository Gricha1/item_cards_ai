"""Compatibility launcher for the official FASHN VTON weight downloader."""
from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    raise SystemExit(subprocess.call([sys.executable, "-m", "fashn_vton.scripts.download_weights", *sys.argv[1:]]))
