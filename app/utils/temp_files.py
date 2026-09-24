from __future__ import annotations

import uuid
from pathlib import Path


def unique_path(directory: Path, suffix: str = ".png") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{uuid.uuid4().hex}{suffix}"
