from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError


class ImageValidationError(ValueError):
    pass


def validate_and_normalize(source: Path, destination: Path, max_side: int = 1536) -> Path:
    """Reject malformed uploads and save a bounded RGB PNG for model inputs."""
    try:
        with Image.open(source) as image:
            image.verify()
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination, "PNG", optimize=True)
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise ImageValidationError("Не удалось прочитать фото. Пришлите корректный JPG или PNG.") from error
    return destination
