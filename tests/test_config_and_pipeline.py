from pathlib import Path

import pytest

from app.config import Settings
from app.services.pipeline import GenerationPipeline, TemplateNotFoundError
from app.utils.image_io import ImageValidationError, validate_and_normalize


def test_settings_resolves_relative_data_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-123456")
    settings = Settings(output_dir="data/test-output")
    assert settings.output_dir.is_absolute()


def test_missing_template_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token-123456")
    pipeline = GenerationPipeline(Settings(output_dir=tmp_path))
    with pytest.raises(TemplateNotFoundError):
        pipeline.template_for("female", "full")


def test_invalid_image_is_rejected(tmp_path):
    source = tmp_path / "bad.jpg"
    source.write_bytes(b"not-a-real-image")
    with pytest.raises(ImageValidationError):
        validate_and_normalize(source, tmp_path / "result.png")
