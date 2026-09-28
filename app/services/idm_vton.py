from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from shutil import copyfile
from threading import Event, Thread
from time import monotonic


class IdmVtonError(RuntimeError):
    pass


class IdmVtonService:
    """Client for the isolated IDM-VTON HD worker on ML4."""

    def __init__(self, remote_url: str | None) -> None:
        self.remote_url = remote_url.rstrip("/") if remote_url else None

    @property
    def enabled(self) -> bool:
        return self.remote_url is not None

    def generate(
        self,
        person_path: Path,
        garment_path: Path,
        output_path: Path,
        progress_callback: Callable[[str, int, int], None] | None = None,
        *,
        variant: str = "first",
        seed: int = 42,
    ) -> Path:
        if not self.remote_url:
            raise IdmVtonError("IDM-VTON worker is not configured.")

        finished = Event()

        def report_estimated_progress() -> None:
            # Gradio's upstream IDM endpoint has no step streaming. Its 30
            # denoising steps take roughly 30–45 seconds on ML4 GPU 2.
            started = monotonic()
            while not finished.wait(3):
                if progress_callback is not None:
                    elapsed = monotonic() - started
                    # A negative total denotes a time estimate in seconds;
                    # the bot distinguishes it from FASHN's slower estimate.
                    progress_callback(variant, min(95, max(1, round(elapsed * 100 / 45))), -45)

        reporter = Thread(target=report_estimated_progress, daemon=True)
        reporter.start()
        try:
            from gradio_client import Client

            client = Client(self.remote_url, verbose=False)
            result = client.predict(
                # gradio-client 0.14 uploads local file paths itself.  This
                # form also remains compatible with newer client versions.
                {"background": str(person_path), "layers": [], "composite": None},
                str(garment_path),
                (
                    "exact product garment; preserve the original color, fabric, silhouette, "
                    "zipper, collar, cuffs, pockets and all visible details"
                ),
                True,
                False,
                30,
                seed,
                api_name="/tryon",
            )
            generated_path = Path(result[0])
            if not generated_path.is_file():
                raise FileNotFoundError(f"IDM-VTON did not return an output image: {generated_path}")
            copyfile(generated_path, output_path)
            if progress_callback is not None:
                progress_callback(variant, 100, -45)
            return output_path
        except Exception as error:
            raise IdmVtonError("IDM-VTON worker is unavailable.") from error
        finally:
            finished.set()
