"""Upload validation, persistence and decoding for incoming images."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from ..config import settings
from ..utils.helpers import get_logger, new_id
from ..utils.preprocessing import decode_bytes

logger = get_logger(__name__)


class ImageValidationError(ValueError):
    """Raised when an uploaded file fails validation."""


def validate_upload(filename: str, data: bytes) -> None:
    """Check extension, size and decodability of an uploaded image."""
    ext = Path(filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise ImageValidationError(
            f"Unsupported file type '{ext}'. Allowed: "
            f"{', '.join(settings.allowed_extensions)}"
        )
    size_mb = len(data) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise ImageValidationError(
            f"File too large ({size_mb:.1f} MB > {settings.max_upload_mb} MB limit)"
        )
    if not data:
        raise ImageValidationError("Empty file")


def save_and_decode(filename: str, data: bytes) -> Tuple[str, Path, np.ndarray]:
    """Validate, persist to the uploads dir and decode to a BGR array.

    Returns ``(prediction_id, saved_path, image_bgr)``.
    """
    validate_upload(filename, data)
    try:
        img = decode_bytes(data)
    except Exception as exc:
        raise ImageValidationError(f"Could not decode image: {exc}") from exc

    pred_id = new_id("pred")
    ext = Path(filename).suffix.lower() or ".png"
    saved = settings.uploads_dir / f"{pred_id}{ext}"
    saved.write_bytes(data)
    logger.info("Saved upload %s (%d bytes)", saved.name, len(data))
    return pred_id, saved, img
