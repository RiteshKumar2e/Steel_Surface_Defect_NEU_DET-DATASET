"""Pytest fixtures and path setup for GraphDefect-KG tests."""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def synthetic_image() -> np.ndarray:
    """A deterministic synthetic BGR image (no dataset dependency)."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(200, 200, 3), dtype=np.uint8)
    # add a few bright diagonal streaks so texture/shape features are non-trivial
    for k in range(0, 200, 25):
        img[k:k + 3, :, :] = 240
    return img


@pytest.fixture(scope="session")
def node_features():
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((12, 160)).astype("float32")
    positions = rng.random((12, 2)).astype("float32")
    meta = [{"index": i, "type": "patch", "bbox": {"x0": 0, "y0": 0, "x1": 1, "y1": 1}}
            for i in range(12)]
    return feats, meta, positions
