"""End-to-end prediction pipeline tests (no dataset required — synthetic image)."""
import numpy as np

from backend.services.prediction_service import run_prediction
from backend.models.model_loader import load_bundle
from backend.config import CLASS_NAMES


def test_full_pipeline_on_synthetic(synthetic_image):
    # Do not auto-fit the KNN baseline (keeps the test fast and dataset-free).
    bundle = load_bundle(auto_fit_knn=False)
    result = run_prediction(synthetic_image, "test_synth", filename="synth.png",
                            bundle=bundle)

    # core fields
    assert result["predicted_class"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["probabilities"].keys()) == set(CLASS_NAMES)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-2

    # honesty flags present
    assert "model_trained" in result
    assert "prediction_source" in result
    assert "untrained_notice" in result

    # graph + explanation
    assert result["graph_summary"]["nodes"] > 0
    assert result["graph_summary"]["edges"] > 0
    assert isinstance(result["explanation"], str) and result["explanation"]

    # reasoning structure
    r = result["reasoning"]
    assert r["predicted_class"] == result["predicted_class"]
    assert "knowledge_graph_path" in r
    assert "alternative_prediction" in r

    # component gates sum ~ 1
    assert abs(sum(result["component_gates"].values()) - 1.0) < 1e-2


def test_untrained_flag_when_no_hybrid_checkpoint(synthetic_image):
    bundle = load_bundle(auto_fit_knn=False)
    result = run_prediction(synthetic_image, "test_flag", filename="synth.png",
                            bundle=bundle)
    if not bundle.hybrid_trained:
        # must be transparently flagged, never presented as a valid trained result
        assert result["untrained_notice"] is True
