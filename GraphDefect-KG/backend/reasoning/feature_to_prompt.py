"""Feature-to-prompt conversion (LLM-inspired, template based).

Turns numeric/graph evidence into structured natural-language *statements* that
a language model — or the built-in rule-based reasoner — can consume. This
implements the "feature-prompt" idea locally, with no external API dependency.
"""
from __future__ import annotations

from typing import Dict, List

# Qualitative buckets for feature magnitudes (feature values are roughly [0,1]).
def _level(v: float, low: float = 0.2, high: float = 0.5) -> str:
    if v >= high:
        return "high"
    if v <= low:
        return "low"
    return "moderate"


FEATURE_PHRASES = {
    "edge_density": "edge density is {level}",
    "std_intensity": "local intensity variation is {level}",
    "entropy": "textural entropy is {level}",
    "glcm_contrast": "GLCM contrast is {level}",
    "glcm_homogeneity": "surface homogeneity is {level}",
    "roughness": "surface roughness is {level}",
    "circularity": "region circularity is {level}",
    "aspect_ratio": "region elongation is {level}",
    "solidity": "region solidity is {level}",
}


def describe_features(feats: Dict[str, float]) -> List[str]:
    """Return human-readable statements for the salient handcrafted features."""
    statements: List[str] = []
    for key, template in FEATURE_PHRASES.items():
        if key not in feats:
            continue
        val = feats[key]
        if key == "aspect_ratio":
            # elongation: distance from 1.0 in log space
            import math
            val = min(1.0, abs(math.log(val + 1e-3)))
        statements.append(template.format(level=_level(val)))
    return statements


def build_prompt(
    predicted_class: str,
    confidence: float,
    features: Dict[str, float],
    top_properties: List[str],
    kg_path: List[Dict],
    important_patches: List[int],
) -> Dict[str, object]:
    """Assemble a structured prompt payload describing all evidence."""
    feature_statements = describe_features(features)
    kg_statements = [
        f"{t['subject']} {t['relation'].replace('_', ' ')} {t['object']}"
        for t in kg_path
    ]
    prompt_text = (
        f"Task: explain why a steel surface image was classified as "
        f"'{predicted_class}' (confidence {confidence:.2f}).\n"
        f"Detected visual properties: {', '.join(top_properties)}.\n"
        f"Feature observations: {'; '.join(feature_statements)}.\n"
        f"Knowledge-graph relations: {'; '.join(kg_statements)}.\n"
        f"Most influential image regions (patch ids): {important_patches}.\n"
        f"Write a concise, evidence-grounded justification."
    )
    return {
        "prompt": prompt_text,
        "feature_statements": feature_statements,
        "kg_statements": kg_statements,
        "top_properties": top_properties,
    }
