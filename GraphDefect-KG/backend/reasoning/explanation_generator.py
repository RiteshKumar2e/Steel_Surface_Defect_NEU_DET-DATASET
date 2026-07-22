"""Natural-language explanation generator.

Produces a fluent, evidence-grounded paragraph from the structured reasoning
object. Template-based by default (deterministic, no external API). An optional
local Transformer hook is provided but never required — if unavailable the
system falls back to the template, so the project runs fully offline.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..config import settings
from .feature_to_prompt import build_prompt


def _join(items: List[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def template_explanation(reasoning: Dict, untrained_notice: bool) -> str:
    cls = reasoning["predicted_class"]
    conf = reasoning["confidence"]
    support = [e["property"] for e in reasoning.get("supporting_evidence", [])]
    patches = reasoning.get("important_patches", [])
    causes = reasoning.get("class_causes", [])
    alt = reasoning.get("alternative_prediction")

    parts: List[str] = []
    lead = (
        f"The image is classified as {cls} with a confidence of {conf:.2f}."
    )
    parts.append(lead)

    if support:
        parts.append(
            f"This decision is supported by the detected visual properties "
            f"{_join(support)}, which are characteristic of {cls} in the domain "
            f"knowledge graph."
        )
    if patches:
        patch_str = ", ".join(f"patch {p}" for p in patches)
        parts.append(
            f"The graph model assigned the highest importance to {patch_str}, "
            f"where these cues are most pronounced; these regions are connected "
            f"through k-nearest-neighbour edges based on feature similarity."
        )
    if causes:
        parts.append(
            f"In manufacturing terms, {cls} is commonly associated with "
            f"{_join(causes)}."
        )
    if alt and alt != cls:
        parts.append(
            f"The most plausible alternative class is {alt}; the current "
            f"prediction is preferred because more supporting evidence aligns "
            f"with {cls}."
        )
    if untrained_notice:
        parts.append(
            "NOTE: the deep hybrid graph model is running with untrained weights; "
            "the reported class comes from the fitted MobileNetV2+KNN baseline. "
            "Train the hybrid model (see README) for research-grade results."
        )
    return " ".join(parts)


def generate_explanation(
    reasoning: Dict,
    global_features: Dict[str, float],
    untrained_notice: bool = False,
    use_local_llm: bool = False,
) -> Dict[str, object]:
    """Return both the prompt payload and the final explanation text."""
    prompt = build_prompt(
        predicted_class=reasoning["predicted_class"],
        confidence=reasoning["confidence"],
        features=global_features,
        top_properties=reasoning.get("visual_properties", []),
        kg_path=reasoning.get("knowledge_graph_path", []),
        important_patches=reasoning.get("important_patches", []),
    )

    text: Optional[str] = None
    if use_local_llm:
        text = _try_local_llm(prompt["prompt"])
    if not text:
        text = template_explanation(reasoning, untrained_notice)

    return {
        "explanation": text,
        "prompt": prompt["prompt"],
        "feature_statements": prompt["feature_statements"],
        "kg_statements": prompt["kg_statements"],
        "generated_by": "local_llm" if (use_local_llm and text and
                                        _LOCAL_LLM_OK) else "template",
    }


_LOCAL_LLM_OK = False


def _try_local_llm(prompt: str) -> Optional[str]:
    """Optional hook for a local Transformer (e.g. HF pipeline). Never required."""
    global _LOCAL_LLM_OK
    try:
        from transformers import pipeline  # type: ignore
        gen = pipeline("text2text-generation", model="google/flan-t5-small")
        out = gen(prompt, max_new_tokens=160)[0]["generated_text"]
        _LOCAL_LLM_OK = True
        return out.strip()
    except Exception:
        return None
