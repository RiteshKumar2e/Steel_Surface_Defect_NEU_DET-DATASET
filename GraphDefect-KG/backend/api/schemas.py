"""Pydantic response schemas for the GraphDefect-KG API.

Kept intentionally permissive (``extra='allow'``) for the rich nested result so
the pipeline can evolve without breaking the contract, while still documenting
the key fields for the OpenAPI docs.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    device: str
    hybrid_trained: bool
    knn_baseline_available: bool


class ClassInfo(BaseModel):
    index: int
    name: str
    visual_properties: List[str]
    causes: List[str]


class ClassesResponse(BaseModel):
    classes: List[ClassInfo]


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    backbone: Dict
    node_feature_dim: int
    graph_models: List[str]
    hybrid_trained: bool
    primary_source: str
    knn_baseline_train_size: int
    num_classes: int


class ComparisonRow(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model: str
    predicted_class: str
    confidence: float
    trained: bool


class PredictionResponse(BaseModel):
    """Top-level prediction payload. Extra nested fields are allowed."""
    model_config = ConfigDict(extra="allow", protected_namespaces=())

    prediction_id: str
    filename: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    prediction_source: str
    model_trained: bool
    hybrid_trained: bool
    untrained_notice: bool
    explanation: str
    visual_features: Dict[str, float]
    kg_affinity: Dict[str, float]
    component_gates: Dict[str, float]
    important_regions: List[Dict]
    model_comparison: List[ComparisonRow]
    graph_summary: Dict[str, int]
    timings_ms: Dict[str, float]


class GraphResponse(BaseModel):
    elements: List[Dict]
    counts: Dict[str, int]
    legend: Dict
    node_types: List[str]
    edge_types: List[str]


class ErrorResponse(BaseModel):
    detail: str
