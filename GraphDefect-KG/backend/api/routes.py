"""FastAPI routes for GraphDefect-KG."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from ..config import CLASS_NAMES, settings
from ..graph.graph_serializer import load_graph
from ..graph.knowledge_graph import KnowledgeGraph
from ..models.model_loader import NODE_FEATURE_DIM, load_bundle
from ..services.batch_service import predict_many
from ..services.image_service import ImageValidationError, save_and_decode, validate_upload
from ..services.prediction_service import run_prediction
from ..utils.helpers import get_logger, new_id, save_json, to_jsonable
from ..utils.preprocessing import decode_bytes

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["graphdefect"])

# In-memory store of the latest results (also persisted to disk as JSON).
_RESULTS: Dict[str, Dict] = {}


def _result_path(pred_id: str) -> Path:
    return settings.graph_data_dir / f"result_{pred_id}.json"


@router.get("/health")
def health():
    bundle = load_bundle()
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "device": bundle.device,
        "hybrid_trained": bundle.hybrid_trained,
        "knn_baseline_available": bundle.knn_baseline is not None,
    }


@router.get("/classes")
def classes():
    kg = KnowledgeGraph.load_or_build()
    return {
        "classes": [
            {
                "index": i,
                "name": name,
                "visual_properties": kg.class_properties(name),
                "causes": kg.class_causes(name),
            }
            for i, name in enumerate(CLASS_NAMES)
        ]
    }


@router.get("/model-info")
def model_info():
    bundle = load_bundle()
    return {
        "backbone": bundle.extractor.info(),
        "node_feature_dim": NODE_FEATURE_DIM,
        "graph_models": ["GCN", "GraphSAGE", "GAT", "HybridDefectModel"],
        "hybrid_trained": bundle.hybrid_trained,
        "primary_source": bundle.primary_source(),
        "knn_baseline_train_size": (
            bundle.knn_baseline.n_train if bundle.knn_baseline else 0
        ),
        "num_classes": len(CLASS_NAMES),
    }


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    try:
        pred_id, saved_path, img = save_and_decode(file.filename or "upload.png", data)
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        result = run_prediction(img, pred_id, filename=saved_path.name)
    except Exception as exc:  # keep the server resilient
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    result = to_jsonable(result)
    _RESULTS[pred_id] = result
    save_json(_result_path(pred_id), result)
    return JSONResponse(result)


@router.post("/predict-batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """Score many images at once (whole-folder analysis).

    Returns a compact result per image — predicted defect, confidence and the
    knowledge-grounded *reason* — plus a class-distribution summary. Uploaded
    images are saved so the gallery can display them and draw region boxes.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    images = []
    skipped = []
    for f in files:
        data = await f.read()
        name = f.filename or "upload"
        try:
            validate_upload(name, data)
            img = decode_bytes(data)
        except Exception as exc:  # skip unreadable files, keep the batch going
            skipped.append({"filename": name, "error": str(exc)})
            continue
        # persist so the frontend can render the thumbnail + boxes
        pid = new_id("batch")
        ext = Path(name).suffix.lower() or ".png"
        saved = settings.uploads_dir / f"{pid}{ext}"
        saved.write_bytes(data)
        images.append({"filename": name, "image_bgr": img,
                       "image_url": f"/uploads/{saved.name}"})

    if not images:
        raise HTTPException(status_code=400, detail="No valid images in batch")

    summary = predict_many(images)
    # predict_many preserves input order 1:1, so attach urls by index.
    for r, im in zip(summary["results"], images):
        r["image_url"] = im["image_url"]
    summary["skipped"] = skipped
    return JSONResponse(to_jsonable(summary))


@router.post("/build-graph")
async def build_graph(file: UploadFile = File(...)):
    """Convenience endpoint: run prediction and return only the graph payload."""
    resp = await predict(file)
    payload = json.loads(resp.body)
    graph = load_graph(payload["prediction_id"])
    return {"prediction_id": payload["prediction_id"], "graph": graph}


@router.get("/graph/{prediction_id}")
def get_graph(prediction_id: str):
    graph = load_graph(prediction_id)
    if graph is None:
        raise HTTPException(status_code=404, detail="Graph not found")
    return graph


@router.get("/prediction/{prediction_id}")
def get_prediction(prediction_id: str):
    if prediction_id in _RESULTS:
        return _RESULTS[prediction_id]
    path = _result_path(prediction_id)
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    raise HTTPException(status_code=404, detail="Prediction not found")


@router.get("/download-report/{prediction_id}")
def download_report(prediction_id: str):
    result = _RESULTS.get(prediction_id)
    if result is None:
        path = _result_path(prediction_id)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        with open(path, encoding="utf-8") as fh:
            result = json.load(fh)
    content = json.dumps(to_jsonable(result), indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition":
                f'attachment; filename="graphdefect_report_{prediction_id}.json"'
        },
    )
