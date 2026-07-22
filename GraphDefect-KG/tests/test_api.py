"""FastAPI endpoint tests using Starlette's TestClient."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def _png_bytes() -> bytes:
    import cv2
    rng = np.random.default_rng(3)
    img = rng.integers(0, 255, size=(160, 160, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "hybrid_trained" in body


def test_classes():
    r = client.get("/api/classes")
    assert r.status_code == 200
    assert len(r.json()["classes"]) == 6


def test_model_info():
    r = client.get("/api/model-info")
    assert r.status_code == 200
    assert r.json()["num_classes"] == 6


def test_predict_and_graph_roundtrip():
    files = {"file": ("test.png", io.BytesIO(_png_bytes()), "image/png")}
    r = client.post("/api/predict", files=files)
    assert r.status_code == 200, r.text
    result = r.json()
    pid = result["prediction_id"]

    # graph endpoint
    g = client.get(f"/api/graph/{pid}")
    assert g.status_code == 200
    graph = g.json()
    ids = {e["data"]["id"] for e in graph["elements"] if e["group"] == "nodes"}
    dangling = [e for e in graph["elements"] if e["group"] == "edges"
                and (e["data"]["source"] not in ids or e["data"]["target"] not in ids)]
    assert not dangling

    # prediction + report endpoints
    assert client.get(f"/api/prediction/{pid}").status_code == 200
    rep = client.get(f"/api/download-report/{pid}")
    assert rep.status_code == 200
    assert "attachment" in rep.headers.get("content-disposition", "")


def test_predict_rejects_bad_extension():
    files = {"file": ("bad.txt", io.BytesIO(b"not an image"), "text/plain")}
    r = client.post("/api/predict", files=files)
    assert r.status_code == 400


def test_graph_404():
    assert client.get("/api/graph/does_not_exist").status_code == 404
