"""FastAPI application factory for GraphDefect-KG.

Serves the JSON API under ``/api`` and the static light-themed frontend at the
root. Model construction is deferred to the first request (or an explicit
warm-up) so the server starts quickly.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import router as api_router
from .config import settings
from .utils.helpers import get_logger

logger = get_logger("graphdefect")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    # Ensure the knowledge graph file exists on startup (cheap).
    from .graph.knowledge_graph import KnowledgeGraph
    KnowledgeGraph.load_or_build()
    logger.info("%s v%s ready | device=%s", settings.app_name,
                settings.app_version, settings.resolve_device())
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Knowledge-Guided Graph Neural Learning for Explainable "
                "Industrial Surface Defect Detection",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# Serve uploaded images (so the results page can display them).
app.mount("/uploads", StaticFiles(directory=str(settings.uploads_dir)), name="uploads")

# Serve the frontend static assets (css / js / assets).
frontend = settings.frontend_dir
if frontend.exists():
    app.mount("/css", StaticFiles(directory=str(frontend / "css")), name="css")
    app.mount("/js", StaticFiles(directory=str(frontend / "js")), name="js")
    app.mount("/assets", StaticFiles(directory=str(frontend / "assets")), name="assets")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(str(frontend / "index.html"))


@app.get("/results.html", include_in_schema=False)
def results():
    return FileResponse(str(frontend / "results.html"))


@app.get("/batch.html", include_in_schema=False)
def batch():
    return FileResponse(str(frontend / "batch.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    icon = frontend / "assets" / "icons" / "favicon.ico"
    if icon.exists():
        return FileResponse(str(icon))
    return FileResponse(str(frontend / "index.html"))  # harmless fallback
