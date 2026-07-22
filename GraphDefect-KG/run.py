"""Entry point for the GraphDefect-KG FastAPI application.

Usage:
    python run.py                 # start the server (http://127.0.0.1:8000)
    python run.py --host 0.0.0.0 --port 8080

This adds ``backend`` to ``sys.path`` so the package imports resolve whether the
project is launched from the repo root or from within ``GraphDefect-KG``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> None:
    import uvicorn
    from backend.config import settings

    parser = argparse.ArgumentParser(description="Run the GraphDefect-KG server")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    args = parser.parse_args()

    print(f"[GraphDefect-KG] starting on http://{args.host}:{args.port}")
    print(f"[GraphDefect-KG] device: {settings.resolve_device()}")
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
