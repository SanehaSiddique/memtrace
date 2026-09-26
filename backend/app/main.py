"""MEMTRACE FastAPI application."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.container import AppContainer
from app.api.routes import router
from app.config import settings
from app.tracing.langsmith import configure_langsmith

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTEND_DIR = _REPO_ROOT / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_langsmith(settings)
    container = AppContainer(settings)
    await container.initialize()
    app.state.container = container
    yield


app = FastAPI(title="MEMTRACE", description="Memory and context control layer for long-running AI agents.", lifespan=lifespan)

# Permissive CORS: local hackathon demo only, never meant to run like this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
