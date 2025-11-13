
# backend/main.py
from __future__ import annotations

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers
from .api.routes_chat import router as chat_router

# Warmup targets
from .rag.retriever import get_retriever
from .services.llm_gemini import get_llm as get_gemini_llm

# Fireworks is optional — import defensively
try:
    from .services.llm_fireworks import get_llm as get_fireworks_llm  # type: ignore
except Exception:  # pragma: no cover
    get_fireworks_llm = None  # not configured / not installed

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    if v is not None:
        v = v.strip()
    return v


# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="RAG Chat Backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

# CORS (Streamlit on localhost)
frontend_origins = {
    "http://127.0.0.1:8501",
    "http://localhost:8501",
    _env("FRONTEND_ORIGIN") or "",
}
frontend_origins = {o for o in frontend_origins if o}  # drop empties

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(frontend_origins) or ["*"],  # relax during dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routes
app.include_router(chat_router)


# -----------------------------------------------------------------------------
# Health & Root
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "RAG Chat Backend",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"ok": True}


# -----------------------------------------------------------------------------
# Warmup utilities
# -----------------------------------------------------------------------------
def _warm_retriever():
    try:
        get_retriever()  # builds vector store + loads embedding model
        logger.info("[warmup] retriever ready")
    except Exception as e:  # pragma: no cover
        logger.warning(f"[warmup] retriever error: {e}")


def _warm_llms():
    # Gemini
    try:
        get_gemini_llm()
        logger.info("[warmup] gemini llm ready")
    except Exception as e:  # pragma: no cover
        logger.warning(f"[warmup] gemini error: {e}")

    # Fireworks (optional)
    if get_fireworks_llm:
        try:
            get_fireworks_llm()
            logger.info("[warmup] fireworks llm ready")
        except Exception as e:  # pragma: no cover
            logger.warning(f"[warmup] fireworks error: {e}")
    else:
        logger.info("[warmup] fireworks llm not configured; skipping")


@app.on_event("startup")
def _startup():
    """
    Preload heavy components so the first user request doesn't hit long
    model downloads or PG connections.
    """
    logger.info("[startup] warming up backend components...")
    _warm_retriever()
    _warm_llms()
    logger.info("[startup] warmup complete")


# Optional manual warmup endpoint if you want to trigger it yourself
@app.post("/warmup")
def manual_warmup():
    _warm_retriever()
    _warm_llms()
    return {"status": "warmed"}
