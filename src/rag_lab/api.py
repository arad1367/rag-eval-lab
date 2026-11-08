"""FastAPI web service.

Concepts:
- REST API: clients send JSON over HTTP; we return JSON.
- Pydantic models validate input automatically (wrong type or empty question -> HTTP 422).
- Lifespan: heavy work (loading data, embedding chunks) happens ONCE at startup,
  not on every request.
- /health endpoint: used by Docker, load balancers and uptime monitors.
- create_app(pipeline=...) lets tests inject a fake pipeline (dependency injection).
- Run locally:  uvicorn rag_lab.api:app --reload --port 8000   then open /docs
"""

import logging
import time
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_lab")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(default=4, ge=1, le=10)
    retriever: Literal["bm25", "dense", "hybrid"] | None = None


def build_default_pipeline():
    """Build the real pipeline: cached index (chunks + vectors) -> retrievers + LLM."""
    from rag_lab.config import settings
    from rag_lab.index import load_index
    from rag_lab.llm import LLMClient
    from rag_lab.rag import RAGPipeline
    from rag_lab.retrievers import build_retrievers

    chunks, embedder, matrix = load_index()
    logger.info("Index has %d chunks", len(chunks))
    retrievers = build_retrievers(chunks, embedder, matrix)
    return RAGPipeline(retrievers, LLMClient(), default_retriever=settings.retriever)


def create_app(pipeline=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline or build_default_pipeline()
        logger.info("Ready.")
        yield

    app = FastAPI(title="RAG Eval Lab", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.0f ms)", request.method, request.url.path, response.status_code, ms
        )
        return response

    @app.get("/", response_class=HTMLResponse)
    def home():
        return files("rag_lab").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ask")
    def ask(body: AskRequest, request: Request):
        try:
            result = request.app.state.pipeline.answer(body.question, body.k, body.retriever)
        except Exception as exc:  # never leak a stack trace to the client
            logger.exception("Pipeline failed")
            raise HTTPException(status_code=502, detail=f"Pipeline error: {exc}") from exc
        logger.info(
            "ask retriever=%s tokens=%s latency=%s",
            result["retriever"],
            result["usage"],
            result["latency_ms"],
        )
        return result

    return app


app = create_app()
