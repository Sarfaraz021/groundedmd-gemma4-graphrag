"""
FastAPI application entry point.

Manages Neo4j driver lifecycle and GraphRAG instance as app state,
shared across all requests without re-initialisation.
"""

import langsmith_env  # noqa: F401 — load .env + LangSmith before other imports

import os
from contextlib import asynccontextmanager

import neo4j
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from retrieval.retriever import build_graph_rag, setup_vector_index

_REQUIRED_ENV = ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")


def _validate_env() -> None:
    missing = [k for k in _REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Set them in your .env file or deployment config before starting the server."
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _validate_env()
    # Recycle connections before load balancers close idle sockets.
    try:
        max_lifetime = int(os.environ.get("NEO4J_MAX_CONNECTION_LIFETIME", "3300"))
    except ValueError:
        max_lifetime = 3300
    driver = neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
        max_connection_lifetime=max_lifetime,
        notifications_min_severity="WARNING",
        notifications_disabled_classifications=["UNRECOGNIZED"],
    )
    setup_vector_index(driver)

    app.state.neo4j_driver = driver
    app.state.graph_rag = build_graph_rag(driver)

    yield

    driver.close()


# Explicit allowed origins — no broad subnet wildcard.
# Add production origins via CORS_ORIGINS env var (comma-separated).
_dev_origins = [
    "http://localhost:8080",
    "http://localhost:5173",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173",
]
_extra = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
_allowed_origins = _dev_origins + _extra

app = FastAPI(
    title="GroundedMD GraphRAG API",
    description="Evidence-grounded TBI Q&A powered by Neo4j GraphRAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)

app.include_router(router)
