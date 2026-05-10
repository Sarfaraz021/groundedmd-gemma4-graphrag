"""
Load environment variables and configure LangSmith tracing.

Import this module before any code that invokes GraphRAG / LLMs so tracing keys
are available. LangChain and LangSmith Python SDKs read LANGCHAIN_* names;
this maps LANGSMITH_* from .env to those names.

Usage (first import in entrypoints):
    import langsmith_env  # noqa: F401
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

_BACKEND_ROOT = Path(__file__).resolve().parent

# Prefer backend/.env, then process cwd
load_dotenv(_BACKEND_ROOT / ".env")
load_dotenv()


def _truthy(val: str | None) -> bool:
    if not val:
        return False
    return val.strip().lower() in ("1", "true", "yes", "on")


def configure_langsmith_tracing() -> None:
    """Sync LANGSMITH_* → LANGCHAIN_* so LangSmith + LangChain tracing picks them up."""
    if _truthy(os.getenv("LANGSMITH_TRACING")):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"

    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if api_key:
        os.environ["LANGCHAIN_API_KEY"] = api_key

    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT")
    if project:
        os.environ["LANGCHAIN_PROJECT"] = project

    endpoint = os.getenv("LANGSMITH_ENDPOINT") or os.getenv("LANGCHAIN_ENDPOINT")
    if endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = endpoint


configure_langsmith_tracing()
