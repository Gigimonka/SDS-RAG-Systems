"""Console entry points for the installed package."""

from __future__ import annotations

import os

import uvicorn


def serve() -> None:
    """Run the API using host and port values from the environment."""
    uvicorn.run(
        "sds_rag.api.app:app",
        host=os.getenv("RAG_HOST", "0.0.0.0"),
        port=int(os.getenv("RAG_PORT", "8000")),
    )
