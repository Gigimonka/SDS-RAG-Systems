#!/usr/bin/env bash

set -euo pipefail

ENV_FILE="/home/gigimon/project/wikijs/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

cd /home/gigimon/project

export RAG_API_KEY="${RAG_API_KEY:-sk-local-rag-change-me}"

exec uv run uvicorn api:app \
  --app-dir help/rag-indexer \
  --host 0.0.0.0 \
  --port 8000