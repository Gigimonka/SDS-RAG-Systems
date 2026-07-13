#!/usr/bin/env bash

set -euo pipefail

ENV_FILE="/home/gigimon/project/wikijs/.env"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

API_KEY="${RAG_API_KEY:-sk-local-rag-change-me}"

echo "1. Проверяем модель:"
curl -sS \
  http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer ${API_KEY}"

echo
echo
echo "2. Проверяем chat/completions:"
curl -sS \
  -X POST \
  http://127.0.0.1:8000/v1/chat/completions \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sds-wiki-rag",
    "stream": false,
    "messages": [
      {
        "role": "user",
        "content": "Что такое лабораторный портал?"
      }
    ]
  }'

echo