# Wiki.js RAG

RAG-система для поиска и ответов по документации Wiki.js.

## Компоненты

- Wiki.js
- Qdrant
- FastAPI
- BGE-M3
- BM25 hybrid search
- vLLM
- Open WebUI

## Настройка

1. Скопировать `wikijs/.env.example` в `wikijs/.env`.
2. Заменить стандартные секреты.
3. Запустить сервисы из `wikijs/docker-compose.yml`.
4. Создать индекс документации.
5. Запустить `help/rag-indexer/start_rag_api.sh`.

## Безопасность

Файлы `.env`, исходная документация и сгенерированный экспорт не хранятся в Git.