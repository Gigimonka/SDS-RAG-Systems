# Wiki.js RAG

Hybrid RAG-система для поиска и ответов по внутренней документации Wiki.js.
Dense-поиск BGE-M3 и sparse-поиск BM25 объединяются в Qdrant через RRF,
после чего FastAPI передаёт найденный контекст в vLLM. API совместим с OpenAI
Chat Completions и может использоваться из Open WebUI.

## Структура проекта

```text
.
├── src/sds_rag/             устанавливаемый Python-пакет
│   ├── api/                 FastAPI, endpoints и Pydantic-схемы
│   ├── core/                конфигурация и общие правила
│   └── indexing/            индексация, миграция и поиск
├── tests/                   автоматические тесты
├── scripts/                 shell-команды запуска и smoke-тестов
├── tools/                   конвертер документации и диагностика
├── data/                    приватные исходники и Wiki.js-экспорт
├── docker-compose.yml       локальная инфраструктура
├── .env.example             шаблон переменных окружения
├── pyproject.toml           метаданные и зависимости Python
└── README.md
```

Используется стандартный `src`-layout: код из корня репозитория случайно не
импортируется, а тестируется именно установленный пакет `sds-rag`.

## Подготовка документации из source

Исходный экспорт Help&Manual нужно полностью распаковать непосредственно в
`data/source`. Дополнительная папка с названием проекта вокруг этих каталогов
не нужна. Ожидаемая структура:

```text
data/source/
├── Images/
├── Maps/
│   └── table_of_contents.xml
├── Topics/
└── Baggage/                  необязательно
```

Каталоги `Images`, `Maps` и `Topics` обязательны. Важно дождаться полного
копирования файлов: обрезанные XML будут отмечены как ошибки разбора и вместо
их содержимого в Wiki.js появятся диагностические страницы.

Запустить конвертацию из корня репозитория:

```bash
uv run python tools/convert_helpman_to_wikijs.py
```

Конвертер создаёт:

- `data/wikijs_export/` — Markdown-страницы и изображения для Wiki.js и RAG;
- `data/wikijs_export-conversion-report.json` — статистику, ошибки XML,
  отсутствующие изображения и битые ссылки.

При повторном запуске содержимое `data/wikijs_export` пересоздаётся, поэтому
его не следует редактировать вручную. Исходники и результат конвертации
добавлены в `.gitignore` и в Git не отправляются.

При необходимости можно переопределить переменные Help&Manual:

```bash
uv run python tools/convert_helpman_to_wikijs.py \
    --brandname 'МИС "Инфоклиника"/"Инфодент"' \
    --compilation-date 13.07.2026
```

После каждого обновления `data/source` порядок действий такой:

1. Повторно запустить конвертер.
2. В Wiki.js выполнить **Storage → Local File System → Import Everything**.
3. Пересоздать dense- и hybrid-индексы командами `uv run sds-rag-index` и
   `uv run sds-rag-migrate --recreate`.

## Запуск

Все команды выполняются из корня репозитория.

### Настройка RAG

1. До первого запуска Docker Compose и RAG API создать файл настроек:

   ```bash
   cp .env.example .env
   ```

   Один раз сгенерировать два разных случайных значения:

   ```bash
   openssl rand -hex 32
   openssl rand -hex 32
   ```

   Первый результат записать в `RAG_API_KEY`, второй — в
   `WEBUI_SECRET_KEY` в корневом файле `.env`:

   ```dotenv
   RAG_API_KEY=sk-local-<первый-результат>
   WEBUI_SECRET_KEY=<второй-результат>
   ```

   `RAG_API_KEY` защищает OpenAI-совместимый RAG API и передаётся в Open
   WebUI как API-ключ. `WEBUI_SECRET_KEY` защищает пользовательские сессии
   Open WebUI и должен оставаться постоянным. Docker Compose автоматически
   читает корневой `.env`, а `scripts/start_rag_api.sh` загружает тот же файл
   перед запуском API. Эти значения нельзя оставлять равными `change-me`,
   публиковать или добавлять в Git.

   Если секреты были изменены после запуска сервисов, пересоздать Open WebUI:

   ```bash
   docker compose up -d --force-recreate open-webui
   ```

   Запущенный RAG API также нужно остановить и снова запустить командой
   `./scripts/start_rag_api.sh`. После изменения `WEBUI_SECRET_KEY`
   существующие пользовательские сессии будут завершены.

2. Установить зависимости основного приложения:

   ```bash
   uv sync
   ```

3. Запустить инфраструктуру:

   ```bash
   docker compose up -d
   ```

4. Построить исходный dense-индекс:

   ```bash
   uv run sds-rag-index
   ```

5. Создать hybrid-коллекцию:

   ```bash
   uv run sds-rag-migrate --recreate
   ```

### Настройка Wiki.js

Перед первым запуском заменить `change_me_strong_password` в настройках
PostgreSQL и Wiki.js в `docker-compose.yml`. Если база уже инициализирована,
пароль нельзя просто поменять только в Compose: сначала его нужно изменить в
PostgreSQL.

После `docker compose up -d` открыть <http://localhost:3000> и создать
учётную запись администратора. Экспорт из `data/wikijs_export` уже подключён
к контейнеру как `/wiki-import`, но автоматически в базу Wiki.js он не
загружается.

Для импорта открыть **Administration → Storage → Local File System** и
указать:

```text
Active: On
Path: /wiki-import
```

Нажать **Apply**, затем выполнить **Actions → Import Everything → Run**.
При первоначальной загрузке не следует запускать **Dump All Content to Disk**,
чтобы пустое содержимое Wiki.js не было записано поверх подготовленного
экспорта. Ход импорта можно смотреть командой:

```bash
docker compose logs -f wiki
```

После завершения проверить главную страницу <http://localhost:3000> и раздел
<http://localhost:3000/infoclinica>, включая отображение изображений. RAG
индексирует Markdown непосредственно из `data/wikijs_export`; импорт в Wiki.js
нужен для просмотра документации и открытия ссылок на источники.

### Установка vLLM

Основное приложение использует Python 3.14, а vLLM устанавливается в
отдельное окружение с Python 3.12. Это также изолирует поставляемые с vLLM
версии PyTorch и CUDA-библиотек от зависимостей RAG. Актуальные требования
приведены в [документации vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

Сначала убедиться, что NVIDIA GPU доступен из WSL2:

```bash
nvidia-smi
```

Затем один раз создать окружение и установить vLLM:

```bash
uv venv .venv-vllm \
    --python 3.12 \
    --seed \
    --managed-python

uv pip install \
    --python .venv-vllm/bin/python \
    vllm \
    --torch-backend=auto
```

### Запуск сервисов

В первом терминале запустить OpenAI-совместимый сервер vLLM. Активировать
окружение не требуется:

```bash
VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
.venv-vllm/bin/vllm serve Qwen/Qwen3-8B-AWQ \
    --host 127.0.0.1 \
    --port 8001 \
    --dtype half \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.72 \
    --default-chat-template-kwargs '{"enable_thinking": false}'
```

Первый запуск скачает модель с Hugging Face. После загрузки проверить сервер:

```bash
curl http://127.0.0.1:8001/v1/models
```

Значения в `.env` должны соответствовать запущенному серверу:

```dotenv
LLM_URL=http://127.0.0.1:8001/v1
LLM_MODEL=Qwen/Qwen3-8B-AWQ
```

Во втором терминале запустить RAG API:

```bash
./scripts/start_rag_api.sh
```

Если vLLM сообщает о нехватке видеопамяти, сначала уменьшить
`--max-model-len` до `8192`. При необходимости также уменьшить
`--gpu-memory-utilization` до `0.65`.

## Проверки

```bash
uv run python -m unittest discover -s tests -v
uv run python tools/check_hybrid_search.py
./scripts/test_openai_api.sh
```

Последние две команды требуют запущенного API и Qdrant. Полный список
настроек RAG находится в `src/sds_rag/core/config.py`.

## Безопасность

Файлы `.env`, исходная документация и сгенерированный экспорт не хранятся в
Git. Не используйте стандартные значения секретов в рабочем окружении.
