# Moodle RAG Chat-bot

RAG-система для поиска ответов в документации Moodle с поддержкой контекста диалога.

## Структура проекта

Основные файлы: app.py содержит FastAPI приложение с RAG pipeline, config.py - конфигурация портов и моделей, docker-compose.yaml - конфигурация Qdrant. 

Данные: scraped_pages.json содержит обработанную документацию Moodle, buffer_table.csv - подготовленные данные для векторизации, qdrant_data/ - данные векторной базы.

Обработка данных: в data_notebooks/ находятся prepare_data.ipynb для парсинга документации, embeddings.ipynb для создания эмбеддингов, fill_qdrant.ipynb для загрузки данных в Qdrant.

## Установка и запуск

### 1. Синхронизация окружения
```bash
uv sync
```

### 2. Запуск Qdrant
```bash
docker compose up -d
```

### 3. Подготовка данных
```bash
# Предобработка JSON в CSV
uv run prepare_data.py --input data/scraped_pages.json --output data/buffer_table.csv --chunk-size 300
# Заполнение Qdrant
uv run fill_qdrant.py --csv-path data/buffer_table.csv --collection-name Docs_Dense --embedding-batch-size 2048    
```

### 4. Запуск языковой модели
```bash
vllm serve "Qwen/Qwen3-8B" --gpu-memory-utilization 0.19 --max_model_len 5000
```

### 5. Запуск модели эмбеддингов
```bash
vllm serve jinaai/jina-embeddings-v4-vllm-retrieval \
    --task embed \
    --gpu-memory-utilization 0.15 \
    --max-model-len 3500 \
    --trust-remote-code \
    --dtype auto --port 8004
```

### 6. Запуск RAG API
```bash
uv run -- uvicorn app:app --reload --port 7998 --host 127.0.0.1    
```

## Тестирование

### Простой запрос
```python
import requests

response = requests.post("http://localhost:7999/rag", json={
    "query": "Как массово задать домашку ученикам?",
    "k": 3,
    "n": 3,
    "top": 15
})

print(response.json())
```

## API Endpoints

POST /rag - основной endpoint для RAG-запросов
localhost:7998/docs - UI для тестирования api

Параметры:
- query (str): вопрос пользователя
- history (list, optional): история диалога в формате [{role, content}]
- k, n, top (int): параметры поиска

Ответ: {answer: str, context: list}