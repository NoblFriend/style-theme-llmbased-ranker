# Ranker

Репозиторий для оценки и ранжирования текстов по стилю и теме.

Основной сценарий: запуск из папок с `.txt` файлами через OpenAI-compatible API.

## Структура

- `src/` — основные скрипты
- `test/` — тестовые клиенты
- `tmp/` — разовые и вспомогательные скрипты
- `data/` — входные папки с текстами
- `results/` — выходные артефакты

## Установка

1. Установить `uv` (если не установлен):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Установить зависимости:

```bash
uv sync
```

## Быстрый старт (pipeline)

1. Заполнить API-конфиг в `config.yaml`, секция `api`:
   - `base_url` (OpenAI-compatible, обычно с `/v1`)
   - `model`
   - `api_key` (если требуется)
2. Подготовить папку в `data/` с `.txt` файлами.
3. Запустить pipeline:

```bash
python src/process_pipeline.py data/09_02_agit --topic astrophysics --output-base new_results
```

4. Забрать финальный файл:

- `new_results/09_02_agit/metrics.csv`

## Pipeline: что делает и как запускать

Pipeline обрабатывает каждую выбранную папку в 3 этапа:

1. `scoring` — выставляет pointwise-оценки для каждого текста.
2. `ranking` — ранжирует тексты в каждой строке.
3. `metrics` — считает метрики и объединяет scorer + ranker в общий `metrics.csv`.

### Команды pipeline

```bash
# обработать все папки из data/
python src/process_pipeline.py --topic astrophysics

# обработать только выбранные папки
python src/process_pipeline.py data/09_02_agit data/16_03_llama_sad --topic astrophysics

# свой каталог для результатов
python src/process_pipeline.py --topic astrophysics --output-base my_results

# автопоиск папок в другом каталоге
python src/process_pipeline.py --topic astrophysics --data-dir data
```

### Что появляется в output

Для папки `data/09_02_agit` в `results/09_02_agit/`:

1. `scored.csv`
2. `scored_detailed.csv`
3. `ranked.csv`
4. `scored_metrics.csv`
5. `ranked_metrics.csv`
6. `metrics.csv` (финальный объединенный результат)

## Запуск объединенного from-folder скрипта напрямую

Если нужно не весь pipeline, а конкретный режим:

```bash
# только scoring
python src/run_openai_from_folder.py data/09_02_agit --mode scoring --topic astrophysics --style melancholic

# только ranking
python src/run_openai_from_folder.py data/09_02_agit --mode ranking --topic astrophysics --style melancholic

# scoring + ranking подряд
python src/run_openai_from_folder.py data/09_02_agit --mode both --topic astrophysics --style melancholic
```

## Поднять локальный OpenAI-compatible сервер (vLLM)

Если нужно поднять собственный endpoint, в который потом будет ходить pipeline:

```bash
python src/run_vllm_server.py
```

Скрипт читает `config.yaml` и берет оттуда:
- `model.name` — что грузить в vLLM
- `model.device` — GPU(ы); поддерживает `cuda`, `cuda:2`, `cuda:0,1`, `[0,1]`
- `model.max_context_length` → `--max-model-len`
- `server.host` / `server.port` — где слушать
- `api.model` — имя модели в API (по умолчанию = `model.name`)

После запуска endpoint доступен по `http://{host}:{port}/v1`. Этот URL кладем в `api.base_url`, и дальше работает `process_pipeline.py`.

Полезные флаги: `--port`, `--gpu-memory-utilization 0.85`, `--dtype bfloat16`, `--extra ...` (доп. аргументы пробрасываются в `vllm serve`).

## Опционально: старый серверный режим

Старый серверный путь сохранен и находится в `src/rank_server.py`.

```bash
python src/rank_server.py
```

Тестовые клиенты:

```bash
python test/test_client.py
python test/test_scoring.py
python test/test_topic_ranking.py
```

## Типовые проблемы

- Ошибка конфигурации API: проверить `config.yaml` и переменные окружения (`OPENAI_BASE_URL`, `OPENAI_API_KEY`, `OPENAI_MODEL`).
- Ошибка HTTP/Auth: проверить `base_url`, ключ и доступность endpoint.
- Пустой результат: убедиться, что в папке есть непустые `.txt` и строки.
- Нет метрик: сначала должны успешно создаться `scored.csv` и `ranked.csv`.
