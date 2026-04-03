# 0) Репо, структура, середовище (Foundation)

## 0.1 Repo scaffold
- Створити repo rustdocs-rag
- Додати структуру:

```text
src/rustrag/{ingest,rag,api}
scripts/
tests/
data/{raw,processed,chunks,vectordb,eval}
```

- Додати `.gitignore` (ігнорити `.venv`, `data/raw`, `data/vectordb`, артефакти)
- Додати `.env.example` (моделі/ключі/шляхи)
- Додати `README.md` з “How to run”

## 0.2 Python env + dependencies
- Створити `python -m venv .venv`
- Встановити базові залежності:
  - parsing: `beautifulsoup4`, `lxml`
  - api: `fastapi`, `uvicorn`, `pydantic`
  - vector db: `chromadb` (для старту)
  - utils: `python-dotenv`, `tqdm`, `loguru` (опційно)
- Зробити “one command run”:

```bash
python scripts/ingest.py ...
python scripts/serve.py
```

## 0.3 Контракти даних (важливо)
- Створити `models.py`:
  - `Document(doc_id, title, source_path, text, metadata)`
  - `Chunk(chunk_id, doc_id, text, metadata)`
- Визначити обов’язкові metadata для Rust:
  - `crate` (std/core/alloc/book/…)
  - `item_path` (напр. std::vec::Vec::push)
  - `item_type` (fn/struct/trait/method/impl/module/book_section)
  - `rust_version` (якщо можеш визначити)
  - `anchor/section` (для цитат)

Готовність етапу 0: репо запускається, є venv, структура, моделі, README.

# 1) Ingest pipeline (офлайн): raw HTML → Documents → Chunks → Index

## 1.1 Джерело офлайн документації
- Визначити, звідки береш docs:
  - rustup doc (локальна папка)
  - або скачаний rustdoc сайт
- Покласти/послати у `data/raw/` (краще symlink, щоб не копіювати)

## 1.2 Discover (пошук html)
- `ingest/discover.py`: знайти всі `*.html`
- Додати фільтри:
  - виключити `search-index.js`, assets, 404 pages
  - для MVP можна індексувати лише std + book

## 1.3 Parse HTML → Document (MVP чистка)
- `ingest/parse_html.py`:
  - витягнути title
  - витягнути main content (видалити sidebar/nav/footer)
  - зберегти source_path
  - спробувати витягнути item_path (із breadcrumbs/заголовків)
- Записати `data/processed/docs.jsonl`

Критерій: ти можеш відкрити `docs.jsonl` і бачиш нормальний текст без меню/сміття.

## 1.4 Clean + normalize (v2 рівень якості)
- `ingest/clean.py`:
  - прибрати boilerplate (повторювані блоки)
  - нормалізувати пробіли/порожні рядки
  - зберегти code blocks (хоча б як fenced у тексті)

## 1.5 Chunking (спочатку простий → потім структурний)
- `ingest/chunk.py` MVP:
  - chunk size ~ 900–1200 символів
  - overlap 150–200
- `ingest/chunk.py` v2:
  - chunk по секціях rustdoc:
    - Description / Examples / Panics / Safety / Errors / Notes / Implementations
  - metadata section
- Записати `data/chunks/chunks.jsonl`

Критерій: чанк виглядає як логічний шматок, а не “обривок на півреченні”.

## 1.6 Dedup (v2)
- Прибрати дублікати чанків:
  - exact dedup: `hash(normalized_text)`
  - (опційно) near-dedup пізніше
- Лог: скільки чанків було і скільки лишилось

## 1.7 Index (Vector DB + embeddings)
- `ingest/index.py`:
  - створити/пересоздати індекс в `data/vectordb/<index_name>/`
  - додати text + metadata
- Версіонування індексу:
  - наприклад `index_std_1_78_0` або `index_std_latest`

На цьому етапі embeddings можуть бути через будь-який провайдер. Головне — зробити це замінним (в `embed.py` або функції `embed_texts()`).

## 1.8 Pipeline orchestration + CLI
- `ingest/pipeline.py`:
  - `--stage discover|parse|chunk|index|all`
  - `--limit N` (щоб тестити на 200 сторінках)
  - `--index-name`
- `scripts/ingest.py` як thin wrapper

Готовність етапу 1: ти можеш зробити `ingest --limit 200` і отримати індекс, який шукає.

# 2) Retrieval (runtime): пошук top-k + джерела (без LLM спочатку)

## 2.1 Retriever
- `rag/retriever.py`:
  - `search(query, k, filters)` → повертає top-k чанків:
    - chunk_text (snippet)
    - score
    - metadata (title/path/section/item_path)
- Фільтри:
  - `crate=std|book|…`
  - `item_type=trait|fn|method|…`

## 2.2 FastAPI /search
- `api/routes.py`:
  - `GET /search?q=...&k=5&crate=std`
- Повернення:
  - `results[] = {title, source_path, section, item_path, score, snippet}`

Критерій: ти вводиш Vec push panics і бачиш релевантні results.

# 3) LLM відповіді (RAG QA): /chat + citations + “I don’t know”

## 3.1 Prompting (базова політика)
- `rag/prompt.py`:
  - інструкція: “відповідай тільки з контексту”
  - формат: коротко + приклад коду якщо є + список джерел
  - якщо контекст слабкий → “Не знайшов у документації”

## 3.2 QA orchestrator
- `rag/qa.py`:
  - retrieve → build_context → call_llm → postprocess
- Heuristic “weak retrieval”:
  - якщо top score < threshold або занадто мало релевантних чанків → відмова

## 3.3 FastAPI /chat
- `POST /chat`:
  - input: question, filters
  - output: answer_markdown, sources[], debug (опційно)

Готовність етапу 3: ти отримуєш осмислені відповіді + 3–5 sources.

# 4) UI (web) + Debug Mode (портфоліо-рівень)

## 4.1 Мінімальний web чат (MVP UI)
- UI (на вибір: Streamlit/Gradio/Next.js):
  - input питання
  - answer markdown
  - Sources (картки): title + snippet + open

## 4.2 Фільтри (v2 UI)
- Дропдауни:
  - crate (std/book/…)
  - item_type
- Зберігати вибір у state

## 4.3 Debug mode (must-have для “готового” проєкту)
- Toggle “Debug”
- Показувати:
  - top-k chunks (collapsed)
  - scores
  - latency: retrieval / generation
  - застосовані filters

# 5) Якість: Hybrid Search + Re-ranking + Citations in text

## 5.1 Hybrid search (keyword + embeddings)
- Додати keyword index (простий варіант):
  - SQLite FTS або Whoosh
- Комбінувати:
  - взяти topN з embeddings + topN з keyword
  - merge + dedup
- Порівняти якість на eval сеті

## 5.2 Re-ranking (v3)
- Після merge взяти top-20 і прогнати reranker’ом (переставити релевантність)
- До LLM віддавати top-5 після rerank

## 5.3 Citations in text
- Нумерувати джерела [1], [2], [3]
- Вимагати, щоб кожен пункт відповіді мав посилання
- UI: клікаєш [2] → показує chunk + source

# 6) Evaluation + Feedback loop (щоб “готовий” виглядав серйозно)

## 6.1 Eval dataset
- `data/eval/questions.jsonl` (50–100 питань):
  - question
  - expected item_path (якщо можливо)
  - expected crate
- Скрипт `scripts/eval.py`:
  - метрики retrieval: recall@k, MRR
  - логувати погані кейси

## 6.2 Feedback в UI
- Кнопки: Helpful / Not helpful
- Причини: wrong source / missing / hallucination
- Зберігати в `data/feedback/` або SQLite

# 7) Продуктові штуки: streaming, кеші, інтеграції

## 7.1 Streaming відповіді
- API підтримує стрімінг токенів
- UI “друкує” відповідь

## 7.2 Кеші
- Кеш query embeddings
- Кеш retrieval результатів (на 5–15 хв)
- Лог latency + basic profiling

## 7.3 Інтеграції (v3 бонус)
- Telegram bot (thin client):
  - коротка відповідь + кнопка Sources
- або [ ] VS Code extension (тонкий клієнт до API)

# “Definition of Done” для повністю готового проєкту

Вважаємо проєкт завершеним, коли є:
- ✅ `scripts/ingest.py --stage all --index-name std_book`
- ✅ `scripts/serve.py` піднімає API
- ✅ UI дає чат + sources + debug
- ✅ є hybrid search + rerank (або хоча б hybrid)
- ✅ є eval скрипт + 50+ питань
- ✅ repo має README: install/run/ingest/eval
- ✅ дані не комітяться, але є `data/sample` для демо
