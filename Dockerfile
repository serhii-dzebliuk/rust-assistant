FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY README.md ./
COPY src ./src

RUN python -m pip install --upgrade pip build \
    && python -m pip wheel --wheel-dir /tmp/wheels .


FROM python:3.12-slim as runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system appuser \
    && useradd --system --gid appuser --create-home --home-dir /app appuser

COPY --from=builder /tmp/wheels /tmp/wheels

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir /tmp/wheels/*.whl \
    && rm -rf /tmp/wheels

USER appuser

CMD ["sh", "-c", "exec python -m uvicorn rust_assistant.asgi:app --host \"$HOST\" --port \"$PORT\""]
