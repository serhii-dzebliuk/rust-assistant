# Architecture Overview

## Purpose

This project is a self-hosted RAG backend that answers questions over offline Rust
documentation.

Its main goal is to provide grounded answers by retrieving relevant documentation chunks
first and only then calling an LLM.

## Current implementation status

The repository is currently in a transition state:

- the serving runtime is still mostly stub-based
- the ingest pipeline still produces file artifacts such as JSONL outputs
- PostgreSQL and Qdrant are already part of the target architecture, but the full
  SQLAlchemy/Alembic integration is still being introduced

The architecture below describes the target structure that new persistence and migration
work should follow.

## Core components

### Caddy

Caddy is the public entrypoint.

Responsibilities:
- terminate HTTP/HTTPS traffic
- reverse proxy requests to the backend
- keep backend and databases off the public edge where possible

### FastAPI backend

The backend handles HTTP requests, request validation, orchestration, and response
generation.

Responsibilities:
- expose API endpoints
- coordinate retrieval and answer generation
- manage configuration, logging, and dependency wiring
- create request-scoped database sessions for Postgres access
- keep HTTP concerns separate from retrieval, persistence, and provider integrations

### PostgreSQL

PostgreSQL is the source of truth for normalized documentation data.

Responsibilities:
- store documents and chunks
- store chunk text and metadata
- store chunk synchronization state
- store relational application data used by both backend and ingest

PostgreSQL contains the canonical chunk text used to assemble LLM context.

### SQLAlchemy

SQLAlchemy is the application's Postgres integration layer.

Responsibilities:
- define ORM mappings for relational tables
- provide shared async engine and session management
- power repository implementations used by backend and ingest
- keep raw SQL and connection management out of routers and ingest parsing code

The project uses SQLAlchemy 2.0 async APIs with the `asyncpg` driver via
`postgresql+asyncpg://...` connection URLs.

### Alembic

Alembic is the schema migration tool for PostgreSQL.

Responsibilities:
- version relational schema changes
- run upgrades before backend or ingest workloads rely on new schema
- keep schema evolution explicit, reviewable, and reproducible

Alembic is the only supported mechanism for schema changes in shared environments.

### Qdrant

Qdrant is the vector retrieval layer.

Responsibilities:
- store embedding vectors
- perform nearest-neighbor search
- support lightweight metadata filtering for retrieval

Qdrant is not the source of truth for chunk text.

### LangChain

LangChain is used as an orchestration layer for LLM-related workflows.

Responsibilities:
- coordinate prompt construction and model calls
- support retrieval-to-generation flow
- keep model orchestration out of routers

### Raw documentation storage

Raw Rust documentation files live outside the main application logic and are treated as
ingest artifacts.

Responsibilities:
- provide local input for the ingest pipeline
- remain reproducible and replaceable
- stay out of version control when generated locally

## Data ownership model

The system intentionally separates vector retrieval from canonical content storage.

### PostgreSQL owns

- document identity
- chunk identity
- chunk text
- chunk metadata
- synchronization state
- relational application state

### Qdrant owns

- embeddings
- vector search index
- lightweight retrieval metadata needed for search

This split keeps retrieval fast while preserving a clean canonical data store.

## PostgreSQL integration with SQLAlchemy and Alembic

All Postgres access, from both the backend and the ingest pipeline, should go through the
same SQLAlchemy-based persistence layer.

### Access pattern

- `src/rust_assistant/core/db/base.py` defines the shared SQLAlchemy `DeclarativeBase` and
  metadata conventions
- `src/rust_assistant/core/db/session.py` creates the async engine and
  `async_sessionmaker` from `DATABASE_URL`
- `src/rust_assistant/models/` contains only SQLAlchemy ORM models
- `src/rust_assistant/repositories/` contains Postgres query and upsert logic built on top
  of `AsyncSession`
- `src/rust_assistant/services/` and ingest persistence orchestrators define transaction
  boundaries and call repositories
- `src/rust_assistant/api/deps.py` exposes request-scoped `AsyncSession` dependencies for
  FastAPI routes and services

### Async database driver

The runtime connection string should use the SQLAlchemy async dialect:

```text
DATABASE_URL=postgresql+asyncpg://postgres:change-me@postgres:5432/rust_assistant
```

Backend and ingest should both use the same SQLAlchemy async stack:

- `create_async_engine(...)`
- `async_sessionmaker(...)`
- `AsyncSession`

This keeps one consistent persistence model across API requests, ingest writes, and
future background jobs.

### Repository boundaries

Repositories are the only layer that should directly issue ORM queries.

Repository responsibilities:
- select canonical document and chunk rows
- insert or upsert ingest output into Postgres
- expose narrow persistence methods to services and ingest orchestration

Repository non-responsibilities:
- no HTTP concerns
- no LangChain orchestration
- no direct Qdrant writes
- no router wiring
- no transaction ownership beyond flush-level operations

Commits and rollbacks should be controlled by services or ingest persistence orchestration,
not by individual repository methods.

### ORM models

`src/rust_assistant/models/` is reserved for SQLAlchemy ORM models only.

Expected initial relational model:

- `documents`
  - canonical document metadata and source identity
  - one row per parsed documentation page
- `chunks`
  - canonical chunk text, ordering, section metadata, and document foreign key
  - one row per retrieval chunk

Additional sync-oriented fields may be added where useful, for example indexing status or
timestamps that indicate whether a canonical Postgres row has already been propagated to
Qdrant.

### Placement of non-ORM models

Non-ORM models should stay outside `src/rust_assistant/models/`, which is reserved for
SQLAlchemy ORM classes. The current project layout should follow this split:

- shared enums such as `Crate` and `ItemType` belong in
  `src/rust_assistant/schemas/enums.py` because they are also used by API filters
- ingest-only entities such as parsed documents, structured blocks, and chunking payloads
  belong in `src/rust_assistant/ingest/`, for example `ingest/entities.py`
- API request and response DTOs belong only in `src/rust_assistant/schemas/`
- legacy API DTOs in `models/` should be removed, not relocated into the ORM layer

### Alembic usage

Alembic lives at the repository root and manages the relational schema:

```text
alembic/
  env.py
  script.py.mako
  versions/
```

Alembic responsibilities:
- create and evolve the Postgres schema
- autogenerate candidate migrations from SQLAlchemy metadata when appropriate
- keep migration history in version control

Architectural rules for migrations:
- do not rely on `Base.metadata.create_all()` in backend startup
- do not create or alter shared schema from ad hoc runtime code
- run `alembic upgrade head` before backend or ingest workloads depend on a new schema

The preferred Alembic environment is the async template so migrations can use the same
`postgresql+asyncpg://...` URL family as the application runtime.

## Main flows

## Ingest flow

1. Raw documentation is downloaded or loaded from disk.
2. Source files are parsed into normalized ingest entities.
3. Documents are cleaned, deduplicated, and split into chunks.
4. The ingest persistence stage opens a batch-scoped `AsyncSession`.
5. Documents and chunks are written to PostgreSQL inside a transaction.
6. After Postgres commit succeeds, embeddings are generated and written to Qdrant.
7. A follow-up Postgres update records indexing or synchronization status when needed.

Design goals:
- reproducible
- idempotent where practical
- source-aware but normalized at the storage boundary
- resilient when Qdrant indexing fails after canonical Postgres writes

This ordering makes Postgres the canonical landing zone for ingest output. If vector
indexing fails, the system can retry Qdrant synchronization without reparsing the source
documents.

## Query flow

1. A user sends a question to the backend.
2. FastAPI resolves a request-scoped `AsyncSession` through API dependencies.
3. The backend generates an embedding for the question.
4. Qdrant returns the most relevant numeric `chunks.id` values.
5. Repository methods load canonical chunk text and metadata from PostgreSQL.
6. Retrieved chunks are assembled into model context.
7. LangChain orchestrates the LLM call.
8. The backend returns the answer.

This flow ensures the LLM answer is grounded in retrieved documentation rather than
relying only on model memory.

## Application module layout

Recommended backend structure:

```text
src/rust_assistant/
  main.py
  api/
  services/
  retrieval/
  ingest/
    parsing/
    entities.py
    persist.py
  clients/
  repositories/
  models/
  schemas/
    enums.py
  core/
    config.py
    logging.py
    db/
      base.py
      session.py
  utils/

alembic/
  env.py
  script.py.mako
  versions/
```

Responsibilities:

- `src/rust_assistant/main.py` - app creation and startup wiring
- `src/rust_assistant/api/` - routers, HTTP dependencies, request/response handling
- `src/rust_assistant/services/` - application and domain orchestration
- `src/rust_assistant/retrieval/` - retrieval flow, ranking, context assembly, Qdrant-facing
  orchestration
- `src/rust_assistant/ingest/` - parsing, chunking, ingest-domain entities, and ingest
  persistence orchestration
- `src/rust_assistant/clients/` - external integrations such as LLM, embedding, and vector
  store clients
- `src/rust_assistant/repositories/` - PostgreSQL persistence access via SQLAlchemy
- `src/rust_assistant/models/` - SQLAlchemy ORM models only
- `src/rust_assistant/schemas/` - Pydantic request/response schemas and shared enums
- `src/rust_assistant/core/` - config, logging, shared dependency wiring, and database setup
- `src/rust_assistant/utils/` - small generic helpers only
- `alembic/` - relational schema migrations

## Architectural rules

- Routers should call services, not directly implement retrieval or persistence logic
- Database access should stay out of routers
- LangChain orchestration should stay out of routers
- PostgreSQL is the source of truth for chunk text and metadata
- Qdrant is used for embeddings and retrieval only
- SQLAlchemy ORM models live only in `src/rust_assistant/models/`
- API schemas and shared enums live in `src/rust_assistant/schemas/`
- Ingest parsing and chunking entities live in `src/rust_assistant/ingest/`
- Repositories use `AsyncSession` and do not own commits
- Backend and ingest share the same SQLAlchemy engine/session conventions
- Alembic is the only supported mechanism for schema evolution
- Runtime configuration should come from a centralized settings module
- Public traffic should enter through Caddy
- Backend, Postgres, and Qdrant should communicate over internal Docker networks where
  possible

## Deployment model

The project is designed for self-hosted deployment on Ubuntu with Docker Compose.

Typical services:
- `proxy` - Caddy
- `backend` - FastAPI
- `postgres` - PostgreSQL
- `qdrant` - Qdrant

Recommended deployment order:

1. Start PostgreSQL and wait for health readiness.
2. Run `alembic upgrade head`.
3. Start backend and ingest workloads.
4. Allow backend requests only after dependencies and migrations are ready.

Deployment conventions:
- use `compose.yaml` as the main deployment definition
- persist Postgres and Qdrant data with Docker volumes
- keep backend containers stateless where possible
- load runtime configuration from environment variables
- use `postgresql+asyncpg://...` for `DATABASE_URL`
- keep secrets out of the repository

## Why this architecture

This design tries to balance:

- clarity - each component has one clear responsibility
- maintainability - API, retrieval, ingest, persistence, and infra stay separated
- self-hosting - all core services can run on a single Ubuntu server
- scalability - Qdrant and PostgreSQL can be tuned independently as the project grows
- schema safety - relational changes are explicit and versioned through Alembic
- testability - retrieval, ingest, repositories, and orchestration can be tested in
  isolation

## Future evolution

Likely directions:

- hybrid retrieval
- reranking
- richer metadata filtering
- background synchronization and retry jobs for Qdrant indexing
- operational dashboards and backup tooling
- support for additional documentation sources






