# Architecture Overview

## Purpose

This project is a self-hosted RAG backend that answers questions over offline Rust documentation.

Its main goal is to provide grounded answers by retrieving relevant documentation chunks first and only then calling an LLM.

## Core components

### Caddy

Caddy is the public entrypoint.

Responsibilities:
- terminate HTTP/HTTPS traffic
- reverse proxy requests to the backend
- keep backend and databases off the public edge where possible

### FastAPI backend

The backend handles HTTP requests, request validation, orchestration, and response generation.

Responsibilities:
- expose API endpoints
- coordinate retrieval and answer generation
- manage configuration, logging, and dependency wiring
- keep HTTP concerns separate from retrieval, persistence, and provider integrations

### PostgreSQL

PostgreSQL is the source of truth for normalized documentation data.

Responsibilities:
- store documents and chunks
- store chunk text and metadata
- store ingest bookkeeping and relational application data

PostgreSQL should contain the canonical chunk text used to assemble LLM context.

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

Raw Rust documentation files live outside the main application logic and are treated as ingest artifacts.

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
- ingest bookkeeping
- relational state

### Qdrant owns

- embeddings
- vector search index
- lightweight retrieval metadata

This split keeps retrieval fast while preserving a clean canonical data store.

## Main flows

## Ingest flow

1. Raw documentation is downloaded or loaded from disk
2. Source files are parsed into a normalized internal representation
3. Documents are split into chunks
4. Embeddings are generated for chunks
5. Chunks and metadata are stored in PostgreSQL
6. Embeddings and retrieval metadata are stored in Qdrant

Design goals:
- reproducible
- idempotent where practical
- source-aware but normalized at the storage boundary

## Query flow

1. A user sends a question to the backend
2. The backend generates an embedding for the question
3. Qdrant returns the most relevant chunk IDs
4. The backend loads chunk text and metadata from PostgreSQL
5. Retrieved chunks are assembled into model context
6. LangChain orchestrates the LLM call
7. The backend returns the answer

This flow ensures the LLM answer is grounded in retrieved documentation rather than relying only on model memory.

## Application module layout

Recommended backend structure:

```text
src/rust_assistant/
  main.py
  api/
  services/
  retrieval/
  ingest/
  clients/
  repositories/
  models/
  schemas/
  core/
  utils/
```

Responsibilities:

- `src/rust_assistant/main.py` — app creation and startup wiring
- `src/rust_assistant/api/` — routers, HTTP dependencies, request/response handling
- `src/rust_assistant/services/` — application and domain logic
- `src/rust_assistant/retrieval/` — retrieval flow, ranking, context assembly
- `src/rust_assistant/ingest/` — parsing, chunking, embedding, pipeline steps
- `src/rust_assistant/clients/` — external integrations such as LLM and embedding providers
- `src/rust_assistant/repositories/` — PostgreSQL persistence access
- `src/rust_assistant/models/` — database models
- `src/rust_assistant/schemas/` — shared Pydantic schemas and DTOs
- `src/rust_assistant/core/` — config, logging, shared wiring
- `src/rust_assistant/utils/` — small generic helpers only

## Architectural rules

- Routers should call services, not directly implement retrieval or persistence logic
- Database access should stay out of routers
- LangChain orchestration should stay out of routers
- PostgreSQL is the source of truth for chunk text and metadata
- Qdrant is used for embeddings and retrieval only
- Runtime configuration should come from a centralized settings module
- Public traffic should enter through Caddy
- Backend, Postgres, and Qdrant should communicate over internal Docker networks where possible

## Deployment model

The project is designed for self-hosted deployment on Ubuntu with Docker Compose.

Typical services:
- `proxy` — Caddy
- `backend` — FastAPI
- `postgres` — PostgreSQL
- `qdrant` — Qdrant

Deployment conventions:
- use `compose.yaml` as the main deployment definition
- persist Postgres and Qdrant data with Docker volumes
- keep backend containers stateless where possible
- load runtime configuration from environment variables
- keep secrets out of the repository

## Why this architecture

This design tries to balance:

- **clarity** — each component has one clear responsibility
- **maintainability** — API, retrieval, ingest, persistence, and infra stay separated
- **self-hosting** — all core services can run on a single Ubuntu server
- **scalability** — Qdrant and PostgreSQL can be tuned independently as the project grows
- **testability** — retrieval, ingest, and orchestration can be tested in isolation

## Future evolution

Likely directions:

- hybrid retrieval
- reranking
- ingest versioning
- richer metadata filtering
- operational dashboards and backup tooling
- support for additional documentation sources
