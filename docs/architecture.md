# Architecture Overview

## Purpose

This project is a self-hosted RAG backend that answers questions over offline Rust
documentation.

Its main goal is to provide grounded answers by retrieving relevant documentation chunks
first and only then calling an LLM.

## Clean Hexagonal Architecture

The system follows a Clean Hexagonal Architecture.

The core of the system is formed by `domain` and `application`:

- `domain` models the problem space, business rules, value objects, policies, and domain
  errors
- `application` implements use cases, orchestration, input/output DTOs, and ports for
  external dependencies

All frameworks, SDKs, databases, and external services live outside the core in
`infrastructure`. `bootstrap` selects concrete implementations and wires the dependency
graph together.

Dependency direction:

```text
domain <- application <- infrastructure
                 ^
                 |
             bootstrap
```

This means the core does not depend on FastAPI, SQLAlchemy, Alembic, OpenAI, Qdrant,
`tiktoken`, or any other external tool. Those concerns are isolated behind ports and
adapters.

## Architectural Layers

### Domain

`domain` contains the pure business model of the system.

Typical contents:
- entities such as `Document` and `Chunk`
- enums and value objects such as `DocumentStatus`, `ItemType`, and `Crate`
- domain rules, normalization logic, and policies
- domain-specific exceptions and invariants

`domain` must not import `application`, `infrastructure`, or `bootstrap`.

### Application

`application` defines what the system does through use cases.

Typical contents:
- `use_cases/` for chat, search, ingest, and related orchestration
- `policies/` for application-level rules that are not stable domain-wide rules
- `ports/` for contracts such as `LLMClient`, `InferenceClient`, `Tokenizer`,
  `VectorStore`, `DocumentRepository`, and `ChunkRepository`
- `dto/` for internal input/output models used by use cases

`application` depends on `domain` and on port abstractions only. It does not import
concrete adapters or framework-specific code.

Use cases follow a command/result pattern:
- class names end with `UseCase`
- `execute(...)` receives one command DTO
- `execute(...)` returns one result DTO

### Infrastructure

`infrastructure` contains all adapters around the application core.

Entrypoints receive external input and call use cases:
- HTTP API
- CLI commands
- background jobs and ingest entrypoints

Adapters implement ports and integrate with external systems:
- PostgreSQL and SQLAlchemy
- Qdrant
- LLM and inference providers such as OpenAI
- tokenizers such as `tiktoken`
- filesystem access and other operational integrations

### Bootstrap

`bootstrap` is responsible for configuration and dependency assembly.

Responsibilities:
- load and validate runtime settings
- choose concrete adapter implementations for each port
- build use case instances and dependency graphs
- expose wiring helpers for API startup and ingest execution

## Target Module Layout

The target package layout is:

```text
src/rust_assistant/
  asgi.py
  __main__.py
  domain/
  application/
    ports/
    use_cases/
    policies/
    dto/
  infrastructure/
    entrypoints/
      api/
        routers/
        schemas/
      cli/
    adapters/
  bootstrap/
    settings.py
    logging.py
    container.py
    api.py
    ingest.py

alembic/
  env.py
  script.py.mako
  versions/
```

Responsibilities:

- `src/rust_assistant/asgi.py` - public ASGI entrypoint that exposes the FastAPI app
- `src/rust_assistant/__main__.py` - public package CLI entrypoint
- `src/rust_assistant/domain/` - domain entities, value objects, enums, policies, and
  domain errors
- `src/rust_assistant/application/ports/` - contracts for external dependencies
- `src/rust_assistant/application/use_cases/` - scenario orchestration for chat,
  search, ingest, and related operations
- `src/rust_assistant/application/policies/` - application-level rules used by use
  cases and adapter mapping logic
- `src/rust_assistant/application/dto/` - internal request/result models for use cases
- `src/rust_assistant/infrastructure/entrypoints/api/` - FastAPI routers and HTTP schemas
- `src/rust_assistant/infrastructure/entrypoints/cli/` - CLI and job entrypoints
- `src/rust_assistant/infrastructure/adapters/` - SQLAlchemy, Qdrant, OpenAI,
  tokenizer, and other adapter implementations
- `src/rust_assistant/bootstrap/` - centralized settings, logging, and dependency
  wiring
- `alembic/` - relational schema migrations for the PostgreSQL adapter

## Model Boundaries

The architecture keeps internal models separate from transport and persistence models.

- Domain entities are not ORM models.
- Domain entities are not API request or response schemas.
- Application DTOs are internal use-case contracts.
- API schemas exist only for HTTP transport concerns.
- ORM models exist only for relational persistence concerns.

Rules:
- `application/dto` is the only place for use-case input/output models
- `infrastructure/entrypoints/api/schemas` is only for external HTTP request and response
  models
- `infrastructure/adapters/.../models` is only for ORM or adapter-specific persistence
  mappings

This separation prevents framework-specific structures from leaking into the core.

## Runtime Components

### Caddy

Caddy is the public entrypoint.

Responsibilities:
- terminate HTTP/HTTPS traffic
- reverse proxy requests to the backend
- keep backend and data services off the public edge where possible

### FastAPI API adapter

FastAPI is an entrypoint in `infrastructure/entrypoints/api/`.

Responsibilities:
- expose HTTP endpoints
- validate and map HTTP requests into application DTOs
- call application use cases
- map use-case results and failures into HTTP responses

FastAPI is not part of the application core.

### PostgreSQL and SQLAlchemy persistence adapter

PostgreSQL is the canonical relational store. SQLAlchemy is the persistence
adapter used to access it.

Responsibilities:
- store canonical documents and chunks
- store chunk text and metadata
- store chunk synchronization state and relational application data
- implement repository ports behind SQLAlchemy-based adapters

PostgreSQL contains the canonical chunk text used to assemble LLM context.

### Alembic

Alembic manages relational schema evolution for the PostgreSQL adapter.

Responsibilities:
- version relational schema changes
- run upgrades before runtime workloads rely on new schema
- keep schema evolution explicit, reviewable, and reproducible

Alembic is the supported mechanism for shared schema changes.

### Qdrant vector adapter

Qdrant is the vector storage and retrieval adapter.

Responsibilities:
- store embeddings
- perform nearest-neighbor search
- support lightweight metadata filtering for retrieval

Qdrant is not the source of truth for chunk text.

### LLM, inference, and tokenizer adapters

LLM providers, inference clients, and tokenizers are adapters behind
application ports.

Responsibilities:
- generate embeddings
- perform answer generation or ranking calls
- provide token counting or tokenization services required by use cases

### Raw documentation storage

Raw Rust documentation files are ingest inputs and operational artifacts.

Responsibilities:
- provide reproducible source material for ingest
- remain replaceable and source-oriented
- stay separate from application core logic

## Data Ownership Model

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

Qdrant payloads should not duplicate canonical chunk text. Store only lightweight filter
metadata such as chunk id, document id, crate, item type, Rust version, source path,
item path, chunk index, and chunk hash. PostgreSQL remains the source of truth for text
and source attribution.

Document and chunk business identities are deterministic UUIDs derived in the application
layer. Database integer primary keys are technical persistence keys only.

This split keeps retrieval fast while preserving a clean canonical data store.

## Persistence and Migrations

All PostgreSQL access should go through adapters that implement application
repository ports.

### Persistence boundaries

Repository adapter responsibilities:
- select canonical document and chunk rows
- insert or upsert ingest output into PostgreSQL
- expose narrow persistence operations to use cases

Repository adapter non-responsibilities:
- no HTTP concerns
- no FastAPI wiring
- no direct LLM orchestration
- no direct Qdrant orchestration unless explicitly modeled through a port

Transaction boundaries should be controlled by use cases or dedicated application
orchestration, not by low-level repository methods.

### Async database driver

The runtime connection string should use the SQLAlchemy async dialect:

```text
DATABASE_URL=postgresql+asyncpg://postgres:change-me@postgres:5432/rust_assistant
```

The PostgreSQL adapter should use:

- `create_async_engine(...)`
- `async_sessionmaker(...)`
- `AsyncSession`

### ORM model placement

ORM classes belong only in infrastructure-level persistence modules. They are adapter
implementation details and must not be used as domain entities or application DTOs.

### Alembic usage

Alembic lives at the repository root:

```text
alembic/
  env.py
  script.py.mako
  versions/
```

Migration rules:
- do not rely on `Base.metadata.create_all()` in backend startup
- do not create or alter shared schema from ad hoc runtime code
- run `alembic upgrade head` before backend or ingest workloads depend on a new schema
- when business identity or payload identity changes, rebuild downstream vector state after
  canonical PostgreSQL data is brought up to date

## Main Flows

### Ingest flow

1. An entrypoint triggers the ingest use case from CLI, a job, or another entrypoint.
2. The application ingest use case loads and parses raw documentation inputs.
3. Domain rules normalize content, metadata, and chunk structure.
4. The use case calls repository, tokenizer, and vector-storage ports.
5. Before persisted writes, the configured embedding-model Hugging Face tokenizer
   populates chunk token counts through the tokenizer port.
6. Adapters persist canonical documents and chunks to PostgreSQL.
7. Adapters generate embeddings and synchronize vectors to Qdrant.
8. The use case records synchronization outcomes through repository ports when needed.

Design goals:
- reproducible
- idempotent where practical
- source-aware but normalized at the persistence boundary
- resilient when vector synchronization fails after canonical PostgreSQL writes

### Search and chat flow

1. A user sends a request to an HTTP endpoint.
2. The FastAPI router validates the request and maps it into an application DTO.
3. The router calls the relevant search or chat use case.
4. The use case calls vector-store, repository, inference, and LLM ports as needed.
5. Adapters query Qdrant and load canonical text and metadata from PostgreSQL.
6. The use case assembles grounded context and produces an application result DTO.
7. The HTTP adapter maps the result DTO into an API response schema.

This flow keeps transport details and infrastructure concerns outside the core while
ensuring the LLM answer is grounded in retrieved documentation.

## Architectural Rules

- `domain` does not import `application`, `infrastructure`, or `bootstrap`
- `application` imports `domain` and port abstractions, not concrete adapters
- `infrastructure/adapters` implements application ports
- `infrastructure/entrypoints` invokes use cases and maps external request/response models
- `bootstrap` owns configuration and dependency assembly
- API schemas are not application DTOs
- ORM models are not domain models
- PostgreSQL is the source of truth for canonical text and metadata
- Qdrant is used for embeddings and retrieval only
- public traffic should enter through Caddy
- runtime configuration should come from a centralized settings module

Rule of placement for new code:
- business rule -> `domain`
- dependency contract -> `application/ports`
- use-case input/output -> `application/dto`
- use-case orchestration -> `application/use_cases`
- application-level policy -> `application/policies`
- FastAPI, CLI, jobs -> `infrastructure/entrypoints`
- OpenAI, Qdrant, SQLAlchemy, `tiktoken`, filesystem integrations -> `infrastructure/adapters`
- config and wiring -> `bootstrap`

## Deployment Model

The project is designed for self-hosted deployment on Ubuntu with Docker Compose.

Typical services:
- `proxy` - Caddy
- `backend` - FastAPI application
- `postgres` - PostgreSQL
- `qdrant` - Qdrant

Recommended deployment order:

1. Start PostgreSQL and wait for health readiness.
2. Run `alembic upgrade head`.
3. Run a full ingest rebuild so PostgreSQL and vector payloads are regenerated from the
   current source material.
4. Rebuild or resynchronize vector-store state if that environment uses Qdrant.
5. Start backend workloads.
6. Allow backend requests only after dependencies and migrations are ready.

Operational note:
- PostgreSQL does not need to be wiped before the UUID identity migration because Alembic
  backfills business UUID ids in-place.
- A full reingest after migration is still recommended to produce a clean canonical state.

Deployment conventions:
- use `compose.yaml` as the main deployment definition
- persist PostgreSQL and Qdrant data with Docker volumes
- keep backend containers stateless where possible
- load runtime configuration from environment variables
- use `postgresql+asyncpg://...` for `DATABASE_URL`
- keep secrets out of the repository
- keep backend, PostgreSQL, and Qdrant on internal Docker networks where practical

## Why This Architecture

This design balances:

- clarity - each layer has a clear responsibility
- maintainability - domain, use cases, transport, persistence, and integrations stay separated
- replaceability - adapters can change without rewriting the core
- self-hosting - all core runtime services can run on a single Ubuntu server
- schema safety - relational changes remain explicit and versioned through Alembic
- testability - domain and application logic can be tested independently from frameworks

## Future Evolution

Likely directions:

- hybrid retrieval
- reranking
- richer metadata filtering
- background synchronization and retry jobs for vector indexing
- operational dashboards and backup tooling
- support for additional documentation sources
