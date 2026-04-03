# Rust Docs RAG Assistant

A self-hosted Retrieval-Augmented Generation (RAG) backend for answering questions over offline Rust documentation.

The project ingests Rust docs into PostgreSQL and Qdrant, retrieves relevant chunks for a user query, and uses an LLM to generate grounded answers. It is designed to run on a self-hosted Ubuntu server with Docker Compose and Caddy.

## What this project does

- Ingests offline Rust documentation such as the Rust Book, Cargo Book, Rust Reference, and standard library docs
- Parses and normalizes different documentation formats into a shared chunk model
- Stores chunk text and metadata in PostgreSQL
- Stores embeddings in Qdrant for semantic retrieval
- Assembles context from retrieved chunks and sends it to an LLM
- Exposes a FastAPI backend behind Caddy

## Stack

- **Backend:** FastAPI
- **Vector store:** Qdrant
- **Document store:** PostgreSQL
- **LLM orchestration:** LangChain
- **Proxy:** Caddy
- **Deployment:** Docker Compose (`compose.yaml`)

## High-level architecture

The system uses two storage layers with different responsibilities:

- **PostgreSQL** is the source of truth for document chunks, metadata, and ingest state
- **Qdrant** stores embeddings and powers semantic retrieval

Query flow:

1. User sends a question to the backend
2. The backend embeds the query
3. Qdrant returns the most relevant chunk IDs
4. The backend loads chunk text from PostgreSQL
5. Context is assembled and passed to the LLM
6. The backend returns a grounded answer

Ingest flow:

1. Raw Rust documentation is downloaded or loaded from disk
2. Documents are parsed and normalized
3. Content is split into chunks
4. Embeddings are generated for each chunk
5. Chunks and metadata are written to PostgreSQL
6. Embeddings are written to Qdrant

For the full system design, see [`docs/architecture.md`](./docs/architecture.md).

## Repository layout

```text
app/           FastAPI application code
tests/         Unit and integration tests
scripts/       Operational and maintenance scripts
docker/        Dockerfiles and container assets
docs/          Project documentation
notebooks/     Exploratory work only
compose.yaml   Main deployment definition
Caddyfile      Reverse proxy configuration
```

## Running the project

The project is intended to run through Docker Compose.

Typical services:

- `proxy` — Caddy
- `backend` — FastAPI application
- `postgres` — relational storage
- `qdrant` — vector search

Basic flow:

1. Copy `.env.example` to `.env`
2. Fill in required configuration values
3. Start the stack with Docker Compose
4. Run or trigger the ingest pipeline
5. Send requests to the backend through Caddy

Exact commands and operational details can be documented in `scripts/` or expanded later in this README once the deployment workflow is finalized.

## Configuration

Runtime configuration is provided through environment variables.

Typical groups of settings:

- application runtime
- PostgreSQL connection
- Qdrant connection
- LLM and embedding provider settings
- logging
- public/proxy settings

Keep real secrets out of version control. Use `.env.example` as the reference for required variables.

## Testing

The project uses:

- `pytest`
- FastAPI `TestClient`
- `unittest.mock` for external dependencies

Tests should live under `tests/` and be organized by type and layer, for example:

- `tests/unit/`
- `tests/integration/`

## Current focus

This repository is primarily focused on:

- a clean backend architecture for a Rust documentation assistant
- reliable ingest and retrieval flows
- self-hosted deployment
- maintainable testing and operational conventions

## Planned improvements

Possible future improvements include:

- hybrid search
- reranking
- ingestion versioning
- better operational tooling
- support for more documentation sources

## Notes

- Raw documentation, caches, logs, and other runtime artifacts should not be committed
- PostgreSQL and Qdrant should use persistent volumes in deployment
- Public traffic should enter through Caddy, while backend and databases stay on internal networks where possible
