# Rust Docs RAG Assistant

## Overview

This project is a self‑hosted Retrieval Augmented Generation (RAG) backend for answering questions over offline Rust documentation.

The system ingests documentation from multiple Rust crates (Rust Book, Cargo Book, Rust Reference, Rust Std docs), normalizes them into a unified chunk model, generates embeddings, and stores them in a vector database for semantic retrieval.

All infrastructure components are self‑hosted.

---

## High Level Architecture

User Query → Embedding → Qdrant (Vector Search) → Chunk IDs → PostgreSQL (Chunk Text) → Context Assembly → LLM

### Components

- Vector Database: Qdrant
- Document Store: PostgreSQL
- Raw Documentation Storage: Filesystem
- Ingest Pipeline: CLI job inside repository
- Application Backend: Retrieval + LLM orchestration

---

## Data Flow

### Ingest Phase

1. Raw documentation is downloaded to local filesystem
2. Parser extracts structured content
3. Documents are split into semantic chunks
4. Embeddings are generated
5. Data is written:
   - embeddings → Qdrant
   - chunks + metadata → PostgreSQL

### Query Phase

1. User query embedding is computed
2. Qdrant returns top‑K similar chunk_ids
3. Chunk text is fetched from PostgreSQL
4. Context is assembled
5. Context is sent to LLM

---

## Vector Database (Qdrant)

### Deployment

Qdrant runs as a self‑hosted service via Docker.

Responsibilities:
- Store embedding vectors
- ANN search
- Metadata filtering

Stored fields:
- chunk_id
- embedding vector
- crate
- document_id
- optional lightweight metadata

---

## Document Store (PostgreSQL)

PostgreSQL stores normalized chunk data.

### Tables

#### documents
- document_id
- crate_name
- source_path
- title
- version

#### chunks
- chunk_id
- document_id
- crate_name
- source_kind
- chunk_index
- text
- token_count
- anchor
- item_type (nullable)
- signature (nullable)
- extra_metadata (jsonb)

---

## Handling Different Rust Documentation Structures

Rust Std documentation differs structurally from Book‑like documentation.

The ingest pipeline normalizes all sources into a canonical chunk schema.

Missing fields are stored as NULL.
Source‑specific data is stored inside JSONB metadata.

---

## Raw Documentation Storage

Raw documentation files are stored on filesystem.

- ignored by git
- treated as ingest artifacts
- can be regenerated

Future option:
- migrate to object storage (MinIO / S3 compatible)

---

## Ingest Pipeline

Implemented as CLI job.

Responsibilities:
- download / read raw docs
- parse
- chunk
- embed
- write to PostgreSQL
- write embeddings to Qdrant

Pipeline must be:
- reproducible
- idempotent
- configurable via env/config

---

## Scaling Considerations

System can scale by:
- increasing Qdrant resources
- sharding collections
- adding PostgreSQL read replicas

---

## Deployment Model

All components are self‑hosted:

- Qdrant Docker container
- PostgreSQL server
- Backend service
- Optional reverse proxy

---

## Future Improvements

- hybrid search
- reranking
- ingestion versioning
- object storage migration
- multi‑tenant support

