# AI Chat Foundations — Week 8: Production Retrieval-Augmented Generation (RAG)

## 1. RAG Core Vocabulary

| Term | Engineering Definition |
| :--- | :--- |
| **Document** | The original source file (e.g. `employee-handbook.md`) uploaded by a user containing reference knowledge. |
| **Chunk** | A bounded segment of text (e.g. 400 to 800 characters) created by splitting a document to fit within context and embedding limits. |
| **Embedding** | A dense floating-point vector (e.g. 768 dimensions) mathematically representing the semantic meaning of text. |
| **Vector Store** | A dedicated database (Chroma) optimized for storing chunk vectors and performing approximate nearest-neighbor search. |
| **Score / Similarity** | Inverted cosine distance normalized to `[0.0, 1.0]`. `1.0` indicates exact semantic alignment; `0.0` indicates orthogonality or divergence. |
| **Top-K** | The maximum number of highest-scoring chunks retrieved from the vector store for a given query. |
| **Threshold (`min_score`)** | A strict similarity floor (e.g. `0.66`). Chunks scoring below this cutoff are dropped to prevent hallucinating on out-of-domain input. |
| **Context** | The sanitized, assembled block of retrieved text chunks injected into the LLM system prompt inside `<context>...</context>` tags. |
| **Citation** | An explicit inline bracketed reference (e.g. `[1]`) produced by the LLM attributing a claim to a specific source chunk. |

---

## 2. Architecture & Pipelines

### The Ingestion Pipeline (Write Path)
```text
Raw File (.txt / .md) -> Sanitize & Decode -> Normalize -> Chunk (800/100) -> Batch Embed (text-embedding-004) -> Chroma Upsert (atomic rollback)
```

### The Query & Generation Pipeline (Read Path)
```text
User Request -> Condensation (if multi-turn) -> Embed & Retrieve (k=top_k) -> Score Gate (min_score=0.66) -> Context Assembly -> LLM Grounded Gen -> Regex Citation Parse
```

> **Developer Rule: Debug Retrieval First.** If an LLM answers incorrectly or refuses, check `POST /knowledge/search` to confirm relevant chunks were indexed and passed threshold before tweaking prompts.

---

## 3. API Endpoints

| Method | Path | Tag | Purpose |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/knowledge/documents` | Knowledge Base | Ingest, chunk, embed, and index a `.txt` or `.md` file. |
| `GET` | `/api/v1/knowledge/documents` | Knowledge Base | List all unique documents, chunk counts, and SHA256 hashes. |
| `DELETE`| `/api/v1/knowledge/documents/{document_id}` | Knowledge Base | Purge a document and delete all corresponding chunks. |
| `POST` | `/api/v1/knowledge/search` | Retrieval | Raw semantic search returning normalized scores and threshold flags. |
| `POST` | `/api/v1/rag/ask` | RAG Chat | Single-turn grounded Q&A withreshold short-circuiting. |
| `POST` | `/api/v1/rag/messages` | RAG Chat | Multi-turn conversational RAG with standalone rewrite condensation. |
| `POST` | `/api/v1/assistant/messages` | Tool Chat | Agentic loop with autonomous `search_knowledge_base` tool calls. |

---

## 4. Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Directory for persistent Chroma vector storage. |
| `RAG_CHUNK_SIZE` | `800` | Target character size per text chunk. |
| `RAG_CHUNK_OVERLAP` | `100` | Character overlap between consecutive chunks. |
| `RAG_TOP_K_DEFAULT` | `4` | Default chunk count retrieved per search. |
| `RAG_TOP_K_MAX` | `8` | Maximum chunk ceiling permitted. |
| `RAG_MIN_SCORE` | `0.66` | Minimum similarity score required to include context. |
| `RAG_MAX_DOCUMENTS` | `20` | Hard cap on total documents in the store. |
| `RAG_MAX_UPLOAD_BYTES` | `200000` | Maximum file upload size (200 KB). |
| `RAG_MAX_CONTEXT_CHARS` | `6000` | Maximum character budget injected into context. |
| `RAG_MAX_QUESTION_CHARS`| `500` | Validation boundary on user question lengths. |
| `EMBEDDING_MODEL` | `text-embedding-004` | Embedding provider model. |
| `EMBEDDING_BATCH_SIZE` | `10` | Chunk batch size per embedding API call. |

---

## 5. Dependencies & Packages

* `chromadb>=0.5.0`: Embedded vector database engine.
* `langchain-chroma>=0.1.0`: LangChain vector store integration.
* `google-genai>=0.1.0`: Native client for Google Gemini LLMs and text embeddings.
* `pytest-asyncio>=0.23.0`: Asynchronous unit and integration test runner.

---

## 6. Findings & Empirical Data

### Task 0.2: Embedding Fundamentals & Model Mixing
* **Score Range:** Cosine distances mapped to `[0.0, 1.0]`. Identical = `1.0`, relevant = `0.70-0.77`, nonsense = `0.40-0.58`.
* **Model Mixing:** Vectors from `text-embedding-004` (768d) cannot be mixed with `gemini-embedding-001` or preview models. Dimension and projection mismatches corrupt index calculations.

### Task 2.2: Chunking & Threshold Tuning
* **Chunk Sizing:** 100 chars fractured policies; 400 chars risked splitting tables; 800 chars with 100 overlap preserved full policy context and hit 100% recall.
* **Threshold (0.66):** Out-of-domain queries scored up to `0.6541`. Setting `0.66` reliably drops irrelevant queries before invoking LLM inference.

### Task 3.1: Top-K Context Budget & Near-Topic Behavior
* **Near-Topic Queries:** Scored `0.6475-0.6541`. LLM returned `NO_ANSWER_TEXT` per system prompt instructions.
* **Token Costs:** `top_k=1` used ~300 tokens; `top_k=8` used >1,200 tokens. `RAG_MAX_CONTEXT_CHARS=6000` prevents prompt budget overflows.

### Task 3.2: Multi-Turn Rewrites
* **Raw vs. Rewritten:** Raw query scored `0.48` (failure). LLM condensation rewrite scored `0.7215` (Rank 1).

### Task 3.3: Guardrails
* **Injection Defense:** Replacing `</context>` with `[/context]` neutralized prompt injection attacks.

### Task 4.1: Static RAG vs Agentic Tool Loop Comparison

| Query | Pipeline | Rounds | Tokens | Correct | Handling on Missing Context |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Handbook Fact** | `/rag/ask` | 1 | 545 | Yes | N/A |
| | `/assistant/messages` | 2 | 1,329 | Yes | N/A |
| **2. Product Fact** | `/rag/ask` | 0 | 0 | False Negative | Chunks dropped by threshold; 0 tokens burned. |
| | `/assistant/messages` | 2 | 1,106 | Yes | Dynamic retrieval succeeded. |
| **3. Capital of France** | `/rag/ask` | 0 | 0 | Correct Refusal | Dropped by threshold; 0 tokens burned. |
| | `/assistant/messages` | 1 | 523 | Answered | Answered from parametric weights (bypassed tool). |
| **4. Unanswerable Near-Topic** | `/rag/ask` | 1 | 495 | Correct Refusal | Model inspected context and returned `NO_ANSWER_TEXT`. |
| | `/assistant/messages` | 2 | 1,072 | Correct Refusal | Agent ran tool search and synthesized explicit refusal. |
| **5. Tool Chaining / Math** | `/rag/ask` | 1 | 801 | Unreliable | Hallucinated calculation without execution tool. |
| | `/assistant/messages` | 2 | 1,771 | Yes | Retained facts from tool and calculated result. |

### Task 4.4: PDF Page Tracking
* **Methodology:** Page numbers are attached to chunks by extracting text page-by-page, calculating the cumulative character length of each normalized page, and matching a chunk’s `start_index` to those boundaries.
* **Failure Modes:** This approximation can be slightly inaccurate when chunks naturally overlap across a page boundary or if Markdown/newline normalization heavily shifts character indices.

### Final Empirical Evaluation (`eval_rag.py`)
* **Hit@1:** `9 / 9` (100.0%)
* **Hit@k:** `9 / 9` (100.0%)
* **Average Answerable Top Score:** `0.7102`
* **Average Unanswerable Top Score:** `0.6260`
* **Separation Margin:** `+0.0842`

---

## 7. Known Limitations
1. **Authentication:** No auth layer; endpoints are public.
2. **Tenant Isolation:** One shared knowledge base across all users.
3. **File Formats:** Supports `.txt`, `.md`, and `.pdf` (at most 10 pages). Text-only PDFs; no OCR for scanned images.
4. **Local Deployment:** Chroma uses local SQLite/Parquet files in a single process.
5. **Retrieval Capabilities:** Semantic vector search only; no BM25 hybrid search or cross-encoders.
6. **Corpus Calibration:** The `0.66` threshold was tuned on a small test corpus and needs recalibration on production data.
7. **Free-Tier Limits:** Subject to API rate limits; data sent over free-tier keys may be logged by Google.
8. **Immutability:** Changing chunk size or embedding model requires purging and re-ingesting the whole database.

## Findings

### Task 0.2: MongoDB Limits and Persistence
* **MongoDB Version:** 7.0.45 (Downgraded from 8 due to Linux kernel compatibility issue SERVER-121912).
* **Permissions:** `app_user` can successfully write to `ai_chat_foundations` but is correctly blocked from accessing the `admin` database (`not authorized on admin`).
* **Persistence:** Data written to the named volume survives container deletion and recreation.

### Task 0.3: PyMongo Async Smoke Test Findings
* **Timezone Awareness:** A naive client returns datetimes with `tzinfo=None`. A client instantiated with `tz_aware=True` returns UTC datetime with `tzinfo=FixedOffset(0, 'UTC')`.
* **Missing ID:** `find_one` returns `None` without throwing an exception.
* **Duplicate `_id` & Unique Indexes:** MongoDB raises `pymongo.errors.DuplicateKeyError`. Calling `create_index` with identical specifications is idempotent.
* **Cursor Pagination & `to_list`:** `to_list(None)` retrieves results up to the cursor limit. Calling `to_list(0)` raises `ValueError: to_list() length must be greater than 0`.
* **Atomic Counter vs Naive Counter:** 
  * Under 20 concurrent requests, `find_one_and_update` with `$inc` strictly returned sequential numbers 1 to 20 without duplicates.
  * The naive read-modify-write pattern suffered race conditions and lost updates, producing duplicate and missing values (e.g., predominantly 1s and 2s).
* **Failure Modes:**
  * **(a) DB Down:** Stopping the container causes `ServerSelectionTimeoutError` after 3.0 seconds (`serverSelectionTimeoutMS=3000`).
  * **(b) Auto-Recovery:** Starting the container allows the same client instance to reconnect and execute commands without restarting the application.
  * **(c) Bad Password:** Passing invalid credentials raises `OperationFailure: Authentication failed` in under 0.1s.

### Task 1.2: MongoDB Indexes Verification (`getIndexes()`)
* **`conversations` indexes:**
  * `_id_`: `{"_id": 1}`
  * `idx_conversation_list`: `{"archived": 1, "updated_at": -1, "_id": -1}`
* **`messages` indexes:**
  * `_id_`: `{"_id": 1}`
  * `idx_message_conv_seq`: `{"conversation_id": 1, "seq": 1}` (`unique: true`)

### Task 2.3: Limits, Failure Semantics & Concurrency
* **Stored History vs. Sent History:** Setting `HISTORY_MAX_MESSAGES=2` proves that while `GET /conversations/{id}/messages` correctly lists *all* messages persisted in the database indefinitely, the assistant physically only "sees" the exact subset retrieved by `messages.recent(...)`. The persistence layer correctly decouples durable long-term storage from the LLM's active context window limit.
* **Long Answers Cut:** In `history_for_model()`, we enforce the Week 6 character limit on a *copy* (`item["content"][:per_message_limit]`). This ensures the actual stored database reply remains fully intact for the UI to display, but the context sent to the LLM on subsequent turns is safely truncated to prevent context bloat.
* **Database Fails After Model Generation:** If `insert_many` fails, nothing is stored and `message_count` remains completely unchanged. I decided to return the standard `DB_ERROR` to the client instead of sending the orphaned answer with `"saved": false` because returning unsaved state creates unpredictable UI drift (where a refresh deletes recent context), violating the core promise of a persistent chat session.
* **Concurrency:** 5 simultaneous requests safely incremented sequence numbers `1` to `10` via MongoDB's atomic `$inc`. However, all 5 requests saw the *same* history because none had written yet. While acceptable for a stateless prototype, a robust system would implement a "busy" lock (`is_generating: true`) on the conversation document to prevent interleaved responses.
* **Indexes (`executionStats`):** `db.messages.find({conversation_id: "..."}).sort({seq: 1})` utilizes `IXSCAN` on `idx_message_conv_seq` (the index we created), preventing a full collection scan (`COLLSCAN`).


### Database-Down Matrix
| Endpoint | State when MongoDB is Stopped | Why |
| :--- | :--- | :--- |
| `GET /api/v1/health/db` | `503 DB_UNAVAILABLE` | Explicit `ping` fails immediately. |
| `GET /api/v1/conversations` (All CRUD) | `503 DB_UNAVAILABLE` | Conversation operations rely entirely on the DB. |
| `POST /api/v1/conversations/{id}/chat` | `503 DB_UNAVAILABLE` | Fails fast on `get(conversation_id)` before LLM is called. |
| `POST /api/v1/knowledge/documents` | `503 DB_UNAVAILABLE` | Fails fast on duplicate check before embeddings. |
| `POST /api/v1/chat/ask` (Week 6) | `200 OK` | Purely stateless proxy to the LLM. |
| `POST /api/v1/rag/ask` (Week 8) | `200 OK` | Retrieval runs entirely on Chroma. |
| `POST /api/v1/knowledge/search` | `200 OK` | Direct query to Chroma vector store. |

### Duplicate `seq` test
*Result:* `SUCCESS: Unique Index Blocked Duplicate! Error was -> E11000 duplicate key error collection: ai_chat_foundations.messages index: idx_message_conv_seq dup key: { conversation_id: "test_dup_seq", seq: 1 }`
## Starting MongoDB & App User Setup
Start MongoDB using Docker:
`docker compose up -d mongo`

To create the app user, connect as root via mongosh:
`mongosh -u root -p <ROOT_PASSWORD> --authenticationDatabase admin`
`use ai_chat_foundations`
`db.createUser({ user: "app_user", pwd: "<APP_PASSWORD>", roles: [{ role: "readWrite", db: "ai_chat_foundations" }] })`

## Database Architecture
| Collection | Purpose | Indexes |
| :--- | :--- | :--- |
| `conversations` | Stores conversation metadata, usage, and keyset cursor state. | `idx_conversation_list` |
| `messages` | Stores individual chat turns and tool traces. | `idx_message_conv_seq` (Unique) |
| `documents` | Document registry ensuring single source of truth for Chroma. | `idx_documents_sha256` (Unique) |

**Embedding vs. Referencing:**
Messages are separate from conversations (referencing) rather than embedded arrays to bypass MongoDB's 16MB document size limit and avoid concurrent update contention. As chats grow, appending references maintains `O(1)` inserts instead of rewriting the entire parent document.

## New Endpoints (Week 9)
| Method | Path | Tag | Purpose |
| :--- | :--- | :--- | :--- |
| `GET` | `/health/db` | Health | Measures DB latency and validates connection pool. |
| `POST` | `/conversations` | Conversations | Creates a new persisted conversation. |
| `GET` | `/conversations` | Conversations | Lists conversations with keyset pagination. |
| `GET` | `/conversations/{id}` | Conversations | Retrieves conversation metadata. |
| `PATCH`| `/conversations/{id}` | Conversations | Renames or archives a conversation. |
| `DELETE`| `/conversations/{id}` | Conversations | Deletes conversation and cascades messages. |
| `GET` | `/conversations/{id}/messages`| Conversations | Fetches the message history. |
| `POST` | `/conversations/{id}/chat` | Persistent Chat | Executes multi-mode chat turns with history. |
| `POST` | `/conversations/{id}/messages/{msg_id}/feedback` | Conversations | Stores user feedback replacing previous values. |

## Maintenance Scripts
- **`init_db.py`**: Initializes indexes (`idx_conversation_list`, `idx_message_conv_seq`, `idx_documents_sha256`). Run via `python3 scripts/init_db.py`.
- **`backfill_documents.py`**: Ensures MongoDB registry and Chroma are in sync. Run with `--check` to report drift without modifying data.
- **`check_db.py`**: Validates referential integrity, sequence gaps, and orphans without mutating data.

## Backup & Restore
Backups contain plaintext personal data and must be secured.
**Backup:**
`mongodump --uri="mongodb://app_user:<APP_PASSWORD>@127.0.0.1:27017/ai_chat_foundations?authSource=ai_chat_foundations" --out=backup/`
**Restore (Drop existing to replace cleanly):**
`mongorestore --uri="mongodb://app_user:<APP_PASSWORD>@127.0.0.1:27017/ai_chat_foundations?authSource=ai_chat_foundations" --drop backup/ai_chat_foundations/`

## Findings
- **App User vs Root (0.2):** Root accesses `admin`; app user is confined to `readWrite` on `ai_chat_foundations`. Persistence survives restarts via the mounted volume.
- **Data Types & Counters (0.3):** `utcnow` enforces MS resolution since Python microsecond drops precision in Mongo. Counters require atomic `$inc`. Reconnect is automatic without app restarts.
- **Indexes (1.2):** Verified `IXSCAN` usage over `COLLSCAN` for optimal keyset sorting and sequence uniqueness.
- **History Sent vs Stored (2.3):** `history_for_model` trims token-heavy texts strictly on the copy sent to the LLM; the database stores the unmodified original.
- **Concurrency (2.3):** Five simultaneous requests properly allocated independent sequence reservations without race conditions.
- **Drift & Deletion (3.2):** Document deletions purge Chroma first. A failed registry deletion leaves a harmless empty record (retryable), avoiding orphaned phantom vectors.
- **Duplicate Sequence (4.1):** `idx_message_conv_seq` uniquely prevents dual assignment of sequence integers per conversation ID.

## Known Limitations
- **No Authentication/Users:** Global visibility across all conversations.
- **No Cross-Store Transactions:** Gaps between Chroma and MongoDB failures could leave orphaned registry records (detectable via `backfill_documents.py --check`).
- **Data Security:** Messages stored in plaintext; no at-rest encryption or strict retention policy.
- **Snapshot Race:** Simultaneous multi-client turns on one conversation view the same history snapshot.

### Task 4.4 Findings & Scaling Note
- **Route Shadowing (4.4):** If `GET /conversations/stats` is declared after `GET /conversations/{conversation_id}`, FastAPI matches the literal string `"stats"` as the `{conversation_id}` path parameter. This throws a `404 CONVERSATION_NOT_FOUND` (or `422` validation error) because it routes to the wrong handler. Strict routing order prevents this shadowing.
- **Scaling Note:** Aggregation pipelines like `$group` and `$sum` without targeted `$match` filters are perfectly fine for a small application. However, at scale, these queries require full collection scans (`COLLSCAN`). In a production system with millions of messages, this would severely degrade performance, requiring dedicated index-covered queries, materialized views refreshed asynchronously, or pre-computed counter increments.

**Note on MongoDB Version (Task 0.2):** The specification requested MongoDB 8.x. However, MongoDB 8.0 failed to boot due to a host kernel incompatibility ('Linux kernel versions 6.19 and newer has a known incompatibility', SERVER-121912). I reverted to MongoDB 7.0 to unblock development.
