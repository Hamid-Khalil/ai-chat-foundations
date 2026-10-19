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
