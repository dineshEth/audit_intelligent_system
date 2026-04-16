# Audit Intelligence System

An offline-first, local Audit Intelligence System for analyzing financial documents and bank statements with:

- **LangGraph orchestration** (Planner → Executor → Reviewer → Labeling)
- **Hybrid RAG** (semantic + lexical retrieval using cosine similarity or FAISS)
- **MongoDB persistence** for documents, queries, labels, model runs, and logs
- **Hybrid bank-statement auto-labeling** (rule engine + optional local LLM refinement)
- **Automatic Q&A generation**
- **Local LoRA/PEFT fine-tuning loop**
- **Streamlit UI** for uploads, analysis, retrieval, and fine-tuning

The code is designed to run fully offline. All model calls use local paths only, and the system falls back to deterministic heuristics when no local model is configured.

## Project structure

```text
audit_intelligence_repo/
├── audit_intelligence/
│   ├── agents/
│   ├── analysis/
│   ├── finetune/
│   ├── ingestion/
│   ├── labeling/
│   ├── llm/
│   ├── rag/
│   ├── services/
│   ├── utils/
│   ├── config.py
│   ├── db.py
│   ├── repositories.py
│   └── schemas.py
├── datasets/
│   ├── raw_samples/
│   ├── labeled_data/
│   └── qa_data/
├── outputs/
│   ├── charts/
│   ├── labeled_docs/
│   ├── models/
│   └── reports/
├── scripts/
├── tests/
├── docs/
├── streamlit_app.py
├── requirements.txt
├── .env.example
└── docker-compose.mongo.yml
```

## Architecture

### Agent flow

1. **Planner Agent**
   - Converts a user request into structured subtasks.
   - Detects whether the uploaded file is a bank statement or a generic financial document.

2. **Executor Agent**
   - Loads and parses documents.
   - Chunks and indexes text for retrieval.
   - Runs hybrid RAG.
   - Performs pandas-based financial analysis.
   - Drafts a response and references.

3. **Reviewer Agent**
   - Checks output completeness and factual consistency.
   - Scores confidence using retrieval coverage, summary quality, and data availability.
   - Requests a retry when confidence is below threshold.

4. **Labeling Agent**
   - Parses bank statement fields:
     `DATE`, `DESCRIPTION`, `DEBIT`, `CREDIT`, `BALANCE`, `CATEGORY`
   - Applies rule-based categorization first.
   - Optionally refines labels using a local LLM.
   - Saves labeled JSON/CSV and human-readable CSV/DOCX.

5. **Auto Fine-Tuning Loop**
   - Watches `datasets/labeled_data/` and `datasets/qa_data/`.
   - Detects new or modified files via SHA-256 manifest.
   - Builds an instruction dataset.
   - Runs local LoRA/PEFT fine-tuning on a local base model.
   - Stores model metadata and metrics in MongoDB.

## Offline-first design

- `LOCAL_LLM_PATH`, `LOCAL_EMBEDDING_MODEL_PATH`, and `BASE_FINETUNE_MODEL_PATH` must point to **local** model folders.
- `local_files_only=True` is used in Transformers loaders.
- If no model path is present, the system still works with:
  - deterministic planning
  - extractive summaries
  - rule-based categories
  - templated Q&A generation
  - cosine retrieval with TF-IDF

## Supported files

- `.csv`
- `.txt`
- `.md`
- `.json`
- `.pdf` (text extraction via `pypdf`)

## Quick start

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate  # Windows
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Configure environment

```bash
cp .env.example .env
```

Edit `.env` and point the model paths to local folders if you want LLM inference or PEFT fine-tuning.

### 4) Start MongoDB

Use a local MongoDB installation, or a local Docker container:

```bash
docker compose -f docker-compose.mongo.yml up -d
```

### 5) Bootstrap indexes

```bash
python scripts/bootstrap_db.py
```

### 6) Run the UI

```bash
streamlit run streamlit_app.py
```

## Demo

The repository includes a sample bank statement:

```bash
python scripts/ingest_demo_data.py
```

This will:
- ingest `datasets/raw_samples/sample_bank_statement.csv`
- generate labels
- create Q&A pairs
- export charts and reports
- attempt a fine-tune trigger if new data is detected

## Fine-tuning notes

The LoRA/PEFT loop expects a **local causal LM** at `BASE_FINETUNE_MODEL_PATH`. A tiny or quantized local model is recommended for laptop execution. The training loop is deliberately small:

- 1 epoch by default
- small batch size
- local adapter output in `outputs/models/`
- metrics logged to MongoDB

If no local base model is present, the run is marked as `skipped` and logged cleanly.

## Collections

- `documents`
- `chunks`
- `labels`
- `qa_pairs`
- `queries`
- `model_runs`
- `logs`

See `docs/DATABASE_SCHEMA.md` for details.

## Scripts

- `python scripts/bootstrap_db.py`
- `python scripts/ingest_demo_data.py`
- `python scripts/run_pipeline.py --file datasets/raw_samples/sample_bank_statement.csv`  
- `python scripts/watch_and_finetune.py --interval 20`

## Tests

```bash
pytest -q
```

Tests default to `mongomock://localhost` so they can run without a real MongoDB server.

## Notes for assessment reviewers

- Agent APIs return structured JSON-like dictionaries.
- The orchestration layer is implemented with **LangGraph StateGraph**.
- MongoDB persistence is used throughout the pipeline.
- The bank-labeling pipeline writes labeled JSON/CSV to `datasets/labeled_data/`.
- Human-readable outputs are written to `outputs/labeled_docs/`.
- Reports are exported as DOCX in `outputs/reports/`.
- Dashboards and manual controls are available in the Streamlit app.

## Current-library compatibility references

This scaffold was aligned to the current public docs for LangGraph graph orchestration, Streamlit upload/state behavior, FAISS exact flat indexes, and Hugging Face PEFT/Transformers local-loading patterns. See the citations in the delivery note.
