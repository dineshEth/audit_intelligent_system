# Delivery notes

This repository implements the requested local Audit Intelligence System with:

- LangGraph-first orchestration, plus a fallback custom event loop if LangGraph is unavailable at runtime
- Planner, Executor, Reviewer, and Labeling agents with structured JSON-like APIs
- Hybrid RAG using cosine similarity by default and optional FAISS
- MongoDB persistence, plus an in-memory fallback for test/dev environments
- Bank statement parsing and hybrid labeling
- Automatic Q&A generation
- Local LoRA/PEFT fine-tuning loop
- Streamlit UI and demo scripts
- Generated sample outputs from the included bank statement

## Verified locally in this build

- `python -m compileall` completed successfully
- `pytest -q` passed with `3 passed`

## Important runtime note

The repository includes the full PEFT fine-tuning loop, but the generated demo run in this package recorded a `skipped` fine-tune because no local base model path was configured in this environment. To enable real local fine-tuning, set `BASE_FINETUNE_MODEL_PATH` in `.env` to a local causal language model directory.

## Included sample outputs

See `outputs/demo_run_summary.json` for the generated example artifacts included in this package.
