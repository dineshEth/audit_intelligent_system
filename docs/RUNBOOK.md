# Runbook

## Standard flow
1. Upload a file through Streamlit or run `scripts/run_pipeline.py`.
2. The pipeline ingests the document, chunks it, stores it, and invokes the LangGraph workflow.
3. If the file is a bank statement, labeling outputs are written to:
   - `datasets/labeled_data/`
   - `outputs/labeled_docs/`
4. Q&A pairs are generated automatically and saved to MongoDB.
5. The fine-tuning watcher checks for new data and may create a new LoRA adapter in `outputs/models/`.

## Manual fine-tuning
Use the Fine-Tuning tab in Streamlit or:
```bash
python scripts/watch_and_finetune.py --interval 20
```

## Troubleshooting
- **MongoDB unavailable**: confirm `MONGODB_URI` and local server status.
- **No fine-tune created**: verify `BASE_FINETUNE_MODEL_PATH` exists locally.
- **PDF parsing weak**: prefer CSV statements for the labeling flow, because text-only PDF extraction loses table structure.
- **Empty retrieval results**: ingest the file first and confirm chunks were stored in MongoDB.
