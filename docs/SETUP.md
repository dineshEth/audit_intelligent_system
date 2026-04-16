# Setup guide

1. Create a virtual environment and install `requirements.txt`.
2. Copy `.env.example` to `.env`.
3. Start a local MongoDB instance.
4. Optionally place local models on disk and configure:
   - `LOCAL_LLM_PATH`
   - `LOCAL_EMBEDDING_MODEL_PATH`
   - `BASE_FINETUNE_MODEL_PATH`
5. Bootstrap indexes:
   ```bash
   python scripts/bootstrap_db.py
   ```
6. Launch the UI:
   ```bash
   streamlit run streamlit_app.py
   ```

## Offline notes

- No remote API keys are required.
- Hugging Face model loaders are called with local paths only.
- If model folders are not configured, the system still runs with deterministic fallbacks.
