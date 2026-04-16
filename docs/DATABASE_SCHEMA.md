# Database schema

The system persists all runtime artifacts in MongoDB collections.

## `documents`
- `id`
- `file_name`
- `file_path`
- `mime_type`
- `doc_type`
- `content_text`
- `metadata`
- `checksum`
- `status`
- `version`
- `created_at`
- `updated_at`

## `chunks`
- `id`
- `document_id`
- `chunk_index`
- `text`
- `metadata`
- `version`
- `created_at`
- `updated_at`

## `labels`
- `id`
- `document_id`
- `source_file`
- `labels[]`
  - `DATE`
  - `DESCRIPTION`
  - `DEBIT`
  - `CREDIT`
  - `BALANCE`
  - `CATEGORY`
  - `LABEL_SOURCE`
  - `CONFIDENCE`
  - `RAW_ROW`
- `export_paths`
- `version`
- `created_at`
- `updated_at`

## `qa_pairs`
- `id`
- `document_id`
- `question`
- `answer`
- `source_type`
- `metadata`
- `version`
- `created_at`
- `updated_at`

## `queries`
- `id`
- `query_text`
- `document_ids`
- `response_text`
- `retrieved_chunk_ids`
- `orchestrator_trace`
- `status`
- `version`
- `created_at`
- `updated_at`

## `model_runs`
- `id`
- `base_model_path`
- `adapter_path`
- `status`
- `dataset_size`
- `train_loss`
- `eval_loss`
- `accuracy`
- `duration_seconds`
- `notes`
- `version`
- `created_at`
- `updated_at`

## `logs`
- `id`
- `agent`
- `action`
- `status`
- `message`
- `input_payload`
- `output_payload`
- `version`
- `created_at`
- `updated_at`
