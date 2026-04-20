from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

try:  # optional heavy dependencies
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator
    TRAINING_DEPS_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    torch = None
    Dataset = None
    LoraConfig = None
    TaskType = None
    get_peft_model = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    Trainer = None
    TrainingArguments = None
    default_data_collator = None
    TRAINING_DEPS_AVAILABLE = False

from ..schemas import ModelRunRecord
from ..utils.dates import utcnow_iso
from .metrics import exact_match, token_f1


def _load_examples(jsonl_path: str | Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _format_instruction(example: Dict[str, str]) -> str:
    instruction = example.get("instruction", "").strip()
    output = example.get("output", "").strip()
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def _format_prompt_only(example: Dict[str, str]) -> str:
    instruction = example.get("instruction", "").strip()
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def _guess_target_modules(model) -> List[str]:
    candidate_names = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "query_key_value",
        "dense",
        "fc_in",
        "fc_out",
    ]
    available = set()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidate_names:
            available.add(leaf)

    if available:
        ordered = [name for name in candidate_names if name in available]
        return ordered[:4]

    fallback = []
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf not in {"lm_head", "embed_out"} and hasattr(module, "weight"):
            fallback.append(leaf)
    deduped = []
    for item in fallback:
        if item not in deduped:
            deduped.append(item)
    return deduped[:4] or ["c_attn"]


def _tokenize_dataset(dataset, tokenizer, max_length: int):
    def tokenize_row(example: Dict[str, str]) -> Dict[str, List[int]]:
        text = _format_instruction(example)
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    return dataset.map(tokenize_row, remove_columns=dataset.column_names)


def _evaluate_accuracy(model, tokenizer, eval_rows: List[Dict[str, str]], device: str) -> float:
    if not eval_rows or not TRAINING_DEPS_AVAILABLE:
        return 0.0

    samples = eval_rows[: min(8, len(eval_rows))]
    scores: List[float] = []
    model.eval()
    for row in samples:
        prompt = _format_prompt_only(row)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prediction = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()
        scores.append(max(exact_match(prediction, row["output"]), token_f1(prediction, row["output"])))
    return round(sum(scores) / len(scores), 4)


class LocalFineTuneTrainer:
    def __init__(self, repositories, settings) -> None:
        self.repositories = repositories
        self.settings = settings

    def run(self, dataset_path: str | Path) -> ModelRunRecord:
        start = time.time()
        dataset_path = Path(dataset_path)
        run_record = ModelRunRecord(
            base_model_path=self.settings.base_finetune_model_path,
            adapter_path="",
            status="running",
            dataset_size=0,
            notes=f"Dataset path: {dataset_path}",
        )
        self.repositories.model_runs.insert(run_record)

        if not TRAINING_DEPS_AVAILABLE:
            completed = self.repositories.model_runs.update(
                run_record.id,
                {
                    "status": "skipped",
                    "notes": "Training dependencies (datasets/peft/transformers/torch) are not installed.",
                    "duration_seconds": round(time.time() - start, 2),
                },
            )
            return completed

        if not self.settings.base_finetune_model_path or not Path(self.settings.base_finetune_model_path).exists():
            completed = self.repositories.model_runs.update(
                run_record.id,
                {
                    "status": "skipped",
                    "notes": "BASE_FINETUNE_MODEL_PATH was not configured or does not exist locally.",
                    "duration_seconds": round(time.time() - start, 2),
                },
            )
            return completed

        try:
            rows = _load_examples(dataset_path)
            if len(rows) < 4:
                completed = self.repositories.model_runs.update(
                    run_record.id,
                    {
                        "status": "skipped",
                        "dataset_size": len(rows),
                        "notes": "Not enough training examples to fine-tune.",
                        "duration_seconds": round(time.time() - start, 2),
                    },
                )
                return completed

            dataset = Dataset.from_list(rows)
            split = dataset.train_test_split(test_size=min(0.2, max(1 / len(rows), 0.1)), seed=42)
            train_rows = split["train"]
            eval_rows = split["test"]

            tokenizer = AutoTokenizer.from_pretrained(self.settings.base_finetune_model_path, local_files_only=True)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(
                self.settings.base_finetune_model_path,
                local_files_only=True,
            )
            model.train()

            target_modules = _guess_target_modules(model)
            peft_config = LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
            )
            model = get_peft_model(model, peft_config)

            train_ds = _tokenize_dataset(train_rows, tokenizer, self.settings.finetune_max_length)
            eval_ds = _tokenize_dataset(eval_rows, tokenizer, self.settings.finetune_max_length)

            output_dir = self.settings.models_dir / f"adapter_{utcnow_iso()}"
            output_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=self.settings.finetune_epochs,
                per_device_train_batch_size=self.settings.finetune_batch_size,
                per_device_eval_batch_size=self.settings.finetune_batch_size,
                learning_rate=self.settings.finetune_learning_rate,
                logging_steps=5,
                eval_strategy="epoch",
                save_strategy="epoch",
                report_to=[],
                remove_unused_columns=False,
                fp16=torch.cuda.is_available(),
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_ds,
                eval_dataset=eval_ds,
                data_collator=default_data_collator,
            )

            train_output = trainer.train()
            eval_metrics = trainer.evaluate()

            model.save_pretrained(str(output_dir))
            tokenizer.save_pretrained(str(output_dir))

            accuracy = _evaluate_accuracy(
                model,
                tokenizer,
                list(eval_rows),
                device=model.device.type if hasattr(model, "device") else ("cuda" if torch.cuda.is_available() else "cpu"),
            )

            completed = self.repositories.model_runs.update(
                run_record.id,
                {
                    "status": "completed",
                    "adapter_path": str(output_dir),
                    "dataset_size": len(rows),
                    "train_loss": round(float(getattr(train_output, "training_loss", 0.0)), 4),
                    "eval_loss": round(float(eval_metrics.get("eval_loss", 0.0)), 4) if "eval_loss" in eval_metrics else None,
                    "accuracy": accuracy,
                    "duration_seconds": round(time.time() - start, 2),
                    "notes": f"LoRA target modules: {target_modules}",
                },
            )
            return completed

        except Exception as exc:
            failed = self.repositories.model_runs.update(
                run_record.id,
                {
                    "status": "failed",
                    "duration_seconds": round(time.time() - start, 2),
                    "notes": str(exc),
                },
            )
            return failed
