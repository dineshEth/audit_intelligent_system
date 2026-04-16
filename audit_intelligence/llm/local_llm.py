from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

try:  # optional dependency
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    torch = None
    AutoModelForCausalLM = None
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None
    pipeline = None
    TRANSFORMERS_AVAILABLE = False

from ..utils.text import normalize_whitespace, simple_summary
from .prompt_templates import SUMMARY_PROMPT, TRANSACTION_LABEL_PROMPT


class LocalLLM:
    def __init__(self, model_path: str = "") -> None:
        self.model_path = model_path
        self.available = bool(TRANSFORMERS_AVAILABLE and model_path and Path(model_path).exists())
        self.init_error: Optional[str] = None
        self._pipeline = None
        self._task = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None or not self.available:
            return

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            device = 0 if (torch is not None and torch.cuda.is_available()) else -1

            try:
                model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True)
                self._task = "text-generation"
            except Exception:
                model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, local_files_only=True)
                self._task = "text2text-generation"

            self._pipeline = pipeline(self._task, model=model, tokenizer=tokenizer, device=device)
        except Exception as exc:
            self.available = False
            self.init_error = str(exc)

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.1) -> str:
        if not self.available:
            return ""
        self._ensure_loaded()
        if self._pipeline is None:
            return ""

        try:
            outputs = self._pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                truncation=True,
                pad_token_id=getattr(self._pipeline.tokenizer, "pad_token_id", None),
            )
        except Exception as exc:
            self.init_error = str(exc)
            self.available = False
            return ""

        if not outputs:
            return ""

        first = outputs[0]
        if self._task == "text-generation":
            text = first.get("generated_text", "")
            return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()
        return first.get("generated_text", "").strip()

    def summarize(self, query: str, contexts: List[str], analysis: dict | None = None) -> str:
        context_text = "\n\n".join(contexts[:5])
        analysis = analysis or {}
        if self.available:
            prompt = SUMMARY_PROMPT.format(query=query, context=context_text, analysis=json.dumps(analysis, indent=2))
            generated = self.generate(prompt, max_new_tokens=220, temperature=0.0)
            if generated:
                return normalize_whitespace(generated)
        fallback = simple_summary(context_text, max_sentences=4)
        if analysis:
            key_numbers = []
            for key in ["transaction_count", "total_debit", "total_credit", "closing_balance"]:
                if key in analysis:
                    key_numbers.append(f"{key.replace('_', ' ')}={analysis[key]}")
            if key_numbers:
                fallback = f"{fallback} Key metrics: " + ", ".join(key_numbers) + "."
        return normalize_whitespace(fallback)

    def classify_transaction(self, description: str, debit: float, credit: float) -> Optional[str]:
        if not self.available:
            return None
        prompt = TRANSACTION_LABEL_PROMPT.format(description=description, debit=debit, credit=credit)
        label = self.generate(prompt, max_new_tokens=10, temperature=0.0).strip().upper()
        label = label.replace(".", "").replace("`", "").split()[0] if label else ""
        return label or None
