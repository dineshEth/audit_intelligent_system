from __future__ import annotations

import json
from typing import List, Optional

try:  # optional dependency
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    genai = None
    GENAI_AVAILABLE = False

from ..utils.text import normalize_whitespace, simple_summary
from .prompt_templates import SUMMARY_PROMPT, TRANSACTION_LABEL_PROMPT


class GeminiLLM:
    def __init__(self, api_key: str = "", model_name: str = "gemini-1.5-flash") -> None:
        self.api_key = api_key.strip()
        self.model_name = model_name.strip() or "gemini-1.5-flash"
        self.available = bool(GENAI_AVAILABLE and self.api_key)
        self.init_error: Optional[str] = None
        self._model = None
        if self.available:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None or not self.available:
            return
        try:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        except Exception as exc:
            self.init_error = str(exc)
            self.available = False
            self._model = None

    def generate(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.1) -> str:
        if not self.available:
            return ""
        self._ensure_loaded()
        if self._model is None:
            return ""
        try:
            response = self._model.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_new_tokens,
                },
            )
            text = getattr(response, "text", "") or ""
            return normalize_whitespace(text)
        except Exception as exc:
            self.init_error = str(exc)
            return ""

    def summarize(self, query: str, contexts: List[str], analysis: dict | None = None) -> str:
        context_text = "\n\n".join(contexts[:5])
        analysis = analysis or {}
        if self.available:
            prompt = SUMMARY_PROMPT.format(query=query, context=context_text, analysis=json.dumps(analysis, indent=2))
            generated = self.generate(prompt, max_new_tokens=220, temperature=0.0)
            if generated:
                return generated
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
