from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str


class LLMClient:
    """
    Free-tier friendly LLM client:
    - If HF_API_TOKEN is set: use Hugging Face Inference API (free tier if available).
    - Else: try local Transformers model (CPU).
    - Else: raise and let caller fall back to rule-based output.
    """

    def __init__(self, model_id: str = "google/flan-t5-small"):
        self.model_id = model_id
        self._local_pipe = None
        self._hf_client = None

    def _get_hf_client(self):
        if self._hf_client is not None:
            return self._hf_client
        token = os.getenv("HF_API_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not token:
            self._hf_client = None
            return None

        try:
            from huggingface_hub import InferenceClient
        except Exception:
            self._hf_client = None
            return None

        self._hf_client = InferenceClient(token=token)
        return self._hf_client

    def _get_local_pipe(self):
        if self._local_pipe is not None:
            return self._local_pipe
        try:
            from transformers import pipeline
        except Exception:
            self._local_pipe = None
            return None

        # Text2text pipeline keeps things lightweight for instruction-style prompts.
        self._local_pipe = pipeline("text2text-generation", model=self.model_id)
        return self._local_pipe

    def generate(self, prompt: str, *, max_new_tokens: int = 350) -> LLMResult:
        hf = self._get_hf_client()
        if hf is not None:
            try:
                # Many models accept plain text generation; flan-t5 works well via text generation wrappers.
                txt = hf.text_generation(
                    prompt,
                    model=self.model_id,
                    max_new_tokens=max_new_tokens,
                    temperature=0.2,
                    top_p=0.9,
                )
                return LLMResult(text=txt.strip(), provider="huggingface_inference_api")
            except Exception:
                # fall through to local
                pass

        pipe = self._get_local_pipe()
        if pipe is None:
            raise RuntimeError("No LLM provider available (HF token missing and local pipeline failed).")

        out = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        # transformers pipeline returns list[dict] with generated_text
        txt = (out[0].get("generated_text") if out else "") or ""
        return LLMResult(text=txt.strip(), provider="local_transformers")

