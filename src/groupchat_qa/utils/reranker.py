"""BGE Reranker (BAAI/bge-reranker-v2-m3) wrapper.

Re-ranks retrieved evidence (session / record) *right before building the QA prompt*.

Design notes:
  - The reranker is a cross-encoder classifier (XLMRobertaForSequenceClassification), not a
    generative LLM, so it is served directly via HuggingFace transformers (vLLM does not
    support rerankers).
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np

# HuggingFace model id (auto-downloaded) or a local path. Override with the
# BGE_RERANKER_PATH environment variable to point at a local snapshot.
DEFAULT_BGE_RERANKER_PATH = os.getenv("BGE_RERANKER_PATH", "BAAI/bge-reranker-v2-m3")


class BGEReranker:
    """BAAI/bge-reranker-v2-m3 wrapper.

    Example:
        reranker = BGEReranker()
        ranked_indices = reranker.rerank("question text", ["doc1", "doc2", "doc3"])
        # ranked_indices: indices sorted by descending score

    When to use: after retrieval, *right before* inserting into the {triples} or {history}
    slot of the QA prompt.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_BGE_RERANKER_PATH,
        device: str | None = None,
        use_fp16: bool = True,
        # BAAI/bge-reranker-v2-m3 supports up to 8192 tokens. The old 512 only saw the opening
        # of a dialogue chunk (1500-3000 tokens) and truncated answer-relevant later lines,
        # causing a silent bias in bge-vs-none comparisons. Raised to 2048 (token budget vs
        # VRAM trade-off, mitigated by fp16).
        max_length: int = 2048,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        # Prefer the RERANKER_DEVICE env var (to bypass CUDA_VISIBLE_DEVICES remapping).
        # e.g. RERANKER_DEVICE=cuda:4 loads onto that GPU.
        env_device = os.environ.get("RERANKER_DEVICE")
        self.device = device or env_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_path).eval()
        if self.use_fp16:
            self._model = self._model.half()
        self._model = self._model.to(self.device)

    @property
    def name(self) -> str:
        return "bge-reranker-v2-m3"

    def score(self, query: str, candidates: Sequence[str]) -> np.ndarray:
        """Return the cross-encoder score of each candidate (higher = more relevant)."""
        if not candidates:
            return np.zeros(0, dtype=np.float32)
        import torch

        pairs = [[query, c] for c in candidates]
        with torch.no_grad():
            enc = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self._model(**enc).logits.view(-1).float().cpu().numpy()
        return logits.astype(np.float32)

    def rerank(
        self, query: str, candidates: Sequence[str], top_k: int | None = None
    ) -> list[int]:
        """Candidate indices sorted by descending score.

        Args:
            query     : the reference query to compare against
            candidates: list of texts to sort
            top_k     : if None, return all; otherwise return only the top k
        """
        scores = self.score(query, candidates)
        order = np.argsort(-scores).tolist()
        if top_k is not None:
            order = order[:top_k]
        return order
