"""Shared test fixtures.

Tests must be FAST, FREE and OFFLINE: no model download, no API key, no internet.
So we use a fake embedder and a fake LLM that behave like the real ones.
"""

import zlib

import numpy as np
import pytest

from rag_lab.embeddings import normalize
from rag_lab.llm import LLMResponse
from rag_lab.retrievers import tokenize


class HashEmbedder:
    """Bag-of-words vectors via hashing: texts sharing words get similar vectors."""

    dim = 256

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            v[zlib.crc32(token.encode()) % self.dim] += 1.0
        return v

    def embed_documents(self, texts):
        return normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text):
        return normalize(self._vec(text))


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete(self, system, user, max_tokens=2048):
        self.calls.append((system, user))
        return LLMResponse(text="Fake answer [1]", prompt_tokens=100, completion_tokens=5)


@pytest.fixture
def papers():
    return [
        {
            "id": "0001",
            "title": "Retrieval augmented generation reduces hallucination",
            "abstract": "We show that retrieving documents before generation reduces "
            "hallucination in large language models on question answering.",
        },
        {
            "id": "0002",
            "title": "Speech recognition for low resource languages",
            "abstract": "We train an acoustic model for Swahili and Yoruba speech with little "
            "labelled audio using self supervised pretraining.",
        },
        {
            "id": "0003",
            "title": "Machine translation with transformers",
            "abstract": "A transformer encoder decoder translates German to English and "
            "improves BLEU score over recurrent baselines.",
        },
        {
            "id": "0004",
            "title": "Sentiment analysis of product reviews",
            "abstract": "We classify customer reviews as positive or negative with a "
            "fine tuned BERT model and report accuracy and F1.",
        },
    ]


@pytest.fixture
def embedder():
    return HashEmbedder()


@pytest.fixture
def fake_llm():
    return FakeLLM()
