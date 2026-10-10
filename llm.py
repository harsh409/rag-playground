"""
Models used by the whole app: the chat LLM and the embedding model.

Chat LLM   -> the freeLLMAPI gateway (OpenAI-compatible), via ChatOpenAI.
Embeddings -> local all-MiniLM-L6-v2 (ONNX model bundled with chromadb).
              Runs on your CPU, no API calls. Downloaded once (~80 MB) to
              ~/.cache/chroma on first use.

Rails analogy: this file is like config/initializers/*.rb - one place that
builds the shared clients everyone else uses.
"""
from __future__ import annotations

import os

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI

BASE_URL = os.getenv("your base url")
API_KEY = os.getenv(
    "your model api key"
)
MODEL = os.getenv("your model")


def build_llm(temperature: float = 0.1, max_tokens: int = 2000) -> ChatOpenAI:
    """The chat model every agent uses (freeLLMAPI gateway)."""
    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=API_KEY,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
        max_retries=2,
    )


class LocalEmbeddings(Embeddings):
    """Adapter: chromadb's built-in ONNX MiniLM model behind LangChain's Embeddings interface.

    An embedding turns text into a list of 384 numbers; texts with similar
    meaning get similar numbers. That's what makes "semantic search" work.
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self._fn(texts)]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
