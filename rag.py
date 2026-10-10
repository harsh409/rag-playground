"""
RAG = Retrieval-Augmented Generation.

Two halves:
  1. INGEST (once per PDF): PDF -> pages -> small overlapping chunks ->
     embedding vectors -> saved in a Chroma vector DB on disk (storage/chroma).
  2. RETRIEVE (every question): embed the question, find the chunks whose
     vectors are closest, hand that text to the LLM as "evidence".

The LLM never sees the whole PDF, only the few chunks relevant to the query.
That's why it works with big documents and why answers can cite page numbers.
"""
from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from llm import LocalEmbeddings

PDF_DIR = Path(__file__).parent / "data" / "pdfs"
STORAGE_DIR = Path(__file__).parent / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
COLLECTION = "documents"

_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)


@lru_cache(maxsize=1)
def vector_store() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=LocalEmbeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def load_pdf(path: Path) -> list[Document]:
    """One Document per page, tagged with file name + page number (for citations)."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Document(page_content=text, metadata={"source": path.name, "page": i}))
    return pages


def ingest_pdf(path: str | Path) -> int:
    """Upload a PDF into the knowledge base. Re-ingesting the same file replaces it.

    Returns the number of chunks indexed.
    """
    src = Path(path).expanduser().resolve()
    if src.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF: {src}")
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    dest = PDF_DIR / src.name
    if src != dest:
        shutil.copy(src, dest)

    pages = load_pdf(dest)
    if not pages:
        raise ValueError(f"No extractable text in {src.name} (scanned PDF? needs OCR first)")
    chunks = _splitter.split_documents(pages)

    vs = vector_store()
    old = vs.get(where={"source": dest.name})["ids"]
    if old:
        vs.delete(ids=old)
    ids = [f"{dest.name}-p{c.metadata['page']}-c{i}" for i, c in enumerate(chunks)]
    vs.add_documents(chunks, ids=ids)
    return len(chunks)


def list_documents() -> dict[str, int]:
    """{file name: chunk count} for everything in the knowledge base."""
    counts: dict[str, int] = {}
    for meta in vector_store().get(include=["metadatas"])["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return counts


def search(query: str, k: int = 5) -> list[Document]:
    return vector_store().similarity_search(query, k=k)


def format_hits(docs: list[Document]) -> str:
    if not docs:
        return "No matching passages found."
    return "\n\n".join(
        f"[{d.metadata['source']} p.{d.metadata['page']}]\n{d.page_content}" for d in docs
    )


def reset() -> None:
    """Wipe the vector DB (uploaded PDFs in data/pdfs are kept)."""
    vector_store.cache_clear()
    shutil.rmtree(STORAGE_DIR, ignore_errors=True)
