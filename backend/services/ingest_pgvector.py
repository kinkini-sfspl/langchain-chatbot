
from __future__ import annotations

import sys
import glob
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class Settings(BaseSettings):
    """Environment + run-time config."""
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    PG_DSN: str                          # e.g., postgresql+psycopg://postgres:pass@localhost:5432/ai_docs
    COLLECTION: str = "business_docs"    # pgvector collection name


def _load_docs(path: str) -> List[Document]:
    """
    Load a file by extension:
      - .txt -> TextLoader (single document)
      - .pdf -> PyPDFLoader (1 document per page, with page metadata)
    """
    ext = Path(path).suffix.lower()
    if ext == ".txt":
        return TextLoader(path, encoding="utf-8").load()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    raise ValueError(f"Unsupported file type: {ext} for {path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python backend/services/ingest_pgvector.py <file-or-glob> [more files/globs...]")
        print("Examples:")
        print("  python backend/services/ingest_pgvector.py data\\data.txt")
        print("  python backend/services/ingest_pgvector.py data\\manual.pdf")
        print('  python backend/services/ingest_pgvector.py "data\\*.pdf"')
        return

    # Expand any glob patterns and collect actual paths
    inputs: list[str] = []
    for arg in sys.argv[1:]:
        matches = glob.glob(arg)
        inputs.extend(matches if matches else [arg])

    all_docs: List[Document] = []
    for p in inputs:
        if not Path(p).exists():
            print(f"[skip] not found: {p}")
            continue
        try:
            docs = _load_docs(p)
            # Normalize/augment metadata
            for d in docs:
                d.metadata = d.metadata or {}
                d.metadata.setdefault("source", Path(p).name)
            all_docs.extend(docs)
        except Exception as e:
            print(f"[skip] failed to load {p}: {e}")

    if not all_docs:
        print("No documents loaded. Provide at least one .txt or .pdf file.")
        return

    # Chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(all_docs)
    if not chunks:
        print("No chunks produced after splitting. Nothing to index.")
        return

    # Embeddings + Vector store
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st = Settings()

    store = PGVector(
        collection_name=st.COLLECTION,
        connection_string=st.PG_DSN,
        embedding_function=emb,
    )

    store.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks into collection '{st.COLLECTION}'.")

    # Optional summary per source file
    by_src: dict[str, int] = {}
    for c in chunks:
        src = (c.metadata or {}).get("source", "unknown")
        by_src[src] = by_src.get(src, 0) + 1

    if by_src:
        print("Breakdown by file:")
        for src, n in sorted(by_src.items()):
            print(f"  - {src}: {n} chunks")


if __name__ == "__main__":
    main()
