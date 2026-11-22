"""Chroma vector store: data/processed ke txt files -> chunks -> Ollama embeddings.

Index banane ke liye:   python -m app.vector_store
Dobara se banane ke liye: python -m app.vector_store --rebuild
"""
import argparse
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.clients import app_params, embedder

ROOT_DIR = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT_DIR / "data" / "processed"
PERSIST_DIR = ROOT_DIR / "saved-embeddings"


def get_vector_store() -> Chroma:
    return Chroma(
        collection_name=app_params.collection_name,
        embedding_function=embedder,
        persist_directory=PERSIST_DIR.as_posix(),
    )


def load_chunks():
    if not DOCS_DIR.is_dir() or not any(DOCS_DIR.glob("*.txt")):
        raise FileNotFoundError(
            f"{DOCS_DIR} me koi .txt file nahi. Pehle `python main.py` chalayein."
        )

    loader = DirectoryLoader(
        path=DOCS_DIR.as_posix(),
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()

    chunker = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=app_params.tokenizer_encoding,
        chunk_size=app_params.chunk_size,
        chunk_overlap=app_params.chunk_overlap,
    )
    return chunker.split_documents(docs)


def build_index(rebuild: bool = False) -> Chroma:
    vs = get_vector_store()

    if rebuild:
        vs.delete_collection()
        vs = get_vector_store()

    if vs._collection.count() > 0:
        print(f"[vector_store] '{app_params.collection_name}' me pehle se "
              f"{vs._collection.count()} chunks hain, skip. (--rebuild se dobara banayein)")
        return vs

    chunks = load_chunks()
    print(f"[vector_store] {len(chunks)} chunks embed ho rahe hain "
          f"({app_params.embedding_model})...")
    vs.add_documents(chunks)
    print(f"[vector_store] done -> {PERSIST_DIR}")
    return vs


@lru_cache(maxsize=1)
def get_retriever():
    vs = build_index(rebuild=False)  # khali ho to khud index bana deta hai

    retriever = vs.as_retriever(
        search_type=app_params.search_type,
        search_kwargs={"k": app_params.k},
    )

    if app_params.contextual_compression:
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import LLMChainExtractor
        from langchain_ollama import ChatOllama

        compressor = LLMChainExtractor.from_llm(
            ChatOllama(
                model=app_params.compression_llm or app_params.llm,
                temperature=0,
                reasoning=False,
            )
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )

    return retriever


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    build_index(rebuild=args.rebuild)
