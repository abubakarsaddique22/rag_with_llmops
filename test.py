"""Ollama connectivity check: LLM + embedding model.

Run: uv run python test.py   (PYTHONPATH=".;src" set hona chahiye)
"""
from app.clients import app_params, embedder, llm

print(f"LLM model       : {app_params.llm}")
print(f"Embedding model : {app_params.embedding_model}")

vector = embedder.embed_query("What is context engineering?")
print("Embedding dimensions:", len(vector))

print("LLM reply:", llm.invoke("Reply with one word: hello").content)
