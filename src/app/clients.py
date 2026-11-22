"""RAG app ka LLM (Groq ya Ollama, params.yaml ke llm_provider se) + embedding (Ollama).

- Groq ke liye .env me GROQ_API_KEY chahiye.
- Ollama ke liye server chalna chahiye (default http://localhost:11434).
"""
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

from config.parameter_config import params_config

# GROQ_API_KEY .env se (import ke waqt hi, taake neeche ka client bana sake)
load_dotenv()

app_params = params_config.rag_app


def _build_llm():
    if app_params.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=app_params.llm,
            temperature=0,
            reasoning_effort="low",   # gpt-oss ke liye: tez jawab
            max_retries=3,            # free tier ke 429 par dobara koshish
            timeout=60,               # 60s me jawab na aaye to error (atakne se bachao)
        )

    from langchain_ollama import ChatOllama

    # qwen3: reasoning=False -> <think> band; gpt-oss: sirf level (\"low\")
    return ChatOllama(
        model=app_params.llm,
        temperature=0,
        reasoning="low" if "gpt-oss" in app_params.llm else False,
    )


llm = _build_llm()

embedder = OllamaEmbeddings(model=app_params.embedding_model)
