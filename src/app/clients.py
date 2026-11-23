from pathlib import Path
from logging import getLogger, StreamHandler, Formatter, INFO

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config.parameter_config import params_config

# load the api keys
load_dotenv()

# load the application parameters
app_params = params_config.rag_app

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# create the logger
logger = getLogger(name="Rag App")
handler = StreamHandler()
logger.addHandler(handler)
logger.setLevel(INFO)
# don't hand records to the root logger too -- some deps (presidio) call
# logging.basicConfig(), which attaches a root handler, so without this every
# line is emitted twice: once by our handler, once by root's.
logger.propagate = False
formatter = Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(fmt=formatter)

# local Ollama server (default port 11434)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# litellm (used by the guardrails-ai hub validators) reads this env var
os.environ.setdefault("OLLAMA_API_BASE", OLLAMA_BASE_URL)


def get_chat_model(model: str | None = None, temperature: float = 0):
    # Default model (model=None) follows params.yaml's llm_provider (groq/ollama).
    # An explicitly passed model name is always a local Ollama model.
    if model is None and app_params.llm_provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=app_params.llm, temperature=temperature)

    # reasoning=False turns off qwen3's <think> mode so the answer comes back
    # clean and fast.
    return ChatOllama(
        model=model or app_params.llm,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        reasoning=False,
    )


# shared llm client -- used by the RAG graph's generation step, the vector
# store's contextual-compression retriever, and the /health dependency check
llm = get_chat_model()

# shared embedding client -- used by the Chroma vector store
embedder = OllamaEmbeddings(
    model=app_params.embedding_model,
    base_url=OLLAMA_BASE_URL,
)
