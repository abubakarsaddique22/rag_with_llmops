"""params.yaml ko typed objects me load karta hai.

Use: from config.parameter_config import params_config
"""
from pathlib import Path

import yaml
from pydantic import BaseModel

ROOT_DIR = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT_DIR / "params.yaml"


class RAGAppParams(BaseModel):
    prompt_label: str = "production"
    llm_provider: str = "ollama"   # "groq" ya "ollama"
    llm: str
    embedding_model: str
    embedding_dimensions: int | None = None
    chunk_size: int
    chunk_overlap: int
    tokenizer_encoding: str
    collection_name: str
    search_type: str = "similarity"
    k: int = 3
    contextual_compression: bool = False
    compression_llm: str | None = None


class FiltrationParams(BaseModel):
    filtration_threshold: float
    max_quality_retries: int
    filtration_critic_model: str


class EvolutionParams(BaseModel):
    num_evolutions: int
    evolutions: dict[str, float]


class ContextConstructionParams(BaseModel):
    context_critic_model: str
    max_contexts_per_document: int
    context_threshold: float
    max_context_retries: int
    context_chunk_size: int
    context_chunk_overlap: int


class GoldenDatasetParams(BaseModel):
    dataset_model: str
    max_goldens_per_context: int
    golden_dataset_filename: str
    filtration_config: FiltrationParams
    evolution_config: EvolutionParams
    context_construction_config: ContextConstructionParams


class EvaluationDatasetParams(BaseModel):
    evaluation_dataset_filename: str


class AsyncParams(BaseModel):
    throttle_value: int
    max_concurrent: int


class DisplayParams(BaseModel):
    results_dir: str
    report_dir: str


class EvaluationParams(BaseModel):
    judge_llm: str
    async_config: AsyncParams
    display_config: DisplayParams


class ParamsConfig(BaseModel):
    rag_app: RAGAppParams
    golden_dataset: GoldenDatasetParams
    evaluation_dataset: EvaluationDatasetParams
    evaluation: EvaluationParams


def load_params(path: Path = PARAMS_PATH) -> ParamsConfig:
    with open(path, "r", encoding="utf-8") as f:
        return ParamsConfig(**yaml.safe_load(f))


params_config = load_params()
