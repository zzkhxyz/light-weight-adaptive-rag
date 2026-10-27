from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AC_RAG_", env_file=".env", extra="ignore")

    qdrant_path: Path = Path("data/qdrant")
    embed_en: str = "BAAI/bge-small-en-v1.5"
    embed_ru: str = "cointegrated/rubert-tiny2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_threshold: float = Field(default=0.65, ge=0, le=1)
    llm_model: str = "gpt-4o-mini"
    enable_web: bool = False
    tavily_api_key: str | None = None
    service_api_key: SecretStr | None = None
    retrieval_limit: int = Field(default=20, ge=5, le=100)
    final_limit: int = Field(default=5, ge=1, le=10)
    max_fallbacks: int = Field(default=1, ge=0, le=2)
