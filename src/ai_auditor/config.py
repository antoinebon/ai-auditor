"""Runtime configuration loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for the ai-auditor CLI. Values come from env vars or .env."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ollama_host: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:7b-instruct")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    controls_path: Path = Field(default=Path("data/controls/iso27001_annex_a.yaml"))

    # MLflow tracing / experiment tracking. Empty tracking URI falls
    # through to MLflow's default (``./mlruns`` local file store).
    mlflow_tracking_uri: str = Field(default="")
    mlflow_experiment: str = Field(default="ai-auditor")


def load_settings() -> Settings:
    return Settings()
