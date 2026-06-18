from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_K8S_URLS: list[str] = [
    "https://kubernetes.io/docs/concepts/workloads/pods/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/deployment/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/statefulset/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/daemonset/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/job/",
    "https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/",
    "https://kubernetes.io/docs/concepts/services-networking/service/",
    "https://kubernetes.io/docs/concepts/services-networking/ingress/",
    "https://kubernetes.io/docs/concepts/services-networking/network-policies/",
    "https://kubernetes.io/docs/concepts/configuration/configmap/",
    "https://kubernetes.io/docs/concepts/configuration/secret/",
    "https://kubernetes.io/docs/concepts/storage/persistent-volumes/",
    "https://kubernetes.io/docs/concepts/storage/storage-classes/",
    "https://kubernetes.io/docs/concepts/security/rbac-good-practices/",
    "https://kubernetes.io/docs/concepts/scheduling-eviction/assign-pod-node/",
    "https://kubernetes.io/docs/tasks/run-application/horizontal-pod-autoscale/",
    "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/",
    "https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/",
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Anthropic
    anthropic_api_key: SecretStr
    anthropic_model: str = "claude-haiku-4-5-20251001"
    anthropic_max_tokens: int = 1024

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str = "cpu"

    # FAISS / Retrieval
    faiss_index_path: Path = Path("data/faiss.index")
    faiss_metadata_path: Path = Path("data/chunks.pkl")
    retrieval_top_k: int = 5
    retrieval_confidence_threshold: float = 0.3

    # Ingestion
    chunk_size_words: int = 500
    chunk_overlap_words: int = 50
    k8s_docs_urls: list[str] = Field(default_factory=lambda: list(_DEFAULT_K8S_URLS))

    # Guardrails
    guardrail_overlap_threshold: float = 0.15

    # API
    rate_limit_per_minute: int = 10
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
