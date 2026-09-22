"""
Configuración central del proyecto EP1 - ISY0101 (Agente RAG de Auditoría Financiera).

Este módulo concentra TODA la lógica de conexión a proveedores de LLM/embeddings
y los parámetros que gobiernan el pipeline (chunking, temperatura, rutas, etc.),
de modo que:
  1) Nunca haya claves API hardcodeadas en el resto del código.
  2) Cambiar de proveedor (Google AI Studio <-> GitHub Models) implique tocar
     solo este archivo.

El enunciado del EP1 exige el uso de Google AI Studio / Gemini
(`text-embedding-004` + Gemini Flash), por lo que ese es el proveedor por
defecto (`LLM_PROVIDER=google`). El curso también enseña el patrón GitHub
Models (langchain-openai contra el endpoint de Azure Inference), que se deja
disponible como alternativa aislada para no reescribir el resto del pipeline
si el docente pide comparar proveedores.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

# Carga las variables definidas en .env (si existe) al entorno del proceso.
# No falla si el archivo no existe: en producción/CI las variables se inyectan
# directamente en el entorno.
load_dotenv()

# --------------------------------------------------------------------------- #
# Rutas del proyecto
# --------------------------------------------------------------------------- #
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
VECTORSTORE_DIR: Path = BASE_DIR / "vectorstore"
FAISS_INDEX_NAME: str = "auditoria_faiss_index"

LLMProvider = Literal["google", "github"]


@dataclass(frozen=True)
class Settings:
    """Parámetros centrales del pipeline RAG.

    Todos los valores tienen un default razonable para desarrollo/pruebas,
    pero pueden sobrescribirse vía variables de entorno (ver `.env.example`).
    """

    # --- Proveedor de LLM/embeddings ------------------------------------- #
    llm_provider: LLMProvider = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "google").lower()  # type: ignore[return-value]
    )

    # --- Google AI Studio / Gemini (proveedor exigido por el enunciado) -- #
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_chat_model: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CHAT_MODEL", "gemini-1.5-flash")
    )
    google_embedding_model: str = field(
        default_factory=lambda: os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004")
    )

    # --- GitHub Models (alternativa vista en los notebooks del curso) ---- #
    github_token: str = field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    github_chat_model: str = field(
        default_factory=lambda: os.getenv("GITHUB_CHAT_MODEL", "gpt-4o-mini")
    )
    github_endpoint: str = field(
        default_factory=lambda: os.getenv(
            "GITHUB_ENDPOINT", "https://models.inference.ai.azure.com"
        )
    )

    # --- Parámetros de generación ----------------------------------------- #
    # temperature = 0.0 exigido por el enunciado: minimiza variabilidad y
    # alucinaciones en un dominio de auditoría donde la precisión numérica
    # (montos, RUT) es crítica.
    temperature: float = field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0")))

    # --- Parámetros de chunking -------------------------------------------- #
    # chunk_size/overlap se documentan y justifican en README.md y docs/.
    # 1200/200 se eligió para minimizar el riesgo de cortar tablas financieras
    # (montos + su etiqueta) a la mitad, manteniendo suficiente overlap para
    # no perder contexto de fila/columna entre fragmentos.
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "1200")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "200")))

    # --- Recuperación / LLM-as-a-judge -------------------------------------- #
    retriever_k: int = field(default_factory=lambda: int(os.getenv("RETRIEVER_K", "8")))
    judge_min_score: float = field(
        default_factory=lambda: float(os.getenv("JUDGE_MIN_SCORE", "6.0"))
    )

    def validate(self) -> None:
        """Valida que existan las credenciales necesarias para el proveedor activo.

        Se llama explícitamente antes de instanciar clientes LLM reales (no en
        `__init__`) para que el resto del código (loaders, chunking, métricas
        deterministas, tests) pueda importarse y ejecutarse sin credenciales.
        """
        if self.llm_provider == "google" and not self.google_api_key:
            raise EnvironmentError(
                "Falta GOOGLE_API_KEY en el entorno (.env). Obtén una clave en "
                "https://aistudio.google.com/app/apikey y cópiala en tu .env "
                "local a partir de .env.example."
            )
        if self.llm_provider == "github" and not self.github_token:
            raise EnvironmentError(
                "Falta GITHUB_TOKEN en el entorno (.env) para usar el proveedor "
                "'github'."
            )


settings = Settings()


def get_chat_model(provider: LLMProvider | None = None):
    """Instancia y retorna el chat model del proveedor solicitado.

    Aislado aquí para que `rag_pipeline.py`, `fact_checking.py`, etc. nunca
    importen directamente `langchain_google_genai` ni `langchain_openai`.
    """
    provider = provider or settings.llm_provider
    settings.validate()

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.google_chat_model,
            temperature=settings.temperature,
            google_api_key=settings.google_api_key,
        )

    if provider == "github":
        # Patrón GitHub Models: endpoint compatible con OpenAI vía Azure Inference.
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.github_chat_model,
            temperature=settings.temperature,
            api_key=settings.github_token,
            base_url=settings.github_endpoint,
        )

    raise ValueError(f"Proveedor de LLM no soportado: {provider!r}")


def get_embeddings(provider: LLMProvider | None = None):
    """Instancia y retorna el modelo de embeddings del proveedor solicitado."""
    provider = provider or settings.llm_provider
    settings.validate()

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=settings.google_embedding_model,
            google_api_key=settings.google_api_key,
        )

    if provider == "github":
        # GitHub Models expone embeddings de OpenAI (p.ej. text-embedding-3-small)
        # bajo el mismo endpoint de inferencia.
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=os.getenv("GITHUB_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=settings.github_token,
            base_url=settings.github_endpoint,
        )

    raise ValueError(f"Proveedor de embeddings no soportado: {provider!r}")
