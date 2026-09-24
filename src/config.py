"""
Configuración central del proyecto EP1 - ISY0101 (Agente RAG de Auditoría Financiera).

Este módulo concentra TODA la lógica de conexión a proveedores de LLM/embeddings
y los parámetros que gobiernan el pipeline (chunking, temperatura, rutas, etc.),
de modo que:
  1) Nunca haya claves API hardcodeadas en el resto del código.
  2) Cambiar de proveedor implique tocar solo este archivo.

El enunciado del EP1 exige el uso de Google AI Studio / Gemini
(`text-embedding-004` + Gemini Flash); ese soporte se mantiene íntegro en
`get_chat_model("google")` / `get_embeddings("google")` para poder volver a
usarlo si el proyecto de Google Cloud asociado a la clave recupera acceso
(ver nota de vigencia más abajo). El proveedor de CHAT activo por defecto es
ahora **Groq** (`LLM_PROVIDER=groq`), y el de EMBEDDINGS es un modelo
**local** vía `sentence-transformers` (`EMBEDDING_PROVIDER=local`), ya que
Groq no ofrece API de embeddings. El curso también enseña el patrón GitHub
Models (langchain-openai contra el endpoint de Azure Inference), que se deja
disponible como alternativa aislada para chat y embeddings.

Nota de vigencia: los proveedores de LLM deprecan modelos con frecuencia
(ver historial de este archivo: "gemini-1.5-flash"/"text-embedding-004" y
luego "gemini-2.5-flash" quedaron retirados; "llama-3.3-70b-versatile" en
Groq tampoco existe ya para cuentas nuevas). Si un modelo empieza a fallar
con 404 "not found"/"no longer available", verifica el catálogo vigente
antes de asumir un bug de código:
  - Google: `client.models.list()` del SDK `google-genai`.
  - Groq: `GET https://api.groq.com/openai/v1/models` con tu API key.
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

# El proveedor de CHAT y el de EMBEDDINGS se configuran por separado (antes
# era una única variable): Groq no tiene API de embeddings, así que forzar
# un único "LLM_PROVIDER" para ambos habría dejado el pipeline RAG
# incompleto. Cada uno se resuelve de forma independiente en
# `get_chat_model()` / `get_embeddings()`.
LLMProvider = Literal["groq", "google", "github"]
EmbeddingProvider = Literal["local", "google", "github"]
MemoryStrategy = Literal["buffer", "summary"]


@dataclass(frozen=True)
class Settings:
    """Parámetros centrales del pipeline RAG.

    Todos los valores tienen un default razonable para desarrollo/pruebas,
    pero pueden sobrescribirse vía variables de entorno (ver `.env.example`).
    """

    # --- Proveedores activos (chat y embeddings, independientes) --------- #
    llm_provider: LLMProvider = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "groq").lower()  # type: ignore[return-value]
    )
    embedding_provider: EmbeddingProvider = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local").lower()  # type: ignore[return-value]
    )

    # --- Groq (proveedor de chat activo por defecto) ---------------------- #
    # Modelo verificado invocando /chat/completions real con la API key del
    # proyecto (no solo /models, que lista modelos a los que la cuenta no
    # necesariamente tiene acceso de inferencia). "openai/gpt-oss-120b" es
    # un modelo "razonador" open-weight servido por Groq: encaja bien con el
    # Chain-of-Thought del prompt de auditoría y tiene el mayor contexto
    # (131K tokens) del catálogo disponible para esta cuenta.
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_chat_model: str = field(
        default_factory=lambda: os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-120b")
    )

    # --- Embeddings locales (proveedor de embeddings activo por defecto) - #
    # Sin llamadas de red ni costo: corre en CPU vía sentence-transformers.
    # Evita además depender de una cuenta de Google que puede quedar
    # bloqueada (ver nota de vigencia del módulo).
    local_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )

    # --- Google AI Studio / Gemini (se mantiene disponible para uso futuro) #
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    google_chat_model: str = field(
        default_factory=lambda: os.getenv("GOOGLE_CHAT_MODEL", "gemini-3.6-flash")
    )
    google_embedding_model: str = field(
        default_factory=lambda: os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
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
    github_embedding_model: str = field(
        default_factory=lambda: os.getenv("GITHUB_EMBEDDING_MODEL", "text-embedding-3-small")
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

    # --- Memoria conversacional (IE4) ---------------------------------------- #
    # "buffer" (default): historial completo, sin costo de LLM extra, sin
    # riesgo de pérdida de precisión. "summary": resumen progresivo, más
    # compacto pero con una llamada extra al LLM por turno. Comparación
    # empírica de ambas en docs/comparacion_memoria.md.
    memory_strategy: MemoryStrategy = field(
        default_factory=lambda: os.getenv("MEMORY_STRATEGY", "buffer").lower()  # type: ignore[return-value]
    )

    def validate_chat(self) -> None:
        """Valida que existan las credenciales necesarias para el proveedor de CHAT activo."""
        if self.llm_provider == "groq" and not self.groq_api_key:
            raise EnvironmentError(
                "Falta GROQ_API_KEY en el entorno (.env). Obtén una clave en "
                "https://console.groq.com/keys y cópiala en tu .env local."
            )
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

    def validate_embeddings(self) -> None:
        """Valida que existan las credenciales necesarias para el proveedor de EMBEDDINGS activo."""
        if self.embedding_provider == "google" and not self.google_api_key:
            raise EnvironmentError(
                "Falta GOOGLE_API_KEY en el entorno (.env) para usar embeddings de Google."
            )
        if self.embedding_provider == "github" and not self.github_token:
            raise EnvironmentError(
                "Falta GITHUB_TOKEN en el entorno (.env) para usar embeddings de GitHub Models."
            )
        # "local" no requiere credenciales: corre con sentence-transformers en CPU.

    def validate(self) -> None:
        """Valida credenciales de AMBOS proveedores (chat + embeddings). Conveniencia."""
        self.validate_chat()
        self.validate_embeddings()


settings = Settings()


def get_chat_model(provider: LLMProvider | None = None):
    """Instancia y retorna el chat model del proveedor solicitado.

    Aislado aquí para que `rag_pipeline.py`, `fact_checking.py`, etc. nunca
    importen directamente `langchain_groq`, `langchain_google_genai` ni
    `langchain_openai`.
    """
    provider = provider or settings.llm_provider
    settings.validate_chat()

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.groq_chat_model,
            temperature=settings.temperature,
            api_key=settings.groq_api_key,
        )

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


def get_embeddings(provider: EmbeddingProvider | None = None):
    """Instancia y retorna el modelo de embeddings del proveedor solicitado.

    Nota: independiente de `get_chat_model()`. Groq (proveedor de chat por
    defecto) no ofrece API de embeddings, por eso el default aquí es
    "local" y se resuelve contra `settings.embedding_provider`, no contra
    `settings.llm_provider`.
    """
    provider = provider or settings.embedding_provider
    settings.validate_embeddings()

    if provider == "local":
        # Sin red ni costo: descarga el modelo una vez (~90 MB) y corre en CPU.
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.local_embedding_model)

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
            model=settings.github_embedding_model,
            api_key=settings.github_token,
            base_url=settings.github_endpoint,
        )

    raise ValueError(f"Proveedor de embeddings no soportado: {provider!r}")
