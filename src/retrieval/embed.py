from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# It tells the embedding model what kind of retrieval is intended: find English
# translations that are semantically similar and share Shoghi Effendi’s formal register.
_QUERY_INSTRUCTION = (
    "Instruct: Retrieve English translations of Persian or Arabic Bahá'í scriptural "
    "texts that are semantically similar and share the formal register of "
    "Shoghi Effendi's translations.\nQuery: "
)


def load_model(
    model_name: str = "intfloat/multilingual-e5-large-instruct",
    device: str | None = None,
) -> SentenceTransformer:
    # Default to CUDA when available; pass device="cpu" to force CPU.
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on device: {device}")
    return SentenceTransformer(model_name, device=device)


def embed_passages(
    model: SentenceTransformer, texts: list[str], batch_size: int = 32
) -> np.ndarray:
    return model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )


def embed_queries(model: SentenceTransformer, texts: list[str], batch_size: int = 32) -> np.ndarray:
    prefixed = [_QUERY_INSTRUCTION + t for t in texts]
    return model.encode(
        prefixed, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
