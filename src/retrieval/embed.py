from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_QUERY_INSTRUCTION = (
    "Instruct: Given a Persian or Arabic Bahá'í scriptural passage, retrieve "
    "passages from the same corpus that are semantically and stylistically similar.\n"
    "Query: "
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
