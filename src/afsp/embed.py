from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

# Queries are Persian/Arabic source segments; passages are raw English texts.
# The instruct model requires this prefix only on the query side.
_QUERY_INSTRUCTION = (
    "Instruct: Retrieve English translations of Persian or Arabic Bahá'í scriptural "
    "texts that are semantically similar and share the formal register of "
    "Shoghi Effendi's translations.\nQuery: "
)


def load_model(model_name: str = "intfloat/multilingual-e5-large-instruct") -> SentenceTransformer:
    return SentenceTransformer(model_name)


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
