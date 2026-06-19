import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def load_embeddings() -> Tuple[np.ndarray, np.ndarray]:
    """Load saved image embeddings and product ids from disk."""
    base_dir = Path(__file__).parent.parent / "data"
    embeddings_path = base_dir / "embeddings.npy"
    product_ids_path = base_dir / "product_ids.npy"

    print(f"Loading embeddings from {embeddings_path}")
    print(f"Loading product ids from {product_ids_path}")

    if not embeddings_path.exists() or not product_ids_path.exists():
        raise FileNotFoundError(
            "Could not find embeddings or product_ids files in data/. "
            "Run generate_embeddings.py first."
        )

    embeddings = np.load(embeddings_path)
    product_ids = np.load(product_ids_path)

    print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")
    print(f"Loaded {product_ids.shape[0]} product ids")

    return embeddings, product_ids


def find_similar_products(query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
    """Find the top_k most similar product ids to the query embedding."""
    print("Starting search for similar products...")

    embeddings, product_ids = load_embeddings()

    query = np.array(query_embedding, dtype=np.float32)
    if query.ndim != 1:
        raise ValueError("query_embedding must be a one-dimensional vector")

    if embeddings.ndim != 2:
        raise ValueError("Embeddings file must be a 2D array")

    if query.shape[0] != embeddings.shape[1]:
        raise ValueError(
            f"Query dimension ({query.shape[0]}) does not match embedding dimension ({embeddings.shape[1]})"
        )

    # Normalize query and stored embeddings for cosine similarity.
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        raise ValueError("Query embedding has zero length")
    query = query / query_norm

    embedding_norms = np.linalg.norm(embeddings, axis=1)
    valid = embedding_norms > 0
    if not np.all(valid):
        print("Warning: some stored embeddings have zero norm and will be skipped")

    normalized_embeddings = embeddings[valid] / embedding_norms[valid, None]
    valid_product_ids = product_ids[valid]

    # Compute cosine similarity as dot product of normalized vectors.
    similarities = normalized_embeddings.dot(query)

    # Get top k indices sorted descending.
    top_k = min(top_k, similarities.shape[0])
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results: List[Tuple[int, float]] = []
    for idx in top_indices:
        product_id = int(valid_product_ids[idx])
        score = float(similarities[idx])
        results.append((product_id, score))

    print(f"Found top {len(results)} similar products")
    for rank, (product_id, score) in enumerate(results, start=1):
        print(f"{rank}. product_id={product_id}, score={score:.4f}")

    return results


if __name__ == "__main__":
    print("This module provides load_embeddings() and find_similar_products().")
