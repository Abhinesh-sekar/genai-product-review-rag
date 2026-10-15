"""
Dense Vector Index & Semantic Embedding Engine.
Implements dense embedding projection and cosine similarity index
with fast top-k retrieval over customer review chunks.
"""

import os
import pickle
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class DenseVectorStore:
    """Dense semantic vector index for fast semantic similarity search."""

    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            stop_words="english",
        )
        self.svd = TruncatedSVD(n_components=min(embedding_dim, 100), random_state=42)
        self.is_fitted = False
        self.chunks: List[Dict] = []
        self.chunk_embeddings: Optional[np.ndarray] = None

    def fit_index(self, chunks: List[Dict]) -> "DenseVectorStore":
        """
        Fits embedding space and builds dense index over document chunks.
        """
        print(f"[VectorStore] Indexing {len(chunks):,} text chunks into {self.embedding_dim}-dim semantic space...")
        self.chunks = chunks
        corpus = [c["text"] for c in chunks]

        # 1. Sparse TF-IDF representations
        tfidf_mat = self.vectorizer.fit_transform(corpus)

        # 2. Dense latent semantic reduction (LSA / SVD)
        n_comp = min(self.embedding_dim, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
        self.svd.n_components = n_comp
        dense_embeds = self.svd.fit_transform(tfidf_mat)

        # 3. L2 normalize embeddings for fast cosine similarity
        norms = np.linalg.norm(dense_embeds, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.chunk_embeddings = dense_embeds / norms
        self.is_fitted = True

        print(f"[VectorStore] Built dense index of shape {self.chunk_embeddings.shape}.")
        return self

    def embed_query(self, query: str) -> np.ndarray:
        """Transforms a user query into the normalized dense semantic space."""
        q_sparse = self.vectorizer.transform([query])
        q_dense = self.svd.transform(q_sparse)
        norm = np.linalg.norm(q_dense)
        if norm > 0:
            q_dense = q_dense / norm
        return q_dense

    def search(
        self,
        query: str,
        top_k: int = 5,
        product_id_filter: Optional[str] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Performs semantic similarity search with optional product filter.
        Returns list of (chunk_dict, cosine_similarity_score).
        """
        if not self.is_fitted or self.chunk_embeddings is None:
            raise ValueError("VectorStore is not fitted. Call fit_index first.")

        q_vec = self.embed_query(query)
        sim_scores = cosine_similarity(q_vec, self.chunk_embeddings)[0]

        # Filter by product_id if provided
        candidate_indices = []
        for idx, chunk in enumerate(self.chunks):
            if product_id_filter is None or chunk["product_id"] == product_id_filter:
                candidate_indices.append(idx)

        if not candidate_indices:
            return []

        cand_scores = [(self.chunks[i], float(sim_scores[i])) for i in candidate_indices]
        cand_scores.sort(key=lambda x: -x[1])
        return cand_scores[:top_k]

    def save(self, filepath: str):
        """Persists index to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "DenseVectorStore":
        """Loads index from disk."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
