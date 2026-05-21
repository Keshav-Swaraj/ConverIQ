# core/index.py

import pickle
from rank_bm25 import BM25Okapi


class DocumentIndex:
    """
    BM25 index for a single document.
    Stores chunks and their BM25-tokenized representations.
    """

    def __init__(self):
        self.chunks: list[str] = []
        self.bm25: BM25Okapi = None

    def build(self, chunks: list[str]) -> None:
        """Build BM25 index from a list of text chunks."""
        self.chunks = chunks
        tokenized = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)
        print(f"Index built: {len(chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve top-K chunks for a query using BM25.

        Returns list of dicts: [{"chunk": str, "score": float, "index": int}]
        """
        if not self.bm25:
            raise RuntimeError("Index not built. Call build() first.")

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        # Get top-K indices sorted by score descending
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        return [
            {"chunk": self.chunks[i], "score": float(scores[i]), "index": i}
            for i in top_indices
            if scores[i] > 0  # Filter zero-score results
        ]

    def save(self, path: str) -> None:
        """Save index to disk."""
        with open(path, "wb") as f:
            pickle.dump({"chunks": self.chunks, "bm25": self.bm25}, f)
        print(f"Index saved: {path}")

    def load(self, path: str) -> None:
        """Load index from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.bm25 = data["bm25"]
        print(f"Index loaded: {len(self.chunks)} chunks")
