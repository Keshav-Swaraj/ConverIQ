# core/reranker.py

from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder reranker for semantic relevance scoring.
    Uses ms-marco-MiniLM-L-6-v2 (80MB, CPU-only).
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name, max_length=512)
        print("Reranker loaded.")

    def rerank(self, query: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
        """
        Rerank BM25 candidates by semantic relevance.

        Args:
            query: User's question
            candidates: List of {"chunk": str, "score": float, "index": int}
            top_k: Number of top chunks to return after reranking

        Returns:
            Top-K chunks sorted by cross-encoder score (descending)
        """
        if not candidates:
            return []

        # Build query-chunk pairs for cross-encoder
        pairs = [(query, c["chunk"]) for c in candidates]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach cross-encoder scores to candidates
        for i, candidate in enumerate(candidates):
            candidate["rerank_score"] = float(scores[i])

        # Sort by rerank score and return top-K
        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]
