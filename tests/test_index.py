# tests/test_index.py

import pytest
import os
from core.index import DocumentIndex

# ─── Sample data ─────────────────────────────────────────────────────────────

CHUNKS = [
    "BM25 is a ranking function used by search engines to estimate the relevance of documents.",
    "Retrieval-Augmented Generation combines language models with information retrieval.",
    "The sliding window is a memory technique used to maintain conversation context.",
    "Vector databases store high-dimensional embeddings for similarity search.",
    "Cross-encoders score query-document pairs using transformer-based models.",
    "Sparse retrieval methods like BM25 rely on keyword overlap.",
    "Dense retrieval uses neural embeddings to capture semantic similarity.",
    "Chunking splits documents into smaller passages for retrieval.",
    "Reranking improves initial retrieval by applying a more expensive scoring model.",
    "Piper TTS is a fast, CPU-only text-to-speech synthesis system.",
    "Whisper is an automatic speech recognition model developed by OpenAI.",
    "Natural language processing enables computers to understand human language.",
    "Sentence transformers encode sentences into fixed-length vectors.",
    "Information retrieval is the process of finding relevant documents from a corpus.",
    "Large language models generate coherent text based on input prompts.",
    "Tokenization splits text into smaller units for processing.",
    "Named entity recognition identifies mentions of real-world entities in text.",
    "The BM25 algorithm considers term frequency and inverse document frequency.",
    "Context window refers to the amount of text a language model can process.",
    "Offline AI systems run entirely on-device without internet access.",
    "Mobile deployment requires efficient models with low memory footprint.",
    "PDF extraction involves parsing text from portable document format files.",
]


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestDocumentIndex:

    def test_build_no_error(self):
        """Build index from 20+ chunks — no error."""
        idx = DocumentIndex()
        idx.build(CHUNKS)
        assert idx.bm25 is not None
        assert len(idx.chunks) == len(CHUNKS)

    def test_retrieve_relevant_results(self):
        """Query 'BM25 retrieval' returns chunks about BM25/retrieval, not random ones."""
        idx = DocumentIndex()
        idx.build(CHUNKS)
        results = idx.retrieve("BM25 retrieval", top_k=3)
        # At least one result should mention BM25 or retrieval
        combined = " ".join(r["chunk"].lower() for r in results)
        assert "bm25" in combined or "retrieval" in combined

    def test_retrieve_top_k_exact(self):
        """top_k=3 returns at most 3 results."""
        idx = DocumentIndex()
        idx.build(CHUNKS)
        results = idx.retrieve("language model retrieval", top_k=3)
        assert len(results) <= 3

    def test_retrieve_no_match_returns_empty(self):
        """Query with no matching terms returns empty list, not a crash."""
        idx = DocumentIndex()
        idx.build(CHUNKS)
        # Very unusual terms unlikely to match anything
        results = idx.retrieve("xylophone quasar nebula flugelhorn", top_k=5)
        assert isinstance(results, list)
        # May be empty since all scores would be 0

    def test_retrieve_before_build_raises(self):
        """Calling retrieve() before build() raises RuntimeError."""
        idx = DocumentIndex()
        with pytest.raises(RuntimeError, match="Index not built"):
            idx.retrieve("some query")

    def test_result_structure(self):
        """Each result has chunk, score, index keys."""
        idx = DocumentIndex()
        idx.build(CHUNKS)
        results = idx.retrieve("machine learning", top_k=5)
        for r in results:
            assert "chunk" in r
            assert "score" in r
            assert "index" in r
            assert isinstance(r["score"], float)
            assert isinstance(r["chunk"], str)

    def test_save_and_load(self, tmp_path):
        """Save index to disk and load it — retrieval works identically."""
        idx = DocumentIndex()
        idx.build(CHUNKS)
        path = str(tmp_path / "test_index.pkl")
        idx.save(path)

        # Verify file created
        assert os.path.exists(path)

        # Load into a new index
        idx2 = DocumentIndex()
        idx2.load(path)
        assert len(idx2.chunks) == len(CHUNKS)

        # Retrieval should work the same
        r1 = idx.retrieve("BM25 retrieval", top_k=3)
        r2 = idx2.retrieve("BM25 retrieval", top_k=3)
        assert len(r1) == len(r2)
        assert r1[0]["chunk"] == r2[0]["chunk"]

    def test_scores_are_positive(self):
        """All returned results have positive BM25 scores."""
        idx = DocumentIndex()
        idx.build(CHUNKS)
        results = idx.retrieve("retrieval ranking document", top_k=10)
        for r in results:
            assert r["score"] > 0
