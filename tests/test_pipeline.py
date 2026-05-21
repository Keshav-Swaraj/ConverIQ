# tests/test_pipeline.py

import pytest
from unittest.mock import patch, MagicMock
from core.memory import ConversationMemory
from core.index import DocumentIndex


# ─── ConversationMemory tests (Step 6) ───────────────────────────────────────

class TestConversationMemory:

    def test_add_single_exchange(self):
        """Add one Q&A pair — history has 2 messages."""
        mem = ConversationMemory(max_turns=5)
        mem.add("What is RAG?", "RAG stands for Retrieval-Augmented Generation.")
        assert len(mem.history) == 2

    def test_add_multiple_exchanges(self):
        """Add 3 exchanges — history has 6 messages."""
        mem = ConversationMemory(max_turns=5)
        for i in range(3):
            mem.add(f"Question {i}", f"Answer {i}")
        assert len(mem.history) == 6
        assert len(mem) == 3  # __len__ counts turns

    def test_sliding_window_trims_to_max_turns(self):
        """After 6 exchanges with max_turns=5, only last 5 turns are kept."""
        mem = ConversationMemory(max_turns=5)
        for i in range(6):
            mem.add(f"Question {i}", f"Answer {i}")
        # Should have at most 5 turns = 10 messages
        assert len(mem.history) <= 10
        assert len(mem) == 5
        # Oldest exchange (Question 0) should be gone
        all_content = " ".join(m["content"] for m in mem.history)
        assert "Question 0" not in all_content
        assert "Question 5" in all_content

    def test_get_returns_copy(self):
        """get() returns a copy — mutations don't affect internal history."""
        mem = ConversationMemory()
        mem.add("Q", "A")
        history = mem.get()
        history.append({"role": "user", "content": "injected"})
        assert len(mem.history) == 2  # Unchanged

    def test_clear_empties_history(self):
        """clear() resets history to empty list."""
        mem = ConversationMemory()
        mem.add("Q1", "A1")
        mem.add("Q2", "A2")
        mem.clear()
        assert mem.history == []
        assert len(mem) == 0

    def test_message_roles(self):
        """Messages have correct roles."""
        mem = ConversationMemory()
        mem.add("My question", "My answer")
        assert mem.history[0]["role"] == "user"
        assert mem.history[0]["content"] == "My question"
        assert mem.history[1]["role"] == "assistant"
        assert mem.history[1]["content"] == "My answer"


# ─── Pipeline integration tests (Step 7) — mocked LLM ────────────────────────

class TestConverseIQPipelineMocked:
    """
    Tests for ConverseIQPipeline without requiring Ollama or real PDFs.
    Uses mocking to isolate components.
    """

    @pytest.fixture
    def mock_pipeline(self, tmp_path):
        """Create pipeline with mocked Reranker and LLM.

        NOTE: The patches are kept alive for the entire duration of each test
        that uses this fixture because pytest yields from within the `with` block.
        """
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from core.pipeline import ConverseIQPipeline

        # Create a simple PDF
        pdf_path = tmp_path / "test.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        sentences = [
            "Retrieval-Augmented Generation is a technique that combines LLMs with document retrieval.",
            "BM25 is a keyword-based retrieval algorithm that requires no neural embeddings.",
            "The sliding window memory keeps the last five question-answer exchanges.",
            "Cross-encoders rerank candidates by scoring query-document pairs semantically.",
            "ConverseIQ processes PDFs and allows users to ask questions about their content.",
            "The system is designed to run fully offline without any internet connection.",
            "Chunking splits the document into overlapping passages to preserve context.",
            "Text-to-speech narration reads the document aloud to the user.",
            "Voice activity detection determines when the user has started speaking.",
            "Whisper transcribes spoken queries into text for processing.",
        ]
        y = 700
        for s in sentences:
            c.drawString(50, y, s)
            y -= 25
        c.save()

        # Patch both Reranker and generate_answer so the pipeline doesn't need
        # a real model or Ollama.  The `with` block stays open across the yield
        # so patches are active for the entire test body.
        with patch("core.pipeline.Reranker") as MockReranker, \
             patch("core.pipeline.generate_answer") as mock_gen:

            mock_reranker_instance = MagicMock()
            MockReranker.return_value = mock_reranker_instance

            def fake_rerank(query, candidates, top_k=3):
                for c_item in candidates:
                    c_item["rerank_score"] = 1.0
                return candidates[:top_k]

            mock_reranker_instance.rerank.side_effect = fake_rerank
            mock_gen.return_value = "This is a mocked grounded answer about the document."

            pipeline = ConverseIQPipeline()
            result = pipeline.load_document(str(pdf_path))

            # Override index.retrieve so queries ALWAYS return a candidate,
            # bypassing BM25 score=0 early-exit. This ensures the full pipeline
            # path (including memory.add) is exercised regardless of PDF content.
            pipeline.index.retrieve = MagicMock(return_value=[
                {"chunk": "Mocked relevant chunk from the document.", "score": 1.0, "index": 0}
            ])

            # yield INSIDE the `with` so patches stay alive during the test
            yield pipeline, result, mock_gen

    def test_load_document_returns_success(self, mock_pipeline):
        """load_document returns success status and chunk count."""
        _, result, _ = mock_pipeline
        assert result["status"] == "Document loaded successfully"
        assert result["chunks"] > 0

    def test_answer_returns_dict_with_keys(self, mock_pipeline):
        """answer() returns dict with answer, sources, retrieved keys."""
        pipeline, _, _ = mock_pipeline
        result = pipeline.answer("What is RAG?")
        assert "answer" in result
        assert "sources" in result
        assert "retrieved" in result

    def test_answer_before_upload_returns_polite_error(self):
        """answer() without load_document returns helpful message."""
        with patch("core.pipeline.Reranker"):
            from core.pipeline import ConverseIQPipeline
            p = ConverseIQPipeline()
            result = p.answer("What is RAG?")
            assert "upload" in result["answer"].lower() or "document" in result["answer"].lower()
            assert result["sources"] == []

    def test_memory_holds_last_5(self, mock_pipeline):
        """After 6 questions, memory holds only last 5."""
        pipeline, _, _ = mock_pipeline
        for i in range(6):
            pipeline.answer(f"Question number {i}?")
        assert len(pipeline.memory) == 5

    def test_load_new_document_clears_memory(self, mock_pipeline, tmp_path):
        """Loading a second document clears old memory."""
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter

        pipeline, _, mock_gen = mock_pipeline
        pipeline.answer("First question?")
        assert len(pipeline.memory) == 1

        # Create second PDF
        pdf2 = tmp_path / "doc2.pdf"
        c = canvas.Canvas(str(pdf2), pagesize=letter)
        c.drawString(50, 700, "This is a completely different document about cooking.")
        c.drawString(50, 675, "Recipes and ingredients are the main topics here.")
        c.save()

        pipeline.load_document(str(pdf2))
        assert len(pipeline.memory) == 0

    def test_get_chunk_returns_correct_text(self, mock_pipeline):
        """get_chunk(0) returns first chunk string."""
        pipeline, _, _ = mock_pipeline
        chunk = pipeline.get_chunk(0)
        assert isinstance(chunk, str)
        assert len(chunk) > 0

    def test_get_chunk_out_of_bounds_returns_empty(self, mock_pipeline):
        """get_chunk with out-of-bounds index returns empty string."""
        pipeline, _, _ = mock_pipeline
        result = pipeline.get_chunk(99999)
        assert result == ""

    def test_answer_fallback_when_retrieval_empty(self, mock_pipeline):
        """When BM25 retrieval returns 0 candidates, pipeline falls back to first 3 chunks."""
        pipeline, _, _ = mock_pipeline
        
        # Override retrieve to return empty list
        pipeline.index.retrieve = MagicMock(return_value=[])
        
        # Mock generate_answer to verify it receives the correct chunks
        with patch("core.pipeline.generate_answer") as mock_generate:
            mock_generate.return_value = "Mocked fallback response."
            
            result = pipeline.answer("hi")
            
            # Assert generate_answer was called with the first 3 chunks
            expected_chunks = pipeline.chunks[:3]
            mock_generate.assert_called_once_with(
                query="hi",
                context_chunks=expected_chunks,
                history=[],
                model=pipeline.model
            )
            
            assert result["answer"] == "Mocked fallback response."
            assert result["sources"] == expected_chunks
            assert result["retrieved"] == 0


class TestConverseIQPipelineStreaming:
    """
    Tests for ConverseIQPipeline and generate_answer streaming functionality.
    """

    @patch("core.llm.ollama.chat")
    def test_generate_answer_stream(self, mock_chat):
        """generate_answer_stream yields tokens from Ollama streaming chat."""
        from core.llm import generate_answer_stream

        # Mock Ollama streaming response
        mock_response = [
            {"message": {"content": "Hello"}},
            {"message": {"content": " "}},
            {"message": {"content": "world"}},
            {"message": {"content": "!"}},
        ]
        mock_chat.return_value = iter(mock_response)

        tokens = list(generate_answer_stream("hi", ["chunk"], []))
        assert tokens == ["Hello", " ", "world", "!"]
        mock_chat.assert_called_once()

    def test_pipeline_answer_stream_before_upload(self):
        """pipeline.answer_stream yields helpful message if document not loaded."""
        from core.pipeline import ConverseIQPipeline
        with patch("core.pipeline.Reranker"):
            p = ConverseIQPipeline()
            chunks = list(p.answer_stream("hi"))
            assert len(chunks) == 1
            assert chunks[0]["done"] is True
            assert chunks[0]["token"] == ""

    @patch("core.pipeline.generate_answer_stream")
    def test_pipeline_answer_stream_success(self, mock_gen_stream, tmp_path):
        """pipeline.answer_stream yields token chunks and updates memory upon completion."""
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        from core.pipeline import ConverseIQPipeline

        # Create a simple PDF
        pdf_path = tmp_path / "stream_test.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(50, 700, "This is a streaming test PDF document.")
        c.save()

        # Mock generator stream
        mock_gen_stream.return_value = iter(["Streamed", " ", "response"])

        with patch("core.pipeline.Reranker") as MockReranker:
            mock_reranker_instance = MagicMock()
            MockReranker.return_value = mock_reranker_instance
            mock_reranker_instance.rerank.side_effect = lambda q, c, top_k=3: c[:top_k]

            pipeline = ConverseIQPipeline()
            pipeline.load_document(str(pdf_path))

            # Force index to return a mock chunk
            pipeline.index.retrieve = MagicMock(return_value=[
                {"chunk": "Streaming test chunk.", "score": 1.0, "index": 0}
            ])

            # Collect streamed chunks
            chunks = list(pipeline.answer_stream("Test query"))
            
            # Assertions on chunks
            assert len(chunks) == 4  # 3 tokens + 1 done chunk
            assert chunks[0] == {"token": "Streamed", "sources": ["Streaming test chunk."], "done": False}
            assert chunks[1] == {"token": " ", "sources": ["Streaming test chunk."], "done": False}
            assert chunks[2] == {"token": "response", "sources": ["Streaming test chunk."], "done": False}
            assert chunks[3] == {"token": "", "sources": ["Streaming test chunk."], "done": True}

            # Assert memory updated correctly
            assert len(pipeline.memory) == 1
            assert pipeline.memory.history[0]["content"] == "Test query"
            assert pipeline.memory.history[1]["content"] == "Streamed response"

