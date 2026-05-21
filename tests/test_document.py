# tests/test_document.py

import pytest
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from core.document import load_pdf, clean_text, chunk_text


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_pdf(tmp_path):
    """Create a simple multi-page PDF for testing."""
    pdf_path = tmp_path / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, "Hello, world!")
    c.drawString(100, 730, "This is a sample document.")
    c.showPage()
    c.drawString(100, 750, "Page 2 content.")
    c.drawString(100, 730, "- 2 -")   # page number artifact
    c.save()
    return str(pdf_path)


@pytest.fixture
def long_pdf(tmp_path):
    """Create a 55-page PDF for testing no-truncation."""
    pdf_path = tmp_path / "long.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    for i in range(55):
        c.drawString(100, 750, f"Page {i+1} content. This is some text on page {i+1}.")
        c.drawString(100, 730, f"Additional text on page {i+1} to make it more realistic.")
        c.showPage()
    c.save()
    return str(pdf_path)


@pytest.fixture
def paragraph_pdf(tmp_path):
    """Create a PDF with multiple sentences for chunking tests."""
    pdf_path = tmp_path / "paragraph.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    sentences = [
        "Machine learning is a subset of artificial intelligence.",
        "It allows computers to learn from data without explicit programming.",
        "Supervised learning uses labeled training data to make predictions.",
        "Unsupervised learning finds hidden patterns in unlabeled data.",
        "Reinforcement learning trains agents through reward and punishment.",
        "Deep learning uses neural networks with many layers.",
        "Convolutional neural networks are excellent for image recognition.",
        "Recurrent neural networks handle sequential data like text.",
        "Transfer learning reuses pre-trained models for new tasks.",
        "Natural language processing enables computers to understand text.",
        "Sentiment analysis classifies text as positive, negative, or neutral.",
        "Named entity recognition identifies people, places, and organizations.",
        "Information retrieval finds relevant documents from a large corpus.",
        "BM25 is a ranking function used in information retrieval systems.",
        "Cross-encoders score query-document pairs for semantic relevance.",
    ]
    y = 750
    for s in sentences:
        c.drawString(50, y, s)
        y -= 20
        if y < 100:
            c.showPage()
            y = 750
    c.save()
    return str(pdf_path)


# ─── Step 1 Tests: load_pdf + clean_text ────────────────────────────────────

class TestLoadPDF:
    def test_load_simple_pdf_returns_text(self, sample_pdf):
        """Load a simple PDF — text is returned."""
        text = load_pdf(sample_pdf)
        assert isinstance(text, str)
        assert len(text) > 0

    def test_load_pdf_contains_expected_content(self, sample_pdf):
        """Text content matches what was written."""
        text = load_pdf(sample_pdf)
        assert "Hello, world!" in text
        assert "This is a sample document." in text
        assert "Page 2 content." in text

    def test_load_pdf_removes_page_numbers(self, sample_pdf):
        """Page number artifacts are removed."""
        text = load_pdf(sample_pdf)
        assert "- 2 -" not in text

    def test_load_pdf_no_extra_blank_lines(self, sample_pdf):
        """No runs of 3+ consecutive newlines."""
        text = load_pdf(sample_pdf)
        assert "\n\n\n" not in text

    def test_load_long_pdf_no_truncation(self, long_pdf):
        """50+ page PDF — all pages extracted, no truncation."""
        text = load_pdf(long_pdf)
        # All 55 pages should have content
        for i in range(1, 56):
            assert f"Page {i} content" in text, f"Page {i} missing from extracted text"

    def test_load_pdf_file_not_found(self):
        """File not found — raises clear error, does not crash silently."""
        with pytest.raises(FileNotFoundError):
            load_pdf("nonexistent_file_that_does_not_exist.pdf")

    def test_clean_text_removes_page_numbers(self):
        """clean_text removes standalone page number lines."""
        raw = "Hello\n\n12\n\nWorld\n\n- 5 -\n\nDone"
        cleaned = clean_text(raw)
        assert "12" not in cleaned
        assert "- 5 -" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned

    def test_clean_text_collapses_blank_lines(self):
        """clean_text collapses 3+ blank lines into 2."""
        raw = "A\n\n\n\nB"
        cleaned = clean_text(raw)
        assert "\n\n\n" not in cleaned

    def test_clean_text_strips_whitespace(self):
        """clean_text strips leading/trailing whitespace per line."""
        raw = "  Hello  \n  World  "
        cleaned = clean_text(raw)
        assert cleaned == "Hello\nWorld"


# ─── Step 2 Tests: chunk_text ─────────────────────────────────────────────────

class TestChunkText:
    def test_chunk_returns_list(self, paragraph_pdf):
        """chunk_text returns a list of strings."""
        text = load_pdf(paragraph_pdf)
        chunks = chunk_text(text)
        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)

    def test_chunk_count_reasonable(self, long_pdf):
        """A 55-page PDF produces a reasonable number of chunks."""
        text = load_pdf(long_pdf)
        chunks = chunk_text(text, chunk_size=450)
        # Should produce at least 1 chunk
        assert len(chunks) >= 1

    def test_no_chunk_ends_mid_sentence(self, paragraph_pdf):
        """No chunk ends mid-sentence — every chunk ends with . ? or !"""
        text = load_pdf(paragraph_pdf)
        chunks = chunk_text(text, chunk_size=100)
        for i, chunk in enumerate(chunks[:-1]):  # All but last
            # Strip trailing whitespace and check ending
            stripped = chunk.strip()
            assert stripped[-1] in ".?!", \
                f"Chunk {i} ends mid-sentence: '...{stripped[-30:]}'"

    def test_consecutive_chunks_share_sentences(self):
        """Consecutive chunks share overlap sentences at boundaries."""
        text = (
            "The first sentence is here. The second sentence follows. "
            "The third sentence adds more. The fourth sentence continues. "
            "The fifth sentence is next. The sixth sentence concludes the first chunk. "
            "The seventh sentence begins the second chunk. The eighth sentence follows."
        )
        chunks = chunk_text(text, chunk_size=20, overlap=2)
        if len(chunks) >= 2:
            # The last 2 sentences of chunk 0 should appear in chunk 1
            # Split them and check
            last_chunk_sentences = chunks[0].split(". ")[-2:]
            first_next_sentences = chunks[1]
            # At least one overlapping sentence should exist
            overlap_found = any(
                s.strip() in first_next_sentences
                for s in last_chunk_sentences
                if s.strip()
            )
            assert overlap_found, "No overlapping sentences found between consecutive chunks"

    def test_empty_text_returns_empty_list(self):
        """Empty text input returns empty list, not a crash."""
        result = chunk_text("")
        assert result == []

    def test_none_like_empty_returns_empty_list(self):
        """Whitespace-only input returns empty list."""
        result = chunk_text("   \n  \t  ")
        assert result == []

    def test_single_sentence_no_crash(self):
        """Single-sentence documents handled without crash."""
        result = chunk_text("This is a single sentence.")
        assert len(result) == 1
        assert result[0] == "This is a single sentence."
