# core/document.py

import PyPDF2
import re
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize


def load_pdf(file_path: str) -> str:
    """
    Extract and clean text from a PDF file.
    Returns full document text as a single string.
    Raises FileNotFoundError if file does not exist.
    Handles scanned (image-only) PDFs gracefully.
    """
    import os
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    full_text = []

    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

    if not full_text:
        # Scanned PDF — no extractable text
        print(f"Warning: No text extracted from {file_path}. May be a scanned/image-based PDF.")
        return ""

    raw = "\n".join(full_text)
    cleaned = clean_text(raw)
    return cleaned


def clean_text(text: str) -> str:
    """
    Remove noise from extracted PDF text:
    - Multiple blank lines → single blank line
    - Leading/trailing whitespace per line
    - Page number artifacts (lines that are just numbers)
    """
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        # Skip lines that are just page numbers (e.g. "12" or "- 12 -")
        if re.fullmatch(r"[-–\s]*\d+[-–\s]*", line):
            continue
        cleaned_lines.append(line)

    # Collapse multiple blank lines into one
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 450, overlap: int = 2) -> list[str]:
    """
    Split text into overlapping chunks that respect sentence boundaries.

    Args:
        text: Full document text
        chunk_size: Target word count per chunk (approximate)
        overlap: Number of sentences to overlap between consecutive chunks

    Returns:
        List of chunk strings
    """
    if not text or not text.strip():
        return []

    sentences = sent_tokenize(text)
    if not sentences:
        return []

    chunks = []
    current_chunk = []
    current_word_count = 0

    for i, sentence in enumerate(sentences):
        word_count = len(sentence.split())
        current_chunk.append(sentence)
        current_word_count += word_count

        if current_word_count >= chunk_size:
            # Save current chunk
            chunks.append(" ".join(current_chunk))

            # Start next chunk with overlap sentences
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_word_count = sum(len(s.split()) for s in current_chunk)

    # Add remaining sentences as final chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
