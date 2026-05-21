# ConverseIQ — Agent Build Guide

> **This document is the single source of truth for building ConverseIQ.**
> Read it fully before writing any code. Follow the phases in order.
> Do not proceed to the next phase until the current phase is complete and verified.

---

## Project Overview

**ConverseIQ** is an AI-powered, voice-first document learning system. It transforms static PDF documents into interactive conversational experiences.

**Core user experience:**
1. User uploads a PDF
2. System narrates the document aloud (TTS)
3. User interrupts with a spoken question at any point
4. System transcribes the question, retrieves relevant content from the document, generates a grounded answer, and speaks it back
5. Narration resumes from exactly where it was interrupted

**What makes it unique:**
- Narration and voice listening run **concurrently on separate threads** — always listening, always narrating
- Uses **Vectorless RAG** — BM25 + cross-encoder reranking instead of a vector database
- Runs **fully offline** — no internet, no API keys, no cloud
- Designed for **mobile deployment** using Gemma 4 + Sherpa-ONNX

---

## Research Contributions

These are the novel technical contributions that differentiate ConverseIQ from existing tools:

1. **Vectorless RAG** — Replace FAISS/Pinecone with BM25 sparse retrieval + cross-encoder reranking. Eliminates vector DB memory overhead, runs on CPU, works on phones.
2. **Fully Offline + Mobile-First** — Complete pipeline on-device: Whisper STT + Gemma 4 4B LLM + Piper TTS. No server required.

---

## Tech Stack

| Layer | Technology | Notes |
|---|---|---|
| PDF Parsing | `pdfplumber`, `PyPDF2` | pdfplumber for complex layouts |
| Text Chunking | `spaCy`, `NLTK` | Sentence-boundary aware |
| Retrieval | `rank-bm25` | BM25 index, no embeddings needed |
| Reranking | `sentence-transformers` CrossEncoder | `ms-marco-MiniLM-L-6-v2`, 80MB |
| LLM | `Ollama` + `llama3` | Local inference via Ollama |
| STT | `openai-whisper` (base model) | Runs on CPU |
| VAD | `silero-vad` | Detects voice before Whisper |
| TTS | `piper-tts` | Fast, CPU-only neural TTS |
| Intent | Fine-tuned BERT classifier | Question / Command / Noise |
| Orchestration | `LangChain` | Prompt templates, context window |
| API | `FastAPI` | REST endpoints |
| UI (desktop) | `Gradio` | Rapid prototype interface |
| UI (mobile) | `Flutter` | Phase 4+ |
| Context Memory | Sliding window (Python dict) | Last 5 exchanges |

---

## Functional Requirements

| ID | Requirement |
|---|---|
| FR-01 | Accept and process uploaded PDF files |
| FR-02 | Extract and normalize text from PDF |
| FR-03 | Chunk text into sentence-boundary-aware overlapping chunks |
| FR-04 | Build BM25 index from chunks (no embeddings) |
| FR-05 | Narrate document aloud chunk by chunk using TTS |
| FR-06 | Continuously detect voice input using VAD during narration |
| FR-07 | Transcribe user voice to text using Whisper |
| FR-08 | Classify query intent (question / command / noise) |
| FR-09 | Retrieve top-K relevant chunks using BM25 |
| FR-10 | Rerank retrieved chunks using cross-encoder |
| FR-11 | Generate grounded answer using LLM + retrieved context |
| FR-12 | Speak generated answer using TTS |
| FR-13 | Resume narration from exact interruption point |
| FR-14 | Maintain sliding window conversation history |

---

## Non-Functional Requirements

| ID | Requirement | Target |
|---|---|---|
| NFR-01 | End-to-end voice-to-answer latency | Under 3 seconds |
| NFR-02 | STT word accuracy | Above 90% in normal conditions |
| NFR-03 | Retrieval precision@3 | Above 85% |
| NFR-04 | Concurrency | Narration + STT run simultaneously |
| NFR-05 | Document size | Up to 500 pages |
| NFR-06 | Hardware | Mid-range laptop, no GPU required |
| NFR-07 | Privacy | Zero external data transmission |
| NFR-08 | Memory usage | Under 8GB RAM total |

---

## Project Folder Structure

```
converseiq/
│
├── main.py                  # Entry point — starts Gradio UI
├── requirements.txt         # All Python dependencies
├── .env                     # Optional: Ollama URL, model name
│
├── core/
│   ├── __init__.py
│   ├── document.py          # PDF loading, text extraction, chunking
│   ├── index.py             # BM25 index build and retrieval
│   ├── reranker.py          # Cross-encoder reranking
│   ├── llm.py               # LLM call via Ollama + prompt template
│   ├── stt.py               # Whisper STT + Silero VAD
│   ├── tts.py               # Piper TTS narration
│   ├── memory.py            # Sliding window conversation history
│   └── pipeline.py          # Orchestrates full RAG answer pipeline
│
├── threads/
│   ├── __init__.py
│   ├── narration.py         # Narration thread — TTS playback loop
│   └── listener.py          # Listener thread — VAD + STT + trigger
│
├── api/
│   ├── __init__.py
│   └── server.py            # FastAPI endpoints (Phase 3)
│
├── ui/
│   ├── __init__.py
│   └── gradio_app.py        # Gradio interface (Phase 1)
│
└── tests/
    ├── test_document.py
    ├── test_index.py
    ├── test_pipeline.py
    └── test_stt_tts.py
```

---

## Environment Setup

Run these once before starting Phase 1:

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 2. Install Ollama and pull Llama 3
# Download from https://ollama.com
ollama pull llama3

# 3. Install Piper TTS
pip install piper-tts
# Download voice model
mkdir -p models/tts
wget -O models/tts/en_US-amy-medium.onnx \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
wget -O models/tts/en_US-amy-medium.onnx.json \
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json

# 4. Install Python dependencies
pip install -r requirements.txt
```

**requirements.txt:**
```
pdfplumber
PyPDF2
spacy
nltk
rank-bm25
sentence-transformers
openai-whisper
torch
torchaudio
silero-vad
piper-tts
ollama
langchain
langchain-community
gradio
fastapi
uvicorn
pydub
sounddevice
numpy
```

---

---

# PHASE 1 — Document Processing + Vectorless RAG Core

> **Goal:** Build and verify the core intelligence of ConverseIQ.
> By the end of this phase the system must be able to:
> - Accept a PDF
> - Extract and chunk its text
> - Build a BM25 index
> - Accept a text query
> - Return the top 3 most relevant chunks with a grounded LLM answer
>
> **No voice, no audio, no threading yet. Text in, text out.**
> This is the brain. Everything else is built on top of it.

---

## Phase 1 — Step 1: PDF Loading and Text Extraction

**File:** `core/document.py`

**What it must do:**
- Accept a file path to a PDF
- Extract all text from every page
- Clean the text (remove extra whitespace, page numbers, headers/footers)
- Return the cleaned full text as a single string

**Implementation:**

```python
# core/document.py

import pdfplumber
import re

def load_pdf(file_path: str) -> str:
    """
    Extract and clean text from a PDF file.
    Returns full document text as a single string.
    """
    full_text = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text.append(text)

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
```

**Verification checklist — do not proceed until all pass:**
- [ ] Load a simple PDF (any textbook page or article) — text is returned
- [ ] Text is clean — no random numbers, no extra blank lines
- [ ] Long PDF (50+ pages) — all pages extracted, no truncation
- [ ] Scanned PDF (image-based) — function handles gracefully (returns empty string or warning, does not crash)
- [ ] File not found — raises a clear error, does not crash silently

---

## Phase 1 — Step 2: Text Chunking

**File:** `core/document.py` (add to same file)

**What it must do:**
- Take the full document text
- Split it into chunks of approximately 400–500 words
- Chunks must respect sentence boundaries — never cut mid-sentence
- Chunks must overlap by 1–2 sentences to preserve context across boundaries
- Return a list of chunk strings

**Why sentence-boundary chunking matters:**
If you cut at a fixed character count you can split a sentence in half. The BM25 retriever needs complete, coherent sentences to match queries accurately. A chunk that starts or ends mid-sentence degrades retrieval quality.

**Implementation:**

```python
# core/document.py — add below load_pdf()

import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize

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
    sentences = sent_tokenize(text)
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
```

**Verification checklist:**
- [ ] A 10-page PDF produces between 15–40 chunks (roughly 1 chunk per 300 words)
- [ ] No chunk ends mid-sentence — every chunk ends with `.`, `?`, or `!`
- [ ] Consecutive chunks share 1–2 sentences at boundaries (overlap working)
- [ ] Single-sentence documents handled without crash
- [ ] Empty text input returns empty list, not a crash

---

## Phase 1 — Step 3: BM25 Index

**File:** `core/index.py`

**What it must do:**
- Accept a list of chunks
- Build a BM25 index from them
- Accept a query string
- Return the top-K most relevant chunks by BM25 score
- Be saveable and loadable from disk (so we don't rebuild on every run)

**Why BM25 instead of vector DB:**
BM25 needs no neural model, no GPU, no embedding generation. Index construction takes milliseconds. Memory usage is a fraction of FAISS. For document-specific QA it performs within 5% of dense retrieval when combined with cross-encoder reranking in Step 4.

**Implementation:**

```python
# core/index.py

import json
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
```

**Verification checklist:**
- [ ] Build index from 20+ chunks — no error
- [ ] Query "what is RAG" on a RAG paper returns chunks about retrieval (not random chunks)
- [ ] Query with no matching terms returns empty list, not crash
- [ ] Save index to disk — file created
- [ ] Load index from disk — retrieval works identically to in-memory version
- [ ] `top_k=3` returns exactly 3 results (or fewer if document is small)

---

## Phase 1 — Step 4: Cross-Encoder Reranking

**File:** `core/reranker.py`

**What it must do:**
- Accept a query and a list of candidate chunks (from BM25)
- Score each chunk against the query for semantic relevance
- Return the top-3 chunks reordered by semantic score
- Run on CPU in under 1 second for 10 candidates

**Why reranking is needed:**
BM25 retrieves by keyword overlap. If the user asks "how does this system handle memory?" but the relevant chunk says "context is maintained using a sliding window," BM25 may miss it because the words don't match. The cross-encoder reads both the query and chunk together and scores their semantic relationship — it catches meaning, not just keywords.

**Implementation:**

```python
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
```

**Verification checklist:**
- [ ] Model loads without error (downloads automatically on first run)
- [ ] 10 BM25 candidates in → 3 reranked candidates out
- [ ] Semantic match test: query "memory management" should score higher than "document upload" against a chunk about "sliding window context"
- [ ] Runs in under 2 seconds for 10 candidates on CPU
- [ ] Empty candidates list → returns empty list, no crash

---

## Phase 1 — Step 5: LLM Answer Generation

**File:** `core/llm.py`

**What it must do:**
- Accept a user query and 3 retrieved+reranked context chunks
- Build a structured prompt that grounds the LLM in the context
- Call Ollama (Llama 3) locally
- Return the generated answer as a string
- Maintain and use conversation history (last 5 exchanges)

**Critical prompt design:**
The prompt must instruct the LLM to ONLY answer from the provided context. This is what prevents hallucination. If the answer is not in the context, the LLM must say so explicitly.

**Implementation:**

```python
# core/llm.py

import ollama

SYSTEM_PROMPT = """You are ConverseIQ, an intelligent document assistant.
You help users understand documents by answering their questions.

STRICT RULES:
1. Answer ONLY using the context provided below.
2. If the answer is not in the context, say: "I couldn't find that in the document."
3. Be concise — 2 to 4 sentences maximum.
4. Do not add information from your training data.
5. Speak naturally, as if talking to a student."""

def build_prompt(query: str, context_chunks: list[str], history: list[dict]) -> list[dict]:
    """
    Build the message list for Ollama chat.

    Args:
        query: Current user question
        context_chunks: Top-3 reranked chunks from document
        history: List of {"role": "user"/"assistant", "content": str}

    Returns:
        List of message dicts for ollama.chat()
    """
    # Format context
    context = "\n\n---\n\n".join(
        [f"[Excerpt {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    # Build system message with context injected
    system_with_context = f"""{SYSTEM_PROMPT}

DOCUMENT CONTEXT:
{context}"""

    messages = [{"role": "system", "content": system_with_context}]

    # Add conversation history (last 5 exchanges = 10 messages)
    messages.extend(history[-10:])

    # Add current query
    messages.append({"role": "user", "content": query})

    return messages


def generate_answer(
    query: str,
    context_chunks: list[str],
    history: list[dict],
    model: str = "llama3"
) -> str:
    """
    Generate a grounded answer using Ollama.

    Args:
        query: User's question
        context_chunks: Reranked document chunks
        history: Conversation history
        model: Ollama model name

    Returns:
        Answer string
    """
    messages = build_prompt(query, context_chunks, history)

    response = ollama.chat(
        model=model,
        messages=messages,
        options={
            "temperature": 0.3,    # Low temperature = more factual
            "top_p": 0.9,
            "num_predict": 300,    # Max ~300 tokens in answer
        }
    )

    return response["message"]["content"].strip()
```

**Verification checklist:**
- [ ] Ollama is running (`ollama serve`) before testing
- [ ] Answer to a question clearly answered by context — returns correct grounded answer
- [ ] Answer to a question not in context — returns "I couldn't find that in the document" (not a hallucinated answer)
- [ ] Answer is 2–4 sentences, not a long essay
- [ ] Conversation history: second question that references first answer — LLM uses history correctly
- [ ] Model not running — clear error message, not silent failure

---

## Phase 1 — Step 6: Context Memory

**File:** `core/memory.py`

**What it must do:**
- Store question-answer pairs from the current session
- Return the last N exchanges as a formatted message list
- Clear on new document upload

**Implementation:**

```python
# core/memory.py

class ConversationMemory:
    """
    Sliding window conversation history.
    Stores last N question-answer pairs for LLM context.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: list[dict] = []  # [{"role": "user"/"assistant", "content": str}]

    def add(self, question: str, answer: str) -> None:
        """Add a Q&A exchange to memory."""
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})

        # Keep only last max_turns * 2 messages (each turn = 2 messages)
        max_messages = self.max_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get(self) -> list[dict]:
        """Return current history for LLM context."""
        return self.history.copy()

    def clear(self) -> None:
        """Clear memory — call on new document upload."""
        self.history = []

    def __len__(self) -> int:
        return len(self.history) // 2  # Number of complete turns
```

---

## Phase 1 — Step 7: Full RAG Pipeline

**File:** `core/pipeline.py`

**What it must do:**
- Tie all Phase 1 components together
- Accept a document path → process it → ready for queries
- Accept a query → retrieve → rerank → generate → return answer
- Be the single interface that the UI and later the voice threads will call

**Implementation:**

```python
# core/pipeline.py

from core.document import load_pdf, chunk_text
from core.index import DocumentIndex
from core.reranker import Reranker
from core.llm import generate_answer
from core.memory import ConversationMemory

class ConverseIQPipeline:
    """
    Main pipeline orchestrator for ConverseIQ.
    Handles document loading, indexing, retrieval, and answer generation.
    """

    def __init__(self, model: str = "llama3"):
        self.model = model
        self.index = DocumentIndex()
        self.reranker = Reranker()
        self.memory = ConversationMemory(max_turns=5)
        self.chunks: list[str] = []
        self.document_loaded = False

    def load_document(self, pdf_path: str) -> dict:
        """
        Load and process a PDF document.
        Extracts text, chunks it, and builds BM25 index.

        Returns: {"chunks": int, "status": str}
        """
        print(f"Loading document: {pdf_path}")

        # Extract text
        text = load_pdf(pdf_path)
        if not text.strip():
            return {"chunks": 0, "status": "Error: Could not extract text from PDF"}

        # Chunk text
        self.chunks = chunk_text(text, chunk_size=450, overlap=2)
        print(f"Chunked into {len(self.chunks)} chunks")

        # Build BM25 index
        self.index.build(self.chunks)

        # Clear conversation memory for new document
        self.memory.clear()

        self.document_loaded = True
        return {"chunks": len(self.chunks), "status": "Document loaded successfully"}

    def answer(self, query: str, top_k_retrieve: int = 10, top_k_rerank: int = 3) -> dict:
        """
        Answer a query using the full RAG pipeline.

        Steps:
        1. BM25 retrieve top-10 candidates
        2. Cross-encoder rerank → top-3
        3. LLM generate grounded answer
        4. Store in memory

        Returns: {"answer": str, "sources": list[str], "retrieved": int}
        """
        if not self.document_loaded:
            return {"answer": "Please upload a document first.", "sources": [], "retrieved": 0}

        # Step 1: BM25 retrieval
        candidates = self.index.retrieve(query, top_k=top_k_retrieve)

        if not candidates:
            return {
                "answer": "I couldn't find relevant content in the document for that question.",
                "sources": [],
                "retrieved": 0
            }

        # Step 2: Cross-encoder reranking
        reranked = self.reranker.rerank(query, candidates, top_k=top_k_rerank)
        context_chunks = [r["chunk"] for r in reranked]

        # Step 3: LLM answer generation
        answer = generate_answer(
            query=query,
            context_chunks=context_chunks,
            history=self.memory.get(),
            model=self.model
        )

        # Step 4: Update memory
        self.memory.add(query, answer)

        return {
            "answer": answer,
            "sources": context_chunks,
            "retrieved": len(candidates)
        }

    def get_chunk(self, index: int) -> str:
        """Get a specific chunk by index — used by narration thread."""
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return ""

    def total_chunks(self) -> int:
        return len(self.chunks)
```

**Verification checklist:**
- [ ] `load_document("test.pdf")` → returns `{"chunks": N, "status": "Document loaded successfully"}`
- [ ] `answer("What is the main topic of this document?")` → returns relevant grounded answer
- [ ] `answer("Who invented the internet in 1823?")` → returns "I couldn't find that in the document" (not hallucinated)
- [ ] Ask 6 questions in a row → memory holds only last 5
- [ ] Second question referencing first answer → LLM responds contextually
- [ ] `load_document()` on a second PDF → memory is cleared, old index is gone

---

## Phase 1 — Step 8: Gradio UI (Text-Only Demo)

**File:** `ui/gradio_app.py` and `main.py`

**What it must do:**
- Simple two-panel interface
- Left: PDF upload button
- Right: Chat interface (text input, text output)
- On upload: show "Document loaded — N chunks"
- On query: show answer + the 3 source excerpts it used

**Implementation:**

```python
# ui/gradio_app.py

import gradio as gr
from core.pipeline import ConverseIQPipeline

pipeline = ConverseIQPipeline()

def upload_document(file):
    if file is None:
        return "No file uploaded."
    result = pipeline.load_document(file.name)
    return f"✅ {result['status']} ({result['chunks']} chunks indexed)"

def ask_question(question, history):
    if not question.strip():
        return history, ""

    result = pipeline.answer(question)
    answer = result["answer"]

    # Format sources
    sources_text = "\n\n".join([
        f"📄 Source {i+1}:\n{src[:300]}..."
        for i, src in enumerate(result["sources"])
    ])

    full_response = f"{answer}\n\n---\n**Sources used:**\n{sources_text}"
    history.append((question, full_response))
    return history, ""

with gr.Blocks(title="ConverseIQ", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🎙️ ConverseIQ\n**Talk to your documents**")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Document")
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_status = gr.Textbox(label="Status", interactive=False)
            file_input.change(upload_document, inputs=file_input, outputs=upload_status)

        with gr.Column(scale=2):
            gr.Markdown("### Ask Questions")
            chatbot = gr.Chatbot(height=400)
            question_input = gr.Textbox(
                placeholder="Ask anything about the document...",
                label="Your Question"
            )
            submit_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear Chat")

            submit_btn.click(
                ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            question_input.submit(
                ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            clear_btn.click(lambda: [], outputs=chatbot)
```

```python
# main.py

from ui.gradio_app import app

if __name__ == "__main__":
    app.launch(share=False, server_port=7860)
```

**Verification checklist:**
- [ ] `python main.py` starts without errors
- [ ] Browser opens at `http://localhost:7860`
- [ ] Upload a PDF → status shows chunk count
- [ ] Type a question → answer appears in chat
- [ ] Answer includes source excerpts
- [ ] Chat history shows previous Q&A pairs
- [ ] Ask a question before uploading → polite error message, no crash

---

## Phase 1 — Complete Verification

Before moving to Phase 2, run through this full end-to-end test:

1. Start the app: `python main.py`
2. Upload a PDF (use any academic paper or textbook chapter)
3. Ask 5 questions:
   - One clearly answered by the document
   - One that references a previous answer ("explain that further")
   - One that is completely off-topic (should say "not in document")
   - One that uses different words than the document (tests reranker)
   - One that asks for a summary

**All 5 must return correct, grounded, non-hallucinated answers.**

If any fail, fix before proceeding.

---

---

# PHASE 2 — Voice Layer (STT + TTS + Concurrent Threads)

> **Goal:** Add voice input and output to the Phase 1 pipeline.
> By the end of this phase the system must:
> - Narrate the document aloud chunk by chunk
> - Listen for voice interruptions simultaneously
> - Transcribe the question, get an answer from Phase 1 pipeline, speak it back
> - Resume narration from the interruption point
>
> **This is the core ConverseIQ experience.**

---

## Phase 2 — Step 1: TTS Narration

**File:** `core/tts.py`

Implement Piper TTS to convert text to audio and play it through the system speaker.

```python
# core/tts.py

import subprocess
import tempfile
import os
import sounddevice as sd
import numpy as np
import wave

PIPER_MODEL = "models/tts/en_US-amy-medium.onnx"

def speak(text: str, block: bool = True) -> None:
    """
    Convert text to speech using Piper TTS and play it.

    Args:
        text: Text to speak
        block: If True, wait until audio finishes playing
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        # Generate audio with Piper
        subprocess.run(
            ["piper", "--model", PIPER_MODEL, "--output_file", wav_path],
            input=text.encode("utf-8"),
            check=True,
            capture_output=True
        )

        # Read and play WAV
        with wave.open(wav_path, "rb") as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        sd.play(audio, samplerate=sample_rate)
        if block:
            sd.wait()

    finally:
        os.unlink(wav_path)


def stop_speaking() -> None:
    """Stop any currently playing audio immediately."""
    sd.stop()
```

**Verification:**
- [ ] `speak("Hello, this is ConverseIQ.")` plays audio through speakers
- [ ] Audio is clear and intelligible
- [ ] `stop_speaking()` cuts audio immediately
- [ ] Long text (500 words) plays fully without cutting off

---

## Phase 2 — Step 2: STT with VAD

**File:** `core/stt.py`

Implement Silero VAD + Whisper for speech detection and transcription.

```python
# core/stt.py

import torch
import whisper
import numpy as np
import sounddevice as sd

# Load models once at module level
print("Loading Whisper...")
WHISPER_MODEL = whisper.load_model("base")

print("Loading Silero VAD...")
VAD_MODEL, VAD_UTILS = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    trust_repo=True
)
(get_speech_timestamps, _, read_audio, _, _) = VAD_UTILS

SAMPLE_RATE = 16000


def record_until_silence(
    max_duration: float = 10.0,
    silence_threshold: float = 0.5,
    vad_threshold: float = 0.5
) -> np.ndarray | None:
    """
    Record audio until the user stops speaking.
    Uses Silero VAD to detect speech start and end.

    Returns: numpy array of audio samples, or None if no speech detected
    """
    chunk_duration = 0.5  # seconds per chunk
    chunk_size = int(SAMPLE_RATE * chunk_duration)

    audio_chunks = []
    speech_started = False
    silence_duration = 0.0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32") as stream:
        total_recorded = 0.0

        while total_recorded < max_duration:
            chunk, _ = stream.read(chunk_size)
            chunk = chunk.flatten()
            audio_chunks.append(chunk)
            total_recorded += chunk_duration

            # Check VAD on this chunk
            tensor = torch.FloatTensor(chunk)
            speech_prob = VAD_MODEL(tensor, SAMPLE_RATE).item()

            if speech_prob > vad_threshold:
                speech_started = True
                silence_duration = 0.0
            elif speech_started:
                silence_duration += chunk_duration
                if silence_duration >= silence_threshold:
                    break  # Speech ended

    if not speech_started:
        return None

    return np.concatenate(audio_chunks)


def transcribe(audio: np.ndarray) -> str:
    """
    Transcribe audio array to text using Whisper.

    Args:
        audio: Float32 numpy array at 16kHz

    Returns:
        Transcribed text string
    """
    result = WHISPER_MODEL.transcribe(audio, language="en", fp16=False)
    return result["text"].strip()


def listen_and_transcribe() -> str | None:
    """
    Record audio and transcribe it.
    Returns transcribed text, or None if no speech detected.
    """
    audio = record_until_silence()
    if audio is None:
        return None
    return transcribe(audio)
```

**Verification:**
- [ ] `listen_and_transcribe()` records your voice and returns accurate text
- [ ] Silence (no speech) returns `None` within 5 seconds
- [ ] Background noise (TV, music) does not trigger false transcription
- [ ] Accuracy above 85% on clear speech

---

## Phase 2 — Step 3: Narration Thread

**File:** `threads/narration.py`

```python
# threads/narration.py

import threading
from core.tts import speak, stop_speaking

class NarrationThread:
    """
    Manages continuous document narration.
    Runs TTS chunk by chunk in a background thread.
    Pauses when interrupted, resumes from exact position.
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.current_chunk = 0
        self._pause_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread = None
        self._paused = False

    def start(self, from_chunk: int = 0) -> None:
        """Start narration from a specific chunk index."""
        self.current_chunk = from_chunk
        self._stop_event.clear()
        self._pause_event.clear()
        self._paused = False
        self._thread = threading.Thread(target=self._narrate_loop, daemon=True)
        self._thread.start()

    def pause(self) -> int:
        """Pause narration. Returns the chunk index where paused."""
        self._paused = True
        stop_speaking()
        return self.current_chunk

    def resume(self) -> None:
        """Resume narration from where it was paused."""
        self._paused = False
        if not self._thread or not self._thread.is_alive():
            self.start(from_chunk=self.current_chunk)

    def stop(self) -> None:
        """Stop narration completely."""
        self._stop_event.set()
        stop_speaking()

    def _narrate_loop(self) -> None:
        """Internal narration loop — runs in background thread."""
        total = self.pipeline.total_chunks()

        while self.current_chunk < total:
            if self._stop_event.is_set():
                break

            if self._paused:
                threading.Event().wait(0.1)  # Small sleep while paused
                continue

            chunk_text = self.pipeline.get_chunk(self.current_chunk)
            if chunk_text:
                speak(chunk_text, block=True)

            if not self._paused:  # Only advance if not interrupted mid-chunk
                self.current_chunk += 1

        if not self._stop_event.is_set():
            speak("Document narration complete.", block=True)
```

---

## Phase 2 — Step 4: Listener Thread

**File:** `threads/listener.py`

```python
# threads/listener.py

import threading
from core.stt import listen_and_transcribe
from core.tts import speak

class ListenerThread:
    """
    Continuously listens for voice interruptions during narration.
    When speech is detected:
    1. Pauses narration
    2. Transcribes the question
    3. Gets answer from pipeline
    4. Speaks the answer
    5. Resumes narration
    """

    def __init__(self, pipeline, narration_thread):
        self.pipeline = pipeline
        self.narration = narration_thread
        self._stop_event = threading.Event()
        self._thread = None

    def start(self) -> None:
        """Start the listener thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop listening."""
        self._stop_event.set()

    def _listen_loop(self) -> None:
        """Internal listen loop — runs concurrently with narration."""
        while not self._stop_event.is_set():
            # This blocks until speech is detected or returns None
            query = listen_and_transcribe()

            if query and not self._stop_event.is_set():
                # Step 1: Pause narration
                paused_at = self.narration.pause()
                print(f"\n[Interruption] Query: {query}")

                # Step 2: Get answer from RAG pipeline
                result = self.pipeline.answer(query)
                answer = result["answer"]
                print(f"[Answer] {answer}\n")

                # Step 3: Speak the answer
                speak(answer, block=True)

                # Step 4: Resume narration
                self.narration.resume()
```

---

## Phase 2 — Step 5: Update Gradio UI for Voice

Update `ui/gradio_app.py` to add a voice session mode alongside the text chat.

Add buttons: **"Start Narration"**, **"Stop Narration"**, and a voice status indicator. The text chat remains functional for reference.

**Verification checklist for Phase 2:**
- [ ] `Start Narration` plays the document aloud chunk by chunk
- [ ] Speaking a question mid-narration causes narration to pause immediately (< 0.5 sec)
- [ ] Question is transcribed correctly
- [ ] Answer is spoken back clearly
- [ ] Narration resumes from correct position after answer
- [ ] Multiple interruptions in one session all work correctly
- [ ] Background noise does not trigger false interruptions
- [ ] `Stop Narration` halts everything cleanly

---

---

# PHASE 3 — FastAPI Backend + Polish

> **Goal:** Move from a Gradio prototype to a proper REST API.
> This prepares ConverseIQ for mobile deployment in Phase 4.

**Endpoints to build:**

```
POST /upload          — Upload and process PDF
POST /query           — Text query (returns JSON answer + sources)
POST /voice-query     — Audio file query (STT → RAG → returns audio)
GET  /narrate/{index} — Get TTS audio for chunk at index
POST /session/clear   — Clear memory for new session
GET  /status          — Health check
```

**Verification checklist:**
- [ ] All endpoints respond correctly via curl/Postman
- [ ] `/upload` processes PDF and returns chunk count
- [ ] `/query` returns grounded answer in under 3 seconds
- [ ] `/voice-query` accepts WAV file, returns answer as WAV audio
- [ ] Error handling: missing file, empty query, model not loaded — all return clear error messages

---

---

# PHASE 4 — Mobile Deployment (Flutter + Gemma 4)

> **Goal:** Replace Ollama/Llama3 with Gemma 4 4B via MediaPipe.
> Build Flutter app that connects to the FastAPI backend (Phase 3).
> Eventually move inference fully on-device.

**Key changes from Phase 3:**
- Swap `ollama` calls for `MediaPipe LLM Inference API`
- Swap Whisper for `Whisper.cpp` via Flutter plugin
- Swap Piper for `Sherpa-ONNX` Flutter bindings
- Replace FAISS (if added) with pure BM25 — already done in Phase 1

**Flutter app screens:**
1. Home — Upload PDF
2. Player — Narration with waveform visualization + interrupt button
3. Chat — Text query fallback

---

---

# Perfection Criteria Per Phase

Use these to judge if a phase is truly complete before moving on:

## Phase 1 is perfect when:
- 10 different PDFs all load and chunk correctly
- 20 different questions all return grounded, non-hallucinated answers
- "Off-topic" questions always return "not in document"
- Multi-turn conversation context works across 5+ exchanges
- Gradio UI is stable — no crashes after 30 minutes of use

## Phase 2 is perfect when:
- 3 consecutive voice interruptions all work without desync
- Narration resumes from correct sentence every time (not from wrong chunk)
- VAD false positive rate below 5% in a normal room
- Total latency (speak → answer spoken back) under 4 seconds
- Works for 1-hour documents without memory leak

## Phase 3 is perfect when:
- All endpoints respond under 3 seconds
- Concurrent requests from 2 clients work simultaneously
- API documented via FastAPI auto-docs at `/docs`

## Phase 4 is perfect when:
- App runs on a physical Android device (not just emulator)
- Full offline — airplane mode on, app works
- Latency under 5 seconds on mid-range device

---

---

# Quick Reference — Key Decisions

| Decision | Choice | Reason |
|---|---|---|
| Vector DB | None (BM25) | Mobile memory, no GPU needed |
| LLM | Llama 3 (Ollama) → Gemma 4 | Offline, free, no API cost |
| STT | Whisper base | Best accuracy/speed tradeoff on CPU |
| TTS | Piper TTS | Fastest CPU neural TTS |
| VAD | Silero VAD | Sub-100ms detection, tiny model |
| Reranker | MiniLM CrossEncoder | 80MB, CPU, closes semantic gap |
| Framework | FastAPI + Flutter | Standard, well-documented |
| Concurrency | Python threading | I/O-bound tasks, simpler than multiprocessing |

---

*ConverseIQ — Major Project Phase-I — 22CBP69*
*Department of Computer Science and Business Systems*
*NITTE Meenakshi Institute of Technology — AY 2025–2026*
