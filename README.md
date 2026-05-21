# 🎙️ ConverseIQ

> **AI-powered, voice-first document learning system.**  
> Upload a PDF. Listen to it narrated aloud. Interrupt with a spoken question. Get a grounded answer. Resume exactly where you left off.

---

## What Makes It Unique

| Feature | Details |
|---|---|
| **Vectorless RAG** | BM25 sparse retrieval + cross-encoder reranking — no FAISS, no embeddings, no GPU |
| **Fully Offline** | Whisper STT + Ollama LLM + Piper TTS — zero cloud calls, zero API keys |
| **Always Listening** | Narration and voice input run on separate threads concurrently |
| **Mobile-First Design** | Built to run on mid-range phones via Gemma 4 + Sherpa-ONNX (Phase 4) |

---

## Architecture

```
PDF → Extract → Chunk → BM25 Index
                              ↓
User Question → BM25 Retrieve (top-10) → CrossEncoder Rerank (top-3) → LLM Answer
                                                                              ↓
                                                               Speak answer → Resume narration
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| PDF Parsing | `pdfplumber`, `PyPDF2` |
| Text Chunking | `NLTK` sentence tokenizer |
| Retrieval | `rank-bm25` (BM25Okapi) |
| Reranking | `sentence-transformers` CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM | `Ollama` + `llama3` (local) |
| STT | `openai-whisper` (base) |
| VAD | `silero-vad` |
| TTS | `piper-tts` |
| UI | `Gradio` (Phase 1), `Flutter` (Phase 4) |
| API | `FastAPI` (Phase 3) |

---

## Project Structure

```
converseiq/
├── main.py                  # Entry point — starts Gradio UI
├── requirements.txt
├── .env                     # Ollama URL, model name
│
├── core/
│   ├── document.py          # PDF loading, text extraction, chunking
│   ├── index.py             # BM25 index — build, retrieve, save, load
│   ├── reranker.py          # Cross-encoder reranking
│   ├── llm.py               # Ollama LLM + prompt template
│   ├── memory.py            # Sliding window conversation history
│   └── pipeline.py          # Full RAG orchestrator
│
├── threads/
│   ├── narration.py         # TTS narration thread (Phase 2)
│   └── listener.py          # VAD + STT listener thread (Phase 2)
│
├── api/
│   └── server.py            # FastAPI REST endpoints (Phase 3)
│
├── ui/
│   └── gradio_app.py        # Gradio interface
│
└── tests/
    ├── test_document.py
    ├── test_index.py
    └── test_pipeline.py
```

---

## Setup & Run

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running

### 1. Clone & create virtual environment

```bash
git clone https://github.com/Keshav-Swaraj/ConverIQ.git
cd ConverIQ
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull the LLM

```bash
ollama pull llama3
```

### 4. Start the app

```bash
python main.py
```

Open **http://localhost:7860** in your browser.

---

## Usage

1. **Upload a PDF** using the left panel
2. **Ask any question** in the chat box
3. The system retrieves the 3 most relevant passages and generates a grounded answer
4. Source excerpts are shown below each answer
5. Ask follow-up questions — the system remembers the last 5 exchanges

---

## Phase Roadmap

| Phase | Status | What it adds |
|---|---|---|
| **Phase 1** | ✅ Complete | PDF → BM25 → Rerank → LLM → Gradio text UI |
| **Phase 2** | 🔲 Next | Piper TTS narration + Whisper STT + concurrent threads |
| **Phase 3** | 🔲 Planned | FastAPI REST backend |
| **Phase 4** | 🔲 Planned | Flutter mobile app + Gemma 4 on-device |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

**37 tests, all passing.**

---

## Research Contributions

1. **Vectorless RAG** — Eliminates the need for a vector database by combining BM25 keyword retrieval with cross-encoder semantic reranking. Achieves near-dense retrieval quality at a fraction of the memory and compute cost.

2. **Fully Offline, Mobile-First** — Complete pipeline (STT → RAG → LLM → TTS) runs on-device without any internet connection or API calls.

---

*Department of Computer Science and Business Systems*  
*NITTE Meenakshi Institute of Technology — AY 2025–2026*  
*22CBP69*
