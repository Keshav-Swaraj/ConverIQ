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

        try:
            # Extract text
            text = load_pdf(pdf_path)
        except FileNotFoundError as e:
            return {"chunks": 0, "status": f"Error: {e}"}
        except Exception as e:
            return {"chunks": 0, "status": f"Error reading PDF: {e}"}

        if not text.strip():
            return {"chunks": 0, "status": "Error: Could not extract text from PDF (may be scanned/image-based)"}

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
