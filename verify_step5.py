"""Step 5 verification script — mocked LLM (runs without Ollama)."""
from unittest.mock import patch, MagicMock
from core.llm import build_prompt, generate_answer, SYSTEM_PROMPT

print("--- Step 5: LLM Answer Generation ---")
print()

# 1. build_prompt constructs correct message structure
context_chunks = [
    "The BM25 algorithm ranks documents by term frequency and inverse document frequency.",
    "Retrieval-Augmented Generation combines a retriever with a language model.",
    "Cross-encoders score query-document pairs for semantic relevance.",
]
history = [
    {"role": "user", "content": "What is RAG?"},
    {"role": "assistant", "content": "RAG stands for Retrieval-Augmented Generation."},
]
messages = build_prompt("How does cross-encoding work?", context_chunks, history)

assert messages[0]["role"] == "system", "First message must be system"
assert SYSTEM_PROMPT in messages[0]["content"], "System prompt must be included"
assert "Excerpt 1" in messages[0]["content"], "Context excerpts must be in system message"
assert messages[1]["role"] == "user", "History user message at index 1"
assert messages[1]["content"] == "What is RAG?", "History user content correct"
assert messages[-1]["role"] == "user", "Last message is the current query"
assert messages[-1]["content"] == "How does cross-encoding work?", "Last message content correct"
print("[PASS] build_prompt constructs correct message structure")

# 2. History is limited to last 10 messages (5 turns)
long_history = []
for i in range(10):  # 10 turns = 20 messages
    long_history.append({"role": "user", "content": f"Q{i}"})
    long_history.append({"role": "assistant", "content": f"A{i}"})

messages = build_prompt("Final question", context_chunks, long_history)
# Should only include last 10 history messages (not 20)
history_messages = [m for m in messages if m != messages[0] and m != messages[-1]]
assert len(history_messages) <= 10, f"Expected <=10 history messages, got {len(history_messages)}"
print("[PASS] History limited to last 10 messages")

# 3. generate_answer works with mocked Ollama
mock_response = {"message": {"content": "  This is the grounded answer from the document.  "}}
with patch("core.llm.ollama.chat", return_value=mock_response):
    answer = generate_answer(
        query="What is BM25?",
        context_chunks=context_chunks,
        history=[],
        model="llama3"
    )
    assert answer == "This is the grounded answer from the document."
    print("[PASS] generate_answer returns stripped answer string")

# 4. Model not running -> clear error message
with patch("core.llm.ollama.chat", side_effect=Exception("Connection refused")):
    try:
        generate_answer("test", context_chunks, [])
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Ollama" in str(e) or "ollama" in str(e).lower()
        print("[PASS] Model not running -> clear RuntimeError, not silent failure")

# 5. Empty context chunks handled without crash
with patch("core.llm.ollama.chat", return_value=mock_response):
    answer = generate_answer("test query", [], [])
    assert isinstance(answer, str)
    print("[PASS] Empty context chunks -> no crash")

print()
print("All Step 5 checks PASSED")
