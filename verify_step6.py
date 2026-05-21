"""Step 6 verification script for ConversationMemory."""
from core.memory import ConversationMemory

print("--- Step 6: ConversationMemory ---")
print()

# 1. Add Q&A exchanges
mem = ConversationMemory(max_turns=5)
mem.add("What is RAG?", "RAG stands for Retrieval-Augmented Generation.")
assert len(mem.history) == 2
print("[PASS] Add single exchange -> 2 messages")

# 2. Roles are correct
assert mem.history[0]["role"] == "user"
assert mem.history[1]["role"] == "assistant"
print("[PASS] Roles correct: user / assistant")

# 3. Sliding window: 6 exchanges, max_turns=5 -> only last 5 kept
for i in range(5):
    mem.add(f"Question {i}", f"Answer {i}")
assert len(mem) == 5, f"Expected 5 turns, got {len(mem)}"
assert "What is RAG?" not in " ".join(m["content"] for m in mem.history)
print("[PASS] Sliding window trims oldest exchanges (6 in, 5 kept)")

# 4. get() returns copy (mutations don't affect internal state)
hist = mem.get()
hist.append({"role": "user", "content": "INJECTED"})
assert len(mem.history) == 10  # 5 turns * 2
print("[PASS] get() returns copy, not reference")

# 5. clear() empties history
mem.clear()
assert mem.history == []
assert len(mem) == 0
print("[PASS] clear() empties history")

print()
print("All Step 6 checks PASSED ✓")
