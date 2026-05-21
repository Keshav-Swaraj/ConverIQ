"""Step 4 verification script for the cross-encoder reranker."""
import time
from core.reranker import Reranker

print("--- Step 4: Cross-Encoder Reranker ---")
print()

# 1. Model loads without error
r = Reranker()
print("[PASS] Model loaded without error")

# 2. 10 candidates in -> 3 out
candidates = [
    {"chunk": f"Candidate chunk number {i} about information retrieval.", "score": float(i), "index": i}
    for i in range(10)
]
results = r.rerank("test query", candidates, top_k=3)
assert len(results) == 3, f"Expected 3, got {len(results)}"
print("[PASS] 10 candidates in -> 3 out")

# 3. Semantic match: "memory management" should rank sliding-window chunk higher than upload chunk
query = "memory management"
chunks_in = [
    {
        "chunk": "The system uses a sliding window context to maintain conversation history.",
        "score": 1.0,
        "index": 0,
    },
    {
        "chunk": "Users can upload a PDF document through the file upload interface.",
        "score": 0.9,
        "index": 1,
    },
]
res = r.rerank(query, chunks_in, top_k=2)
assert res[0]["index"] == 0, (
    f"Expected memory chunk (index=0) ranked first, got index={res[0]['index']}"
)
print("[PASS] Memory chunk scores higher than upload chunk for 'memory management' query")

# 4. Speed test for 10 candidates on CPU
t0 = time.time()
r.rerank("performance test query", candidates, top_k=3)
elapsed = time.time() - t0
assert elapsed < 2.0, f"Expected <2s, got {elapsed:.2f}s"
print(f"[PASS] Runs in {elapsed:.3f}s for 10 candidates (< 2s)")

# 5. Empty candidates -> empty list, no crash
result = r.rerank("anything", [], top_k=3)
assert result == [], f"Expected empty list, got {result}"
print("[PASS] Empty candidates -> empty list, no crash")

print()
print("All Step 4 checks PASSED ✓")
