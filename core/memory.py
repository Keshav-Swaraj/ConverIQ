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
