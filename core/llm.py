# core/llm.py

import ollama

SYSTEM_PROMPT = """You are ConverseIQ, an intelligent document assistant.
You help users understand documents by answering their questions.

STRICT RULES:
1. Answer ONLY using the context provided below.
2. If the answer is not in the context, say: "I couldn't find that in the document." (However, for general greetings like "hi" or "hello", or requests asking what the document is about, respond politely and briefly using the provided document introduction/context).
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

    try:
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
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Ollama (model='{model}'). "
            f"Make sure Ollama is running: `ollama serve`. Error: {e}"
        )


def generate_answer_stream(
    query: str,
    context_chunks: list[str],
    history: list[dict],
    model: str = "llama3"
):
    """
    Generate a grounded answer using Ollama, yielding tokens in real time.

    Args:
        query: User's question
        context_chunks: Reranked document chunks
        history: Conversation history
        model: Ollama model name

    Yields:
        Individual string tokens from response
    """
    messages = build_prompt(query, context_chunks, history)

    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            stream=True,
            options={
                "temperature": 0.3,    # Low temperature = more factual
                "top_p": 0.9,
                "num_predict": 300,    # Max ~300 tokens in answer
            }
        )
        for chunk in response:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to Ollama (model='{model}'). "
            f"Make sure Ollama is running: `ollama serve`. Error: {e}"
        )

