# ui/gradio_app.py

import gradio as gr
from core.pipeline import ConverseIQPipeline

pipeline = ConverseIQPipeline()


def upload_document(file):
    if file is None:
        return "No file uploaded."
    result = pipeline.load_document(file)
    return f"✅ {result['status']} ({result['chunks']} chunks indexed)"


def normalize_history(history):
    if not history:
        return []
    normalized = []
    for item in history:
        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                user_msg = item[0]
                bot_msg = item[1]
                normalized.append({"role": "user", "content": str(user_msg)})
                normalized.append({"role": "assistant", "content": str(bot_msg)})
            elif len(item) == 1:
                normalized.append({"role": "user", "content": str(item[0])})
        elif isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role and content is not None:
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            text_parts.append(part["text"])
                        else:
                            text_parts.append(str(part))
                    content = "".join(text_parts)
                elif isinstance(content, dict) and "text" in content:
                    content = content["text"]
                else:
                    content = str(content)
                normalized.append({"role": role, "content": content})
        else:
            # Skip invalid structures to keep list-of-dicts strict
            pass
    return normalized


def ask_question_stream(question, history):
    if not question.strip():
        yield history, ""
        return

    history = normalize_history(history)

    # Display user query with immediate thinking placeholder
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": "⏳ *Thinking...*"})
    yield history, ""

    if not pipeline.document_loaded:
        history[-1]["content"] = "⚠️ Please upload a PDF document first before asking questions."
        yield history, ""
        return

    try:
        # Clear thinking status and start streaming from pipeline
        history[-1]["content"] = ""
        sources_used = []
        
        for chunk in pipeline.answer_stream(question):
            if not chunk["done"]:
                history[-1]["content"] += chunk["token"]
                yield history, ""
            else:
                sources_used = chunk["sources"]

        # Append source chunks at the very end
        if sources_used:
            sources_text = "\n\n".join([
                f"📄 **Source {i+1}:**\n{src[:300]}..."
                for i, src in enumerate(sources_used)
            ])
            history[-1]["content"] += f"\n\n---\n**Sources used:**\n{sources_text}"
        
        yield history, ""
    except Exception as e:
        history[-1]["content"] = f"⚠️ Error: {e}"
        yield history, ""




with gr.Blocks(title="ConverseIQ — Talk to Your Documents") as app:
    gr.Markdown(
        """# 🎙️ ConverseIQ
**Talk to your documents — AI-powered, fully offline**

Upload a PDF, then ask any question about it. The system retrieves the most relevant excerpts and generates a grounded, context-only answer.
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Upload Document")
            file_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Upload a PDF to begin..."
            )
            file_input.change(upload_document, inputs=file_input, outputs=upload_status)

        with gr.Column(scale=2):
            gr.Markdown("### 💬 Ask Questions")
            chatbot = gr.Chatbot(
                height=450,
                label="Conversation",
            )
            question_input = gr.Textbox(
                placeholder="Ask anything about the document...",
                label="Your Question",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("Ask", variant="primary")
                clear_btn = gr.Button("Clear Chat")

            submit_btn.click(
                ask_question_stream,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            question_input.submit(
                ask_question_stream,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            clear_btn.click(lambda: [], outputs=chatbot)  # reset to empty messages list

    gr.Markdown(
        """---
*ConverseIQ — Vectorless RAG • Fully Offline • Phase 1 Text Mode*
*NITTE Meenakshi Institute of Technology — 22CBP69*"""
    )
