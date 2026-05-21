# ui/gradio_app.py

import gradio as gr
from core.pipeline import ConverseIQPipeline

pipeline = ConverseIQPipeline()


def upload_document(file):
    if file is None:
        return "No file uploaded."
    result = pipeline.load_document(file)
    return f"✅ {result['status']} ({result['chunks']} chunks indexed)"


def ask_question(question, history):
    if not question.strip():
        return history, ""

    history = history or []

    if not pipeline.document_loaded:
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": "⚠️ Please upload a PDF document first before asking questions."})
        return history, ""

    result = pipeline.answer(question)
    answer = result["answer"]

    # Format sources
    sources_text = "\n\n".join([
        f"📄 **Source {i+1}:**\n{src[:300]}..."
        for i, src in enumerate(result["sources"])
    ])

    full_response = f"{answer}\n\n---\n**Sources used:**\n{sources_text}" if sources_text else answer
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": full_response})
    return history, ""


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
                ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            question_input.submit(
                ask_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            clear_btn.click(lambda: [], outputs=chatbot)  # reset to empty messages list

    gr.Markdown(
        """---
*ConverseIQ — Vectorless RAG • Fully Offline • Phase 1 Text Mode*
*NITTE Meenakshi Institute of Technology — 22CBP69*"""
    )
