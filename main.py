# main.py

from ui.gradio_app import app
import gradio as gr

if __name__ == "__main__":
    app.launch(
        share=False,
        server_port=7860,
        theme=gr.themes.Soft()
    )
