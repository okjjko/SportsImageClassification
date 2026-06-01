"""Gradio 启动入口

使用方法:
    python run_app.py
"""

import gradio as gr

from src.app import create_interface


if __name__ == "__main__":
    interface, css = create_interface()
    interface.launch(
        share=False,
        css=css,
        theme=gr.themes.Soft(),
    )
