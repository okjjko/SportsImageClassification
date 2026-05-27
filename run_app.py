"""Gradio 启动入口

使用方法:
    python run_app.py
"""

from src.app import create_interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=False)
