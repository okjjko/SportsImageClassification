"""Gradio Web 界面 — Claude 设计风格"""

import base64
import os
from pathlib import Path

import gradio as gr
from PIL import Image

from .config import BEST_MODEL_PATH
from .predict import load_model, predict_image

_model = None
_class_names = None
_device = None


def _font_css():
    font_dir = Path(__file__).resolve().parent.parent / "public" / "fonts"
    css = ""
    for fname, family, style, weight in [
        ("Inter-VF.woff2", "Inter", "normal", "400 700"),
        ("Inter-Italic-VF.woff2", "Inter", "italic", "400 700"),
        ("SourceSerif4-VF.woff2", "Source Serif 4", "normal", "400 700"),
        ("SourceSerif4-Italic-VF.woff2", "Source Serif 4", "italic", "400 700"),
        ("JetBrainsMono-VF.ttf", "JetBrains Mono", "normal", "400 700"),
        ("JetBrainsMono-Italic-VF.ttf", "JetBrains Mono", "italic", "400 700"),
    ]:
        font_path = font_dir / fname
        if font_path.exists():
            data = base64.b64encode(font_path.read_bytes()).decode()
            fmt = "woff2" if fname.endswith(".woff2") else "truetype"
            mime = "font/woff2" if fname.endswith(".woff2") else "font/ttf"
            css += f"""@font-face {{
                font-family: '{family}';
                src: url(data:{mime};charset=utf-8;base64,{data}) format('{fmt}');
                font-style: {style};
                font-weight: {weight};
                font-display: swap;
            }}"""
    return css


def _global_css():
    return f"""
{_font_css()}

:root {{
    --parchment: #f5f4ed;
    --ivory: #faf9f5;
    --near-black: #141413;
    --olive-gray: #5e5d59;
    --stone-gray: #87867f;
    --terracotta: #c96442;
    --border-cream: #f0eee6;
    --border-warm: #e8e6dc;
    --warm-sand: #e8e6dc;
    --charcoal-warm: #4d4c48;
}}

html, body, .gradio-container {{
    background: var(--parchment) !important;
    font-family: 'Inter', Arial, sans-serif !important;
    color: var(--near-black);
}}

.gradio-container {{
    max-width: 1100px !important;
    margin: 0 auto !important;
    padding: 24px 16px !important;
}}

h1, h2, h3, h4 {{
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-weight: 500 !important;
    color: var(--near-black);
    letter-spacing: -0.01em;
}}

.app-header {{
    background: var(--ivory) !important;
    border: 1px solid var(--border-cream) !important;
    border-radius: 16px !important;
    padding: 24px 32px !important;
    box-shadow: rgba(0,0,0,0.04) 0px 4px 20px !important;
    margin-bottom: 20px !important;
}}

.app-header h1 {{
    font-size: 28px !important;
    margin: 0 0 4px 0 !important;
    line-height: 1.2 !important;
}}

.app-header p {{
    font-family: 'Inter', Arial, sans-serif !important;
    font-size: 15px !important;
    color: var(--olive-gray) !important;
    margin: 0 !important;
    line-height: 1.5 !important;
}}

.card {{
    background: var(--ivory) !important;
    border: 1px solid var(--border-cream) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    box-shadow: rgba(0,0,0,0.03) 0px 2px 12px !important;
    height: 100%;
}}

.card h3 {{
    font-size: 18px !important;
    margin: 0 0 12px 0 !important;
    color: var(--near-black);
}}

.card p, .card label {{
    font-size: 14px !important;
    color: var(--olive-gray) !important;
}}

.upload-area {{
    border: 2px dashed var(--border-warm) !important;
    border-radius: 12px !important;
    background: var(--parchment) !important;
    padding: 20px !important;
    transition: border-color 0.2s;
}}

.upload-area:hover {{
    border-color: var(--terracotta) !important;
}}

button, .gr-button {{
    font-family: 'Inter', Arial, sans-serif !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.15s ease !important;
}}

.gr-button-primary {{
    background: var(--terracotta) !important;
    border: none !important;
    color: #fff !important;
    box-shadow: 0px 0px 0px 1px var(--terracotta) !important;
}}

.gr-button-primary:hover {{
    background: #b85a3a !important;
    box-shadow: 0px 0px 0px 2px #d97757 !important;
}}

.gr-button-secondary {{
    background: var(--warm-sand) !important;
    border: none !important;
    color: var(--charcoal-warm) !important;
    box-shadow: 0px 0px 0px 1px #d1cfc5 !important;
}}

.gr-button-secondary:hover {{
    background: #dddacf !important;
}}

input, textarea, select, .gr-input, .gr-dropdown {{
    border-radius: 12px !important;
    border: 1px solid var(--border-warm) !important;
    background: var(--parchment) !important;
    font-family: 'Inter', Arial, sans-serif !important;
    font-size: 14px !important;
    padding: 8px 12px !important;
    transition: border-color 0.2s;
}}

input:focus, textarea:focus, select:focus {{
    border-color: var(--terracotta) !important;
    box-shadow: 0 0 0 2px rgba(201, 100, 66, 0.15) !important;
}}

.gr-label {{
    font-family: 'Inter', Arial, sans-serif !important;
}}

.gr-label .label-text {{
    font-weight: 500 !important;
    color: var(--near-black) !important;
}}

.gr-label .probability-bar {{
    border-radius: 4px !important;
    background: var(--terracotta) !important;
}}

.footer {{
    text-align: center !important;
    padding: 16px 0 8px 0 !important;
    border-top: 1px solid var(--border-cream) !important;
    margin-top: 20px !important;
}}

.footer p {{
    font-family: 'Inter', Arial, sans-serif !important;
    font-size: 12px !important;
    color: var(--stone-gray) !important;
    margin: 2px 0 !important;
}}

.gr-gallery {{
    border: none !important;
}}

.gr-box {{
    border: 1px solid var(--border-cream) !important;
    border-radius: 8px !important;
}}

.example-label {{
    font-family: 'Inter', Arial, sans-serif !important;
    font-size: 13px !important;
    color: var(--olive-gray) !important;
    margin-bottom: 8px !important;
}}

.gr-sample-text {{
    font-family: 'Inter', Arial, sans-serif !important;
}}

button[aria-label="设置"], .settings {{
    display: none !important;
}}
"""


def _get_model():
    global _model, _class_names, _device
    if _model is None:
        if not os.path.exists(str(BEST_MODEL_PATH)):
            raise FileNotFoundError(
                f"模型文件不存在: {BEST_MODEL_PATH}\n请先运行 python run_train.py 训练模型"
            )
        _model, _class_names, _device = load_model()
    return _model, _class_names, _device


def predict(image):
    if image is None:
        return {"请上传一张图片": 1.0}

    try:
        if isinstance(image, str):
            image = Image.open(image)
        elif not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        model, class_names, device = _get_model()
        results = predict_image(model, image, class_names, device, top_k=3)

        output = {}
        for r in results:
            output[r["class"]] = r["confidence"]
        return output

    except FileNotFoundError as e:
        return {"模型未找到，请先训练": 1.0}
    except Exception as e:
        return {f"处理出错: {str(e)}": 1.0}


def create_interface():
    try:
        model, class_names, device = _get_model()
        has_model = True
    except FileNotFoundError:
        has_model = False
        class_names = []

    def get_status():
        if not has_model:
            return "模型未加载，请先训练"
        return f"最佳验证准确率: 97.20% | Top-3准确率: 99.60%"

    example_images = []
    test_base = Path(__file__).resolve().parent.parent / "archive" / "test"
    example_cats = ["basketball", "swimming", "tennis", "snow boarding", "surfing", "weightlifting"]
    for cat in example_cats:
        cat_dir = test_base / cat
        if cat_dir.exists():
            files = sorted(cat_dir.iterdir())
            if files:
                example_images.append(str(files[0]))

    with gr.Blocks(
        css=_global_css(),
        theme=gr.themes.Soft(),
        title="Sports Image Classification",
    ) as interface:
        gr.HTML(
            f"""
            <div class="app-header">
                <h1>Sports Image Classification</h1>
                <p>基于 EfficientNet-B0 深度学习模型的 100 类运动项目智能识别系统</p>
            </div>
            """
        )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=320):
                with gr.Group(elem_classes="card"):
                    gr.HTML('<h3>上传图片</h3>')
                    image_input = gr.Image(
                        type="pil",
                        label="",
                        show_label=False,
                        elem_classes="upload-area",
                    )

                    if example_images:
                        gr.HTML('<div class="example-label">快速测试 — 点击下方示例图片</div>')
                        gr.Examples(
                            examples=example_images,
                            inputs=image_input,
                            label="",
                            examples_per_page=6,
                        )

            with gr.Column(scale=2, min_width=280):
                with gr.Group(elem_classes="card"):
                    gr.HTML('<h3>识别结果</h3>')
                    label_output = gr.Label(
                        num_top_classes=3,
                        label="",
                        show_label=False,
                    )

        status_text = get_status()
        gr.HTML(
            f"""
            <div class="footer">
                <p>模型: EfficientNet-B0 · 100 个运动类别 · {status_text}</p>

            </div>
            """
        )

        image_input.change(
            fn=predict,
            inputs=image_input,
            outputs=label_output,
        )

    return interface
