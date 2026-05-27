"""Gradio Web 界面"""

import os

import gradio as gr
from PIL import Image

from .config import BEST_MODEL_PATH
from .predict import load_model, predict_image

_model = None
_class_names = None
_device = None


def _get_model():
    """懒加载模型"""
    global _model, _class_names, _device
    if _model is None:
        if not os.path.exists(str(BEST_MODEL_PATH)):
            raise FileNotFoundError(
                f"模型文件不存在: {BEST_MODEL_PATH}\n请先运行 python run_train.py 训练模型"
            )
        _model, _class_names, _device = load_model()
    return _model, _class_names, _device


def predict(image):
    """预测函数，供 Gradio 调用"""
    if image is None:
        return {"error": 1.0}

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
    """创建 Gradio 界面"""
    try:
        model, class_names, device = _get_model()
        desc = f"上传一张运动项目的图片，系统将识别出是哪种运动（共 {len(class_names)} 种类别）。"
    except FileNotFoundError:
        desc = "模型未加载。请先运行 python run_train.py 训练模型。"

    interface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="上传一张运动图片"),
        outputs=gr.Label(num_top_classes=3, label="Top-3 预测结果"),
        title="Sports Image Classification",
        description=desc,
        theme=gr.themes.Soft(),
    )
    return interface
