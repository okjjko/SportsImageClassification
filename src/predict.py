"""推理模块"""

import pickle

import torch
from PIL import Image

from .config import BEST_MODEL_PATH
from .dataset import get_val_transforms
from .model import create_efficientnet_b0


def load_model(model_path=None):
    """从 checkpoint 加载模型和 class_names

    Args:
        model_path: 模型 checkpoint 路径，默认使用最佳模型

    Returns:
        model, class_names, device

    Raises:
        FileNotFoundError: 模型文件不存在
        ValueError: 模型文件损坏或格式不正确
    """
    if model_path is None:
        model_path = BEST_MODEL_PATH

    if not model_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}\n请先运行 python run_train.py 训练模型"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as e:
        raise ValueError(
            f"模型文件损坏或格式不正确: {model_path}\n请删除后重新训练"
        ) from e

    # 验证 checkpoint 结构完整性
    required_keys = {"class_names", "model_state_dict"}
    missing_keys = required_keys - set(checkpoint.keys())
    if missing_keys:
        raise ValueError(
            f"模型文件缺少必要的键: {missing_keys}\n文件可能已损坏，请重新训练"
        )

    class_names = checkpoint["class_names"]

    try:
        model = create_efficientnet_b0(num_classes=len(class_names), pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        raise ValueError(
            f"模型权重与网络结构不匹配，请确认模型文件与当前代码版本一致\n详情: {e}"
        ) from e

    model = model.to(device)
    model.eval()

    print(f"模型加载完成: {model_path}")
    print(f"类别数: {len(class_names)}, 设备: {device}")
    if "val_acc" in checkpoint:
        print(f"训练时最佳验证准确率: {checkpoint['val_acc']:.2f}%")

    return model, class_names, device


def predict_image(model, image, class_names, device, top_k=3):
    """单张图片推理

    Args:
        model: 模型
        image: PIL Image
        class_names: 类别名称列表
        device: 计算设备
        top_k: 返回 Top-K 结果

    Returns:
        list of dicts with "class" and "confidence" keys
    """
    transform = get_val_transforms()

    if image.mode != "RGB":
        image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, top_k)

    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
        results.append({
            "class": class_names[idx],
            "confidence": float(prob),
        })

    return results
