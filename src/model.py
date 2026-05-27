"""模型定义模块"""

import torch
import torch.nn as nn
import torchvision.models as models

from .config import NUM_CLASSES


def create_efficientnet_b0(num_classes=NUM_CLASSES, pretrained=True):
    """创建 EfficientNet-B0 模型，全网络微调

    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重

    Returns:
        model: EfficientNet-B0 模型
    """
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # 替换分类头
    # 自定义: Dropout(0.3) → Linear(1280, 512) → ReLU → Dropout(0.2) → Linear(512, num_classes)
    num_features = model.classifier[1].in_features  # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    return model


def create_resnet50(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=True):
    """创建 ResNet-50 模型（备选方案：冻结层微调）

    Args:
        num_classes: 分类类别数
        pretrained: 是否使用预训练权重
        freeze_backbone: 是否冻结骨干网络

    Returns:
        model: ResNet-50 模型
    """
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features  # 2048
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(512, num_classes),
    )

    return model


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total:,}")
    print(f"可训练参数量: {trainable:,}")
    print(f"冻结参数量: {total - trainable:,}")
    return total, trainable


if __name__ == "__main__":
    model = create_efficientnet_b0()
    count_parameters(model)
