"""数据加载与预处理模块"""

import os
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import (
    BATCH_SIZE,
    IMAGE_SIZE,
    NUM_WORKERS,
    TEST_DIR,
    TRAIN_DIR,
    VALID_DIR,
)


def get_train_transforms():
    """训练集数据增强"""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms():
    """验证集/测试集预处理"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """创建训练、验证、测试 DataLoader

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_dataset = datasets.ImageFolder(
        root=str(TRAIN_DIR),
        transform=get_train_transforms(),
    )
    val_dataset = datasets.ImageFolder(
        root=str(VALID_DIR),
        transform=get_val_transforms(),
    )
    test_dataset = datasets.ImageFolder(
        root=str(TEST_DIR),
        transform=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names


def analyze_dataset():
    """分析数据集类别分布"""
    print("=" * 60)
    print("数据集分析")
    print("=" * 60)

    for name, directory in [("训练集", TRAIN_DIR), ("验证集", VALID_DIR), ("测试集", TEST_DIR)]:
        if not directory.exists():
            print(f"  {name}: 目录不存在 ({directory})")
            continue

        dataset = datasets.ImageFolder(root=str(directory))
        class_counts = Counter(dataset.targets)
        total = len(dataset)
        num_classes = len(dataset.classes)

        print(f"\n{name}:")
        print(f"  总图片数: {total}")
        print(f"  类别数: {num_classes}")
        print(f"  每类平均: {total / num_classes:.1f} 张")

        counts = list(class_counts.values())
        print(f"  最少类别: {min(counts)} 张")
        print(f"  最多类别: {max(counts)} 张")

    print("=" * 60)


if __name__ == "__main__":
    analyze_dataset()
