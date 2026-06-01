"""数据加载与预处理模块"""

import random
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import (
    BATCH_SIZE,
    FIGURE_DIR,
    IMAGE_SIZE,
    MATPLOTLIB_FONT,
    NUM_WORKERS,
    TEST_DIR,
    TRAIN_DIR,
    VALID_DIR,
)

# 设置中文字体
plt.rcParams["font.sans-serif"] = [MATPLOTLIB_FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def get_train_transforms():
    """训练集数据增强流水线。

    包含随机裁剪、水平翻转、旋转、色彩抖动四项增强，
    每项增强对应运动图片拍摄中的真实变化（构图、镜像、角度、光照）。
    最后使用 ImageNet 标准参数（mean/std）进行归一化。

    Returns:
        transforms.Compose: 训练集变换序列
    """
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
    """验证集/测试集预处理流水线。

    仅做 Resize(256) + CenterCrop(224) + ImageNet 归一化，
    不使用随机增强以确保评估结果可复现。

    Returns:
        transforms.Compose: 验证/测试集变换序列
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """创建训练、验证、测试 DataLoader。

    使用 ImageFolder 从目录结构自动推断类别，
    训练集启用 shuffle 和数据增强，验证/测试集仅做标准化预处理。

    Args:
        batch_size (int): 批大小，默认取自 config.BATCH_SIZE
        num_workers (int): 数据加载子进程数，默认取自 config.NUM_WORKERS

    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)

    Raises:
        FileNotFoundError: 数据集目录不存在或为空
    """
    # 预检查数据集目录
    for name, directory in [("训练集", TRAIN_DIR), ("验证集", VALID_DIR), ("测试集", TEST_DIR)]:
        if not directory.exists():
            raise FileNotFoundError(
                f"{name}目录不存在: {directory}\n请确认数据集已正确放置到 archive/ 目录下"
            )
        if not any(directory.iterdir()):
            raise FileNotFoundError(
                f"{name}目录为空: {directory}\n请检查数据集是否完整"
            )

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
    """分析数据集类别分布并生成可视化图表。

    统计训练集、验证集、测试集的图片总数、类别数、每类图片数量，
    绘制类别分布柱状图并保存到 outputs/figures/ 目录。
    """
    print("=" * 60)
    print("数据集分析")
    print("=" * 60)

    splits = []
    for name, directory in [("训练集", TRAIN_DIR), ("验证集", VALID_DIR), ("测试集", TEST_DIR)]:
        if not directory.exists():
            print(f"  {name}: 目录不存在 ({directory})")
            continue

        dataset = datasets.ImageFolder(root=str(directory))
        class_counts = Counter(dataset.targets)
        total = len(dataset)
        num_classes = len(dataset.classes)

        counts = list(class_counts.values())

        print(f"\n{name}:")
        print(f"  总图片数: {total}")
        print(f"  类别数: {num_classes}")
        print(f"  每类平均: {total / num_classes:.1f} 张")
        print(f"  最少类别: {min(counts)} 张")
        print(f"  最多类别: {max(counts)} 张")
        if min(counts) > 0:
            print(f"  不平衡比 (max/min): {max(counts) / min(counts):.2f}")

        splits.append((name, dataset, class_counts, counts))

    # 打印训练集逐类明细（前 10 + 后 10）
    if splits:
        train_name, train_ds, train_counts, _ = splits[0]
        class_names = train_ds.classes
        sorted_classes = sorted(
            range(len(class_names)),
            key=lambda i: train_counts[i],
            reverse=True,
        )
        print(f"\n训练集类别分布（样本数最多的 10 个）:")
        print("-" * 50)
        for idx in sorted_classes[:10]:
            print(f"  {class_names[idx]:30s} {train_counts[idx]:>5d} 张")
        print("  ...")
        print(f"\n训练集类别分布（样本数最少的 10 个）:")
        print("-" * 50)
        for idx in sorted_classes[-10:]:
            print(f"  {class_names[idx]:30s} {train_counts[idx]:>5d} 张")

    # 绘制训练集类别分布柱状图
    if splits:
        train_name, train_ds, train_counts, _ = splits[0]
        class_names = train_ds.classes
        counts_sorted = [train_counts[i] for i in range(len(class_names))]
        names_sorted = [class_names[i] for i in range(len(class_names))]

        fig, ax = plt.subplots(figsize=(20, 6))
        x = np.arange(len(names_sorted))
        ax.bar(x, counts_sorted, color="steelblue", alpha=0.8)
        ax.set_xlabel("类别", fontsize=12)
        ax.set_ylabel("图片数量", fontsize=12)
        ax.set_title("训练集类别分布", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(names_sorted, rotation=90, fontsize=5)
        ax.axhline(
            y=np.mean(counts_sorted), color="red", linestyle="--",
            label=f"平均值: {np.mean(counts_sorted):.1f}",
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()

        save_path = FIGURE_DIR / "class_distribution.png"
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
        print(f"\n类别分布图已保存至: {save_path}")
        plt.close(fig)

    print("=" * 60)


def visualize_samples(num_per_class=1, max_classes=20):
    """从训练集每个类别中随机抽取样本图片，生成网格展示图。

    Args:
        num_per_class (int): 每个类别抽取的样本数，默认 1
        max_classes (int): 最多展示的类别数，默认 20（选取训练集前 20 个类别）
    """
    if not TRAIN_DIR.exists():
        print(f"训练集目录不存在: {TRAIN_DIR}")
        return

    class_dirs = sorted([d for d in TRAIN_DIR.iterdir() if d.is_dir()])
    if not class_dirs:
        print("训练集中没有类别子目录")
        return

    selected_classes = class_dirs[:max_classes]
    n_cols = 5
    n_rows = (len(selected_classes) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    if n_rows == 1:
        axes = [axes]
    axes_flat = [ax for row in axes for ax in row]

    for i, ax in enumerate(axes_flat):
        if i < len(selected_classes):
            class_dir = selected_classes[i]
            class_name = class_dir.name
            images = list(class_dir.glob("*.[jJ][pP][gG]"))
            images += list(class_dir.glob("*.[pP][nN][gG]"))
            images += list(class_dir.glob("*.[jJ][pP][eE][gG]"))

            if images:
                img_path = random.choice(images)
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(class_name, fontsize=9)
        ax.axis("off")

    plt.suptitle("训练集样本展示（每类随机 1 张）", fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = FIGURE_DIR / "sample_grid.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"样本展示图已保存至: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    analyze_dataset()
