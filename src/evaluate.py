"""评估与可视化模块"""

import pickle

import matplotlib
# 必须在 pyplot 导入之前设置后端，否则会弹窗报错
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from .config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    FIGURE_DIR,
    MATPLOTLIB_FONT,
    NUM_WORKERS,
    TEST_DIR,
)
from .dataset import get_val_transforms
from .model import create_efficientnet_b0

# 设置中文字体
plt.rcParams["font.sans-serif"] = [MATPLOTLIB_FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_training_history(history, save_path=None):
    """绘制训练曲线（loss、acc、lr）。

    生成 1×3 子图，分别展示 Loss、Accuracy、Learning Rate 随 epoch 的变化趋势。

    Args:
        history (dict): train_model() 返回的训练历史字典，
            包含 train_loss, val_loss, train_acc, val_acc, lr 五个列表
        save_path (str | Path, optional): 图片保存路径，默认 FIGURE_DIR/training_history.png
    """
    if save_path is None:
        save_path = FIGURE_DIR / "training_history.png"

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss 曲线
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 曲线
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=4)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate 曲线
    axes[2].plot(epochs, history["lr"], "g-o", markersize=4)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"训练曲线已保存至: {save_path}")
    plt.close(fig)


def evaluate_on_test():
    """在测试集上进行完整评估。

    加载最佳模型权重，在测试集上逐批次推理，计算 Top-1 / Top-3 准确率，
    输出 sklearn 分类报告并保存到文件。

    Returns:
        tuple: (all_labels, all_preds, all_probs, class_names)
            - all_labels (list[int]): 真实标签
            - all_preds (list[int]): 预测标签
            - all_probs (list[np.ndarray]): 每张图的各类别概率分布
            - class_names (list[str]): 类别名称列表
    """
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {BEST_MODEL_PATH}\n请先运行 python run_train.py 训练模型"
        )
    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"测试集目录不存在: {TEST_DIR}\n请确认数据集已正确放置"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        checkpoint = torch.load(
            str(BEST_MODEL_PATH), map_location=device, weights_only=False
        )
    except (pickle.UnpicklingError, EOFError, RuntimeError) as e:
        raise ValueError(
            f"模型文件损坏: {BEST_MODEL_PATH}\n请删除后重新训练"
        ) from e

    class_names = checkpoint["class_names"]

    try:
        model = create_efficientnet_b0(
            num_classes=len(class_names), pretrained=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as e:
        raise ValueError(
            f"模型权重与网络结构不匹配，请确认模型文件与当前代码版本一致\n详情: {e}"
        ) from e

    model = model.to(device)
    model.eval()

    test_dataset = datasets.ImageFolder(
        root=str(TEST_DIR), transform=get_val_transforms()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    all_preds = []
    all_labels = []
    all_probs = []

    correct = 0
    correct_top3 = 0
    total = 0

    print("在测试集上评估...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)
            _, top3_preds = torch.topk(outputs, 3, dim=1)

            correct += (preds == labels).sum().item()
            correct_top3 += (top3_preds == labels.unsqueeze(1)).any(dim=1).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    test_acc = correct / total
    test_top3_acc = correct_top3 / total

    print(f"\n{'=' * 60}")
    print(f"测试集评估结果")
    print(f"{'=' * 60}")
    print(f"Top-1 准确率: {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"Top-3 准确率: {test_top3_acc:.4f} ({test_top3_acc * 100:.2f}%)")

    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0
    )
    print(f"\n分类报告:")
    print(report)

    # 保存分类报告到文件
    report_path = FIGURE_DIR / "classification_report.txt"
    try:
        with open(str(report_path), "w", encoding="utf-8") as f:
            f.write(f"Top-1 准确率: {test_acc:.4f}\n")
            f.write(f"Top-3 准确率: {test_top3_acc:.4f}\n\n")
            f.write("分类报告:\n")
            f.write(report)
        print(f"分类报告已保存至: {report_path}")
    except OSError as e:
        print(f"[警告] 无法保存分类报告: {e}")

    return all_labels, all_preds, all_probs, class_names


def plot_confusion_matrix(all_labels, all_preds, class_names):
    """绘制混淆矩阵和每类准确率柱状图。

    生成两张图片：
        1. confusion_matrix.png — 原始计数 + 归一化混淆矩阵（1×2 子图）
        2. per_class_accuracy.png — 准确率最低/最高的 10 个类别（1×2 子图）

    Args:
        all_labels (list[int]): 真实标签
        all_preds (list[int]): 预测标签
        class_names (list[str]): 类别名称列表
    """
    cm = confusion_matrix(all_labels, all_preds)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized, nan=0.0)

    # 完整混淆矩阵（归一化）
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    sns.heatmap(cm, cmap="Blues", ax=axes[0], xticklabels=False, yticklabels=False)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"Confusion Matrix (Total: {len(all_labels)} samples)")

    sns.heatmap(cm_normalized, cmap="Blues", vmin=0, vmax=1,
                ax=axes[1], xticklabels=False, yticklabels=False)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Normalized Confusion Matrix")

    plt.tight_layout()
    save_path = FIGURE_DIR / "confusion_matrix.png"
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"混淆矩阵已保存至: {save_path}")
    plt.close(fig)

    # 每类准确率
    with np.errstate(divide="ignore", invalid="ignore"):
        class_acc = cm.diagonal() / cm.sum(axis=1)
    class_acc = np.nan_to_num(class_acc, nan=0.0)
    sorted_idx = np.argsort(class_acc)

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

    # 准确率最低的 10 个类别
    top10_idx = sorted_idx[:10]
    axes2[0].barh(range(10), class_acc[top10_idx], color="salmon")
    axes2[0].set_yticks(range(10))
    axes2[0].set_yticklabels([class_names[i] for i in top10_idx])
    axes2[0].set_xlabel("Accuracy")
    axes2[0].set_title("Bottom 10 Classes by Accuracy")
    axes2[0].set_xlim(0, 1)
    axes2[0].grid(True, alpha=0.3, axis="x")

    # 准确率最高的 10 个类别
    bottom10_idx = sorted_idx[-10:]
    axes2[1].barh(range(10), class_acc[bottom10_idx], color="steelblue")
    axes2[1].set_yticks(range(10))
    axes2[1].set_yticklabels([class_names[i] for i in bottom10_idx])
    axes2[1].set_xlabel("Accuracy")
    axes2[1].set_title("Top 10 Classes by Accuracy")
    axes2[1].set_xlim(0, 1)
    axes2[1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    save_path2 = FIGURE_DIR / "per_class_accuracy.png"
    plt.savefig(str(save_path2), dpi=150, bbox_inches="tight")
    print(f"每类准确率图已保存至: {save_path2}")
    plt.close(fig2)


def analyze_errors(all_labels, all_preds, class_names, top_k=10):
    """找出最易混淆的类别对。

    将混淆矩阵对角线置零后按错误次数降序排列，
    输出被错误分类次数最多的 top_k 个类别对。

    Args:
        all_labels (list[int]): 真实标签
        all_preds (list[int]): 预测标签
        class_names (list[str]): 类别名称列表
        top_k (int): 输出错误次数最多的前 k 个类别对，默认 10

    Returns:
        list[tuple]: [(错误次数, 真实类别索引, 预测类别索引), ...]
    """
    cm = confusion_matrix(all_labels, all_preds)
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    error_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if cm_no_diag[i, j] > 0:
                error_pairs.append((cm_no_diag[i, j], i, j))
    error_pairs.sort(reverse=True)

    print(f"\n最易混淆的 {top_k} 个类别对:")
    print("-" * 60)
    for count, true_idx, pred_idx in error_pairs[:top_k]:
        true_name = class_names[true_idx]
        pred_name = class_names[pred_idx]
        print(f"  {true_name:30s} -> {pred_name:30s} ({int(count)} 次)")

    # 保存错误分析到文件
    error_path = FIGURE_DIR / "error_analysis.txt"
    try:
        with open(str(error_path), "w", encoding="utf-8") as f:
            f.write(f"最易混淆的 {top_k} 个类别对\n")
            f.write("=" * 60 + "\n")
            for count, true_idx, pred_idx in error_pairs[:top_k]:
                true_name = class_names[true_idx]
                pred_name = class_names[pred_idx]
                f.write(f"{true_name} -> {pred_name} ({int(count)} 次)\n")
        print(f"错误分析已保存至: {error_path}")
    except OSError as e:
        print(f"[警告] 无法保存错误分析: {e}")

    return error_pairs[:top_k]


def visualize_errors(all_labels, all_preds, class_names, top_pairs=5):
    """可视化被错误分类的样本图片。

    从测试集中找出预测错误的样本，选取错误次数最多的前 top_pairs 个
    类别对，每对展示最多 3 张错误样本，生成网格图。

    Args:
        all_labels (list[int]): 真实标签
        all_preds (list[int]): 预测标签
        class_names (list[str]): 类别名称列表
        top_pairs (int): 展示前几个高混淆类别对，默认 5
    """
    # 找出错分样本的索引
    all_labels_arr = np.array(all_labels)
    all_preds_arr = np.array(all_preds)
    error_mask = all_labels_arr != all_preds_arr
    error_indices = np.where(error_mask)[0]

    if len(error_indices) == 0:
        print("没有错误分类的样本，无需可视化。")
        return

    # 统计错分类别对
    error_pairs_count = {}
    for idx in error_indices:
        key = (all_labels_arr[idx], all_preds_arr[idx])
        error_pairs_count[key] = error_pairs_count.get(key, 0) + 1

    sorted_pairs = sorted(error_pairs_count.items(), key=lambda x: x[1], reverse=True)
    selected_pairs = sorted_pairs[:top_pairs]

    # 从测试集加载图片路径
    test_dataset = datasets.ImageFolder(root=str(TEST_DIR))
    samples_per_pair = 3

    fig, axes = plt.subplots(
        len(selected_pairs), samples_per_pair,
        figsize=(4 * samples_per_pair, 3.5 * len(selected_pairs)),
    )
    if len(selected_pairs) == 1:
        axes = [axes]

    for row, ((true_cls, pred_cls), count) in enumerate(selected_pairs):
        # 找出该类别对的所有错分索引
        pair_indices = [
            idx for idx in error_indices
            if all_labels_arr[idx] == true_cls and all_preds_arr[idx] == pred_cls
        ]
        shown = pair_indices[:samples_per_pair]

        for col in range(samples_per_pair):
            ax = axes[row][col] if len(selected_pairs) > 1 else axes[col]
            if col < len(shown):
                img_path, _ = test_dataset.samples[shown[col]]
                img = Image.open(img_path)
                ax.imshow(img)
                ax.set_title(
                    f"真: {class_names[true_cls][:15]}\n"
                    f" pred: {class_names[pred_cls][:15]}",
                    fontsize=8,
                )
            else:
                ax.set_title("")
            ax.axis("off")

        # 在行首标注错误次数
        axes[row][0].set_ylabel(
            f"{row + 1}. {class_names[true_cls]}\n"
            f"  -> {class_names[pred_cls]}\n"
            f"  ({count} 次)",
            fontsize=8,
            rotation=0,
            labelpad=120,
            ha="left",
            va="center",
        )

    plt.suptitle(f"错误分类样本展示（Top {top_pairs} 混淆类别对）", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = FIGURE_DIR / "error_samples.png"
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    print(f"错误样本图已保存至: {save_path}")
    plt.close(fig)
