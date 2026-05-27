"""评估与可视化模块"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from .config import BATCH_SIZE, BEST_MODEL_PATH, FIGURE_DIR, MATPLOTLIB_FONT, NUM_WORKERS, TEST_DIR
from .dataset import get_val_transforms
from .model import create_efficientnet_b0

# 设置中文字体
plt.rcParams["font.sans-serif"] = [MATPLOTLIB_FONT, "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def plot_training_history(history, save_path=None):
    """绘制训练曲线（loss、acc、lr）"""
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
    """在测试集上进行完整评估

    Returns:
        all_labels, all_preds, all_probs, class_names
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(str(BEST_MODEL_PATH), map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]

    model = create_efficientnet_b0(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    test_dataset = datasets.ImageFolder(root=str(TEST_DIR), transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

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

    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(f"\n分类报告:")
    print(report)

    # 保存分类报告到文件
    report_path = FIGURE_DIR / "classification_report.txt"
    with open(str(report_path), "w", encoding="utf-8") as f:
        f.write(f"Top-1 准确率: {test_acc:.4f}\n")
        f.write(f"Top-3 准确率: {test_top3_acc:.4f}\n\n")
        f.write("分类报告:\n")
        f.write(report)
    print(f"分类报告已保存至: {report_path}")

    return all_labels, all_preds, all_probs, class_names


def plot_confusion_matrix(all_labels, all_preds, class_names):
    """绘制混淆矩阵和每类准确率"""
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

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
    class_acc = cm.diagonal() / cm.sum(axis=1)
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
    """找出最易混淆的类别对"""
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

    return error_pairs[:top_k]
