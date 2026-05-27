"""训练与验证循环模块"""

import copy
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from .config import (
    BATCH_SIZE, BEST_MODEL_PATH, LEARNING_RATE,
    LR_GAMMA, LR_STEP_SIZE, NUM_EPOCHS, NUM_WORKERS,
)
from .dataset import get_dataloaders
from .model import create_efficientnet_b0, count_parameters


def train_one_epoch(model, loader, criterion, optimizer, device):
    """单个 epoch 训练

    Returns:
        avg_loss, accuracy (percentage)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="  Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.0 * correct / total:.1f}%")

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """验证集评估

    Returns:
        avg_loss, accuracy (percentage)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train_model():
    """完整训练流水线：加载数据 → 创建模型 → 训练 → 保存最佳模型

    Returns:
        history: dict with train_loss, train_acc, val_loss, val_acc, lr
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_loader, val_loader, _, class_names = get_dataloaders()
    print(f"类别数: {len(class_names)}")
    print(f"训练集: {len(train_loader.dataset)} 张")
    print(f"验证集: {len(val_loader.dataset)} 张")

    model = create_efficientnet_b0(num_classes=len(class_names))
    model = model.to(device)
    count_parameters(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "lr": [],
    }
    best_val_acc = 0.0
    best_model_state = None

    print(f"\n开始训练: {NUM_EPOCHS} epochs, lr={LEARNING_RATE}")
    print(f"学习率调度: StepLR(step_size={LR_STEP_SIZE}, gamma={LR_GAMMA})")
    print("-" * 70)

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch}/{NUM_EPOCHS}]  lr={current_lr:.6f}")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.2f}%")
        print(f"  Val   Loss: {val_loss:.4f}  Val   Acc: {val_acc:.2f}%")
        print(f"  Time: {elapsed:.1f}s")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": best_model_state,
                "class_names": class_names,
                "val_acc": best_val_acc,
            }
            torch.save(checkpoint, str(BEST_MODEL_PATH))
            print(f"  ** Best model saved (val_acc={val_acc:.2f}%)")

        print("-" * 70)

    # 加载最佳模型权重
    model.load_state_dict(best_model_state)
    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    print(f"模型已保存到: {BEST_MODEL_PATH}")

    return history
