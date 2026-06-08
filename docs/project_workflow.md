# 项目工作流程调查报告

## 一、项目概览

基于 **PyTorch + EfficientNet-B0** 的运动图片分类系统，能识别 **100 种运动项目**。采用全网络微调（Fine-tuning）策略，提供从训练到推理的完整流水线，以及 Claude 设计风格的 Gradio Web 界面。

- **数据集来源**: Kaggle — [Sports Image Classification](https://www.kaggle.com/datasets/saurabhshahane/sports-image-classification)
- **分类目标**: 100 个运动类别
- **数据规模**: 训练集 ~13,493 张，验证集 500 张，测试集 500 张

---

## 二、项目结构

```
SportsImageClassification/
├── archive/                    # 数据集（Kaggle 下载）
│   ├── train/                  # 训练集（按类别分子文件夹）
│   ├── valid/                  # 验证集
│   └── test/                   # 测试集
├── src/
│   ├── __init__.py
│   ├── config.py               # 全局配置（路径、超参数）
│   ├── dataset.py              # 数据加载与预处理、数据集分析
│   ├── model.py                # 模型定义（EfficientNet-B0 / ResNet-50）
│   ├── train.py                # 训练与验证循环
│   ├── evaluate.py             # 评估与可视化（混淆矩阵、错误分析）
│   ├── predict.py              # 推理模块
│   └── app.py                  # Gradio Web 界面
├── outputs/
│   ├── models/                 # 模型权重（best_model.pth）
│   └── figures/                # 图表（训练曲线、混淆矩阵等）
├── public/fonts/               # 自定义字体（Inter、Source Serif 4、JetBrains Mono）
├── docs/                       # 文档
├── run_train.py                # 训练入口脚本
├── run_app.py                  # Gradio 启动入口
├── requirements.txt            # Python 依赖
└── README.md
```

---

## 三、核心工作流程（6 步）

入口文件 `run_train.py` 按顺序执行以下步骤：

```
┌─────────────────────────────────────────────────────────┐
│  Step 0: 数据集分析 (dataset.py)                         │
│    analyze_dataset()  → 统计类别分布，生成柱状图           │
│    visualize_samples() → 每类随机抽取 1 张，生成网格图      │
├─────────────────────────────────────────────────────────┤
│  Step 1: 模型训练 (train.py)                             │
│    加载数据 → 创建 EfficientNet-B0 → 15 epoch 训练循环     │
│    每个 epoch: train_one_epoch → validate → scheduler.step│
│    保存验证集最佳模型到 best_model.pth                     │
├─────────────────────────────────────────────────────────┤
│  Step 2: 训练曲线可视化 (evaluate.py)                     │
│    plot_training_history() → Loss / Accuracy / LR 曲线    │
├─────────────────────────────────────────────────────────┤
│  Step 3: 测试集评估 (evaluate.py)                         │
│    evaluate_on_test() → Top-1 / Top-3 准确率 + 分类报告    │
├─────────────────────────────────────────────────────────┤
│  Step 4: 混淆矩阵 (evaluate.py)                          │
│    plot_confusion_matrix() → 原始 + 归一化混淆矩阵         │
│                          → 准确率最高/最低 10 类别柱状图    │
├─────────────────────────────────────────────────────────┤
│  Step 5: 错误分析 (evaluate.py)                           │
│    analyze_errors()    → 最易混淆的 Top-10 类别对           │
│    visualize_errors()  → 错误样本图片展示                   │
└─────────────────────────────────────────────────────────┘
```

---

## 四、各模块详细说明

### 4.1 配置模块 — `src/config.py`

集中管理所有路径和超参数：

| 配置项 | 值 | 说明 |
|---|---|---|
| `NUM_CLASSES` | 100 | 分类类别数 |
| `BATCH_SIZE` | 32 | 批大小 |
| `NUM_EPOCHS` | 15 | 训练轮数 |
| `LEARNING_RATE` | 1e-4 | 初始学习率 |
| `IMAGE_SIZE` | 224 | 输入图片尺寸 |
| `LR_STEP_SIZE` | 5 | 学习率衰减步长（每 5 epoch） |
| `LR_GAMMA` | 0.5 | 学习率衰减因子（减半） |
| `NUM_WORKERS` | 0 | 数据加载子进程数（Windows 兼容） |

### 4.2 数据管道 — `src/dataset.py`

**训练集数据增强**（4 项）：

```
RandomResizedCrop(224, scale=(0.8, 1.0))  → 随机裁剪缩放
RandomHorizontalFlip(p=0.5)               → 水平翻转
RandomRotation(15)                        → 随机旋转 ±15°
ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
                                          → 色彩抖动
→ ToTensor()
→ Normalize(ImageNet mean/std)            → 标准归一化
```

**验证/测试集预处理**（无随机增强）：

```
Resize(256) → CenterCrop(224) → ToTensor() → Normalize(ImageNet mean/std)
```

**数据加载**: 使用 `torchvision.datasets.ImageFolder` 从目录结构自动推断类别。

**分析功能**: `analyze_dataset()` 统计类别分布并生成柱状图；`visualize_samples()` 每类随机抽取一张生成网格展示。

### 4.3 模型定义 — `src/model.py`

**主模型: EfficientNet-B0**

```
Input (3×224×224)
    │
    ▼
EfficientNet-B0 Backbone (ImageNet 预训练, 全网络微调)
    │
    ▼ 特征向量 (1280维)
Dropout(0.3)
    │
    ▼
Linear(1280 → 512)
    │
    ▼
ReLU
    │
    ▼
Dropout(0.2)
    │
    ▼
Linear(512 → 100)  ← 100 个运动类别
```

**备选模型: ResNet-50**（冻结骨干网络 + 自定义分类头，代码中已实现但未启用）

### 4.4 训练模块 — `src/train.py`

```
train_model() 完整流程:
  │
  ├── 检测设备 (CUDA / CPU)
  ├── 加载数据 (get_dataloaders)
  ├── 创建模型 (create_efficientnet_b0)
  ├── 定义损失函数 (CrossEntropyLoss)
  ├── 定义优化器 (Adam, lr=1e-4)
  ├── 定义学习率调度 (StepLR, step_size=5, gamma=0.5)
  │
  └── 训练循环 (15 epochs):
        │
        ├── train_one_epoch()  → 前向 + 反向 + 更新权重
        ├── validate()         → 无梯度评估
        ├── scheduler.step()   → 调整学习率
        │
        └── 保存最佳模型 (按 val_acc 最大化)
              │
              └── checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": ...,
                    "class_names": ...,
                    "val_acc": ...
                  }
```

**异常处理**: GPU OOM 时自动保存当前最佳模型；键盘中断 (Ctrl+C) 同样保存。

### 4.5 评估模块 — `src/evaluate.py`

提供 6 个核心功能：

| 函数 | 功能 | 输出 |
|---|---|---|
| `plot_training_history()` | 绘制训练曲线 | `training_history.png` (Loss / Acc / LR 三合一) |
| `evaluate_on_test()` | 测试集完整评估 | Top-1/Top-3 准确率 + 分类报告 |
| `plot_confusion_matrix()` | 混淆矩阵可视化 | `confusion_matrix.png` + `per_class_accuracy.png` |
| `analyze_errors()` | 错误分析 | 最易混淆的 Top-10 类别对 + `error_analysis.txt` |
| `visualize_errors()` | 错误样本展示 | `error_samples.png` |

### 4.6 推理模块 — `src/predict.py`

```
predict_image(model, image, class_names, device, top_k=3)
    │
    ├── get_val_transforms()           # 预处理
    ├── image.convert("RGB")           # 统一颜色空间
    ├── transform(image).unsqueeze(0)  # 转 tensor + 加 batch 维度
    ├── model(input)                   # 前向推理
    ├── softmax(outputs)               # 转概率
    ├── topk(probabilities, 3)         # 取 Top-3
    │
    └── 返回 [{"class": "basketball", "confidence": 0.92}, ...]
```

`load_model()` 负责从 checkpoint 加载模型，包含文件校验和结构完整性验证。

### 4.7 Web 界面 — `src/app.py`

**技术栈**: Gradio Blocks + 自定义 CSS（Claude 设计风格）

**界面布局**:

```
┌──────────────────────────────────────────────┐
│  🏅 Sports Image Classification (标题区)      │
│  基于 EfficientNet-B0 · 100类运动识别          │
├──────────────────────┬───────────────────────┤
│                      │                       │
│   上传图片区域        │    识别结果 Top-3       │
│   (拖拽/点击上传)     │    (类别 + 置信度条)    │
│                      │                       │
│   快速测试示例图片    │                       │
│   (6 个运动类别)      │                       │
│                      │                       │
├──────────────────────┴───────────────────────┤
│  底部状态栏: 模型信息 · 验证/测试准确率         │
└──────────────────────────────────────────────┘
```

**关键设计**:
- **懒加载单例**: 首次预测时才加载模型，避免启动慢
- **示例图片**: 从测试集自动选取 6 个类别（basketball、swimming、tennis 等）
- **自定义字体**: Inter / Source Serif 4 / JetBrains Mono（base64 内嵌，离线可用）
- **事件绑定**: `image_input.change` → 上传即识别，无需点击按钮

---

## 五、推理 / Web 应用流程

```
python run_app.py
       │
       ▼
  create_interface()              构建 Gradio 界面
       │
       ▼
  interface.launch(theme=Soft)    启动本地 Web 服务
       │
       ▼
  用户上传图片 ──→ predict()
       │                │
       │                ▼
       │          _get_model()  ← 懒加载单例（首次加载，后续复用）
       │                │
       │                ▼
       │          predict_image()  → 预处理 → softmax → Top-3 结果
       │                │
       ▼                ▼
  Gradio Label 组件渲染 {类别: 置信度}
```

---

## 六、关键技术参数汇总

| 维度 | 选择 | 原因 |
|---|---|---|
| 模型架构 | EfficientNet-B0 | 参数量小 (~5.3M)、精度高、推理快 |
| 微调策略 | 全网络微调 | 数据集与 ImageNet 有差异，需要适应运动场景特征 |
| 优化器 | Adam (lr=1e-4) | 自适应学习率，收敛稳定 |
| 学习率调度 | StepLR (5 epoch, ×0.5) | 简单有效，逐步精细调参 |
| 损失函数 | CrossEntropyLoss | 多分类标准选择 |
| 正则化 | Dropout(0.3+0.2) + 数据增强 | 防止过拟合 |
| 归一化 | ImageNet mean/std | 与预训练权重匹配 |
| Web 框架 | Gradio | 快速原型、支持图片交互 |

---

## 七、性能指标

| 指标 | 值 |
|---|---|
| Top-1 准确率 | ~95%+ |
| Top-3 准确率 | ~99%+ |

> 具体结果以实际训练为准。

---

## 八、项目缺失项

- 无 CI/CD 配置（无 `.github/workflows`）
- 无 Dockerfile
- 无单元测试
- 无环境变量管理（全靠 `config.py` 硬编码）
