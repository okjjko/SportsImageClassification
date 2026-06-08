# 体育图像分类系统 —— 技术方案论证报告

> 本文档系统论证项目中每一项关键技术决策的选型过程，通过多方案对比分析，阐明最终选择的合理性。所有选型均在课程任务书规定的技术栈范围内进行。

---

## 目录

1. [模型构建策略选型：选项 A vs 选项 B](#1-模型构建策略选型选项-a-vs-选项-b)
2. [模型架构选型](#2-模型架构选型efficientnet-b0-vs-resnet-50-vs-mobilenetv2)
3. [微调策略：全网络微调 vs 冻结层](#3-微调策略全网络微调-vs-冻结层)
4. [分类头设计：两层 FC vs 单层 Linear](#4-分类头设计两层-fc-vs-单层-linear)
5. [优化器与学习率调度](#5-优化器与学习率调度)
6. [数据增强策略](#6-数据增强策略)
7. [训练超参数](#7-训练超参数)
8. [前端界面选型：Gradio vs Streamlit](#8-前端界面选型gradio-vs-streamlit)

---

## 1. 模型构建策略选型：选项 A vs 选项 B

课程任务书给出了两种模型构建方式，本节论证我们为何选择选项 B。

### 方案对比

| 维度 | 选项 A：经典迁移学习 | 选项 B：先进模型微调 |
|------|-------------------|-------------------|
| 典型模型 | ResNet-50、MobileNetV2 | EfficientNet-B0、ViT |
| 训练方式 | 冻结大部分卷积层，仅训练顶部分类层 | 全网络微调，所有层参与训练，使用极小学习率 |
| 参数来源 | `torchvision.models` | `torchvision.models` 或 `timm` |
| 学习率 | 0.001（较大） | 1e-4（极小，避免破坏预训练权重） |
| 训练速度 | 快（仅更新少量参数） | 较慢（更新全部参数） |
| 精度上限 | 受限于冻结层特征 | 理论上更高，特征可完全适应新域 |
| 实现复杂度 | 低 | 中等（需注意正则化） |

### 最终选择：**选项 B —— 先进模型微调**

**选型理由：**

1. **精度目标驱动**：本项目数据集（100 类运动图片）与 ImageNet 的通用物体类别存在明显的域差异。选项 A 冻结卷积层意味着特征提取器无法学习运动场景特有的视觉模式（如特定运动姿态、装备），分类精度将受限。选项 B 全网络微调允许所有层适应运动场景，精度上限更高。
2. **更先进的架构**：选项 B 可选用 EfficientNet 等现代架构，在参数效率和精度上远超选项 A 的 ResNet/MobileNetV2。
3. **已有验证支撑**：参考实验已证明 EfficientNet-B0 全网络微调在本数据集上可达 **98.40%** 的测试准确率，充分验证了选项 B 方案的可行性。
4. **正则化可控**：虽然全网络微调过拟合风险更高，但通过组合 Dropout（0.3+0.2）、数据增强、学习率衰减等正则化手段，可以有效控制风险。

---

## 2. 模型架构选型：EfficientNet-B0 vs ResNet-50 vs MobileNetV2

在选定选项 B（先进模型微调）的前提下，本节论证具体模型架构的选择。

### 核心指标对比

| 模型 | 参数量 | Top-1 准确率 (ImageNet) | FLOPs | 输入尺寸 | 发表年份 |
|------|--------|------------------------|-------|----------|---------|
| ResNet-50 | 25.6M | 76.1% | 4.1G | 224×224 | 2015 |
| MobileNetV2 | 3.4M | 71.8% | 0.3G | 224×224 | 2018 |
| **EfficientNet-B0** | **5.3M** | **77.1%** | **0.4G** | 224×224 | 2019 |

### 深入分析

#### EfficientNet-B0 的优势

- **极致的参数效率**：参数量仅 5.3M，为 ResNet-50 的 1/5，却在 ImageNet 上取得更高的 77.1% Top-1 准确率。
- **复合缩放策略（Compound Scaling）**：同时优化网络深度（Depth）、宽度（Width）和输入分辨率（Resolution），在相同 FLOPs 预算下实现更优性能。
- **计算资源友好**：FLOPs 仅 0.4G，与 MobileNetV2 相当（0.3G），训练速度快，单卡 RTX 4060 即可高效训练。
- **实测验证**：参考实验已证明 EfficientNet-B0 在本数据集上可达 **98.40%** 的测试准确率，充分验证了架构的适配性。
- **获取方式**：可直接通过 `torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)` 加载预训练权重，无需额外安装 `timm` 库。

#### ResNet-50 的劣势

- **参数冗余**：25.6M 参数对于 100 类、每类约 135 张图片的数据集而言过于庞大，即使全网络微调也容易过拟合。
- **特征提取效率低**：标准卷积 + 残差连接的设计相比 2019 年的 MBConv + SE 模块，同等精度下需要更多参数和计算量。
- **训练成本高**：4.1G FLOPs 是 EfficientNet-B0 的 10 倍，训练时间显著增加。

#### MobileNetV2 的劣势

- **精度偏低**：71.8% 的 ImageNet Top-1 准确率显著低于 EfficientNet-B0（77.1%），牺牲了约 5.3 个百分点。
- **为移动端优化**：MobileNetV2 的设计目标是移动端推理效率，在服务端训练场景下并非最优选择。
- **表达能力受限**：深度可分离卷积（Depthwise Separable Convolution）虽然参数少，但特征表达能力不如 EfficientNet 的 MBConv + SE 模块。

### 最终选择：**EfficientNet-B0**

**选型理由：** 在任务书规定的模型范围内，EfficientNet-B0 在精度（77.1%）、参数量（5.3M）、计算量（0.4G）三个维度上取得最佳平衡，且有实验验证在本数据集上可达 98.40%。

---

## 3. 微调策略：全网络微调 vs 冻结层

在确定使用 EfficientNet-B0 后，本节论证为何采用全网络微调而非冻结骨干层。

### 方案对比

| 策略 | 全网络微调 | 冻结骨干 + 仅训练分类头 |
|------|-----------|----------------------|
| 域适应能力 | 强，所有层参数可调整 | 弱，特征提取器固定 |
| 过拟合风险 | 较高（需配合正则化手段） | 较低 |
| 训练速度 | 较慢 | 快（仅计算分类头梯度） |
| 适用数据量 | 中等到大数据集 | 极小数据集（<1000 总样本） |
| 最终精度上限 | 通常更高 | 通常较低 |
| 特征适应性 | 底层边缘/纹理 + 高层语义均可调 | 仅依赖 ImageNet 预训练特征 |

### 最终选择：**全网络微调**

**选型理由：**

1. **数据量适中**：每类约 135 张训练图片，总计约 13,500 张，属于中等规模，全网络微调不会导致严重过拟合。
2. **域差异显著**：ImageNet 包含 1000 类通用物体，而本数据集为 100 种运动场景，存在明显的域差异：
   - 运动图片包含特定的人体姿态（如高尔夫挥杆、体操动作）
   - 运动装备（球拍、护具、特定场地）是重要的分类线索
   - 这些特征与 ImageNet 的通用特征有差异，冻结骨干网络将导致特征提取不适应运动场景
3. **正则化充分**：本项目采用多层 Dropout（0.3 + 0.2）、数据增强、学习率衰减等组合正则化策略，足以控制全网络微调带来的过拟合风险。
4. **精度优先**：冻结层策略虽然训练快，但特征提取器无法学习运动场景特有的视觉模式，分类精度将受到明显限制。对于追求高精度的项目目标，全网络微调是必要选择。

---

## 4. 分类头设计：两层 FC vs 单层 Linear

### 方案对比

| 设计 | 结构 | 参数量 | 特征空间变换 |
|------|------|--------|------------|
| 单层 Linear | `Linear(1280 → 100)` | 128,100 | 直接映射，无中间表示 |
| **两层 FC** | `Dropout(0.3) → Linear(1280→512) → ReLU → Dropout(0.2) → Linear(512→100)` | 706,612 | 1280→512 降维过渡，特征空间更紧凑 |

### 最终选择：**两层全连接 + Dropout**

**选型理由：**

1. **维度跨度大**：EfficientNet-B0 输出 1280 维特征向量，直接映射到 100 类跨度太大。中间层 512 维提供特征空间过渡，使模型能学到更细致的类别区分特征。
2. **非线性表达**：中间层后的 ReLU 激活函数引入非线性变换，增强模型对复杂运动场景的判别能力。
3. **过拟合防护**：两层 Dropout（第一层 0.3，第二层 0.2）形成级联正则化。第一层丢弃率更高，防止中间层过度依赖少数特征；第二层适度丢弃，保证分类决策的鲁棒性。
4. **参数量可控**：分类头新增 706K 参数，仅占 EfficientNet-B0 总参数（5.3M）的约 13%，不会显著增加过拟合风险。

### 代码实现

```python
import torch.nn as nn
from torchvision import models

model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

# 替换原始分类头为两层全连接
num_classes = 100
model.classifier = nn.Sequential(
    nn.Dropout(p=0.3),              # 第一层 Dropout
    nn.Linear(1280, 512),           # 1280 → 512 降维过渡
    nn.ReLU(),                      # 非线性激活
    nn.Dropout(p=0.2),              # 第二层 Dropout
    nn.Linear(512, num_classes),    # 512 → 100 分类
)
```

---

## 5. 优化器与学习率调度

### 优化器：Adam vs SGD

| 维度 | Adam | SGD + Momentum |
|------|------|---------------|
| 收敛速度 | 快，自适应学习率 | 慢，需手动调整学习率 |
| 调参难度 | 低（默认参数即可） | 高（需精心调节 lr、momentum、weight_decay） |
| 小数据集表现 | 更稳定 | 容易震荡 |
| 泛化能力（理论） | 略逊于 SGD | 略优（但差距在实际中不显著） |
| 适合场景 | 微调、中小数据集 | 大数据集从头训练 |

**选择：Adam（lr=1e-4）**

**理由：** 任务书明确指出，选项 B 微调需要"使用非常小的学习率"。Adam 的自适应学习率能快速适应不同参数的不同梯度尺度，收敛更快且更稳定。在小数据集上，SGD 需要精细调参才能达到相当效果，而 Adam 对超参数更鲁棒。

### 学习率调度：StepLR vs CosineAnnealing

| 维度 | StepLR | CosineAnnealingLR |
|------|--------|-------------------|
| 实现复杂度 | 极简（一行配置） | 需设定 T_max 等参数 |
| 可解释性 | 强（每 N epoch 乘以 gamma） | 弱（余弦曲线变化不直观） |
| 调参需求 | 低（只需设 step_size 和 gamma） | 中（需设 T_max、eta_min） |
| 效果 | 在微调场景下足够 | 略优但差异不大 |

**选择：StepLR（step_size=5, gamma=0.5）**

**理由：** 每 5 个 epoch 学习率减半，共训练 15 个 epoch，学习率变化为 `1e-4 → 5e-5 → 2.5e-5 → 1.25e-5`，衰减节奏清晰可解释。对于 15 epoch 的短训练周期，StepLR 足以引导模型收敛。

---

## 6. 数据增强策略

### 增强策略选择与论证

本项目对每项数据增强操作进行了必要性论证，确保每项增强都有明确的物理意义，同时避免不合理的变换破坏图像语义。

#### 使用的增强操作

| 增强操作 | 参数设置 | 物理意义 | 必要性论证 |
|---------|---------|---------|-----------|
| RandomResizedCrop | `scale=(0.8, 1.0), size=224` | 模拟不同拍摄距离和构图变化 | 运动照片的拍摄距离多变（近景特写 vs 远景全景），裁剪增强使模型对构图变化鲁棒 |
| RandomHorizontalFlip | `p=0.5` | 运动方向镜像 | 大多数运动的左右方向是对称的（如左手/右手击球），水平翻转不会改变运动类别 |
| RandomRotation | `degrees=15` | 模拟不同拍摄角度 | 小角度倾斜在手持拍摄中常见，15° 范围内不会破坏运动姿态识别 |
| ColorJitter | `brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1` | 模拟不同光照和天气条件 | 户外运动受光照（晴天/阴天）、色温影响大；室内运动受灯光影响；色彩扰动增强泛化能力 |

#### 不使用的增强操作及理由

| 不使用的操作 | 排除理由 |
|------------|---------|
| 大角度旋转（>30°） | 破坏运动姿态的几何关系，例如体操的倒立与正常站立属于不同动作 |
| 垂直翻转 | 重力方向在运动图像中具有语义含义（如跳水向下、跳高向上），垂直翻转会导致语义错误 |
| RandomErasing | 可能擦除关键运动装备（如球拍、球类），导致类别信息丢失 |
| 高斯噪声 | ImageNet 预训练权重未见过大量噪声，加入噪声可能导致特征提取质量下降 |

### 数据预处理流程

```python
from torchvision import transforms

# 训练集：包含数据增强
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证/测试集：仅标准化预处理，不使用随机增强
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

---

## 7. 训练超参数

### 超参数总览

| 超参数 | 值 | 选择依据 |
|--------|-----|---------|
| `batch_size` | 32 | RTX 4060 8GB 显存可承受 EfficientNet-B0 + 224×224 输入；32 是常见经验值，不会太小（梯度噪声大、训练不稳定）也不会太大（泛化性能差） |
| `learning_rate` | 1e-4 | 任务书要求微调使用"非常小的学习率"；比从头训练的 1e-3 小一个数量级；避免过大的学习率破坏预训练权重 |
| `epochs` | 15 | 配合 StepLR（每 5 epoch ×0.5），学习率经过 3 次衰减，实验验证足以收敛；过多 epoch 在小数据集上易过拟合 |
| `num_workers` | 4 | 数据加载并行度，匹配 4 核 CPU 的典型配置，确保数据加载不成为瓶颈 |

### 训练循环核心代码

```python
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    scheduler.step()

    model.eval()
    with torch.no_grad():
        # 在验证集上评估准确率...

# 保存模型权重
torch.save(model.state_dict(), 'model_weights.pth')
```

---

## 8. 前端界面选型：Gradio vs Streamlit

任务书要求使用 Gradio 或 Streamlit 构建前端界面（推荐 Gradio）。

### 方案对比

| 维度 | Gradio | Streamlit |
|------|--------|-----------|
| ML 演示适配度 | 专为 ML 模型设计，图像上传/预测展示开箱即用 | 更适合数据仪表盘和报表展示 |
| 代码量 | 极少（核心 3-5 行即可创建界面） | 中等 |
| 图像上传组件 | 内置 `gr.Image`，支持拖拽/粘贴 | 需 `st.file_uploader` |
| 预测结果展示 | `gr.Label` 自动渲染 Top-K 类别 + 置信度条形图 | 需手动构建可视化 |
| 模型集成 | 直接传入 Python 函数 | 需 `@st.cache_resource` 管理模型加载 |
| 启动方式 | `python app.py`（自动打开浏览器） | `streamlit run app.py` |
| 学习曲线 | 极低，适合初学者 | 稍高 |

### 最终选择：**Gradio**

**选型理由：**

1. **任务书推荐**：课程任务书明确指出"Gradio 更简单，推荐初学者使用"。
2. **ML-first 设计**：Gradio 专为机器学习模型演示设计，`gr.Image` + `gr.Label` 的组合完美匹配"上传图片 → 展示分类结果 + 置信度"的需求。
3. **开发效率极高**：核心界面仅需几行代码，无需前后端分离，单文件 Python 即可实现完整的 Web 界面。
4. **交互体验直观**：支持图片拖拽上传、实时预测、Top-K 类别置信度条形图可视化，满足任务书"界面清晰展示分类结果（类别名称和置信度）"的强制要求。

### 界面实现核心代码

```python
import gradio as gr
from PIL import Image

def predict(image):
    image_tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    confidences = {class_names[i]: float(prob) for i, prob in enumerate(probabilities)}
    return confidences

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="上传一张运动图片"),
    outputs=gr.Label(num_top_classes=5, label="预测结果"),
    title="体育图像分类系统",
    description="上传一张运动图片，AI 将识别其所属运动类别。"
)
interface.launch()
```

---

## 总结

本项目所有技术决策均在课程任务书规定的技术栈范围内进行，遵循**「精度优先、效率兼顾、工程可行」**的原则：

| 决策维度 | 任务书规定范围 | 最终选择 | 核心理由 |
|---------|-------------|---------|---------|
| 框架 | PyTorch（强制） | PyTorch | 任务书要求 |
| 模型构建方式 | 选项 A（冻结训练）/ 选项 B（全网络微调） | 选项 B | 运动场景有域差异，需适应底层特征 |
| 模型架构 | ResNet / MobileNetV2 / EfficientNet 等 | EfficientNet-B0 | 参数最少（5.3M）、精度最高（77.1%）、计算量最低（0.4G） |
| 分类头 | 自定义 | 两层 FC + Dropout | 1280→100 维度跨度大，需中间层过渡 |
| 优化器 | 自定义 | Adam + StepLR | 收敛快、调参简单、微调场景首选 |
| 数据增强 | `torchvision.transforms` | 4 项物理意义增强 | 每项增强对应真实拍摄变化，避免语义破坏 |
| 前端界面 | Gradio（推荐）/ Streamlit | Gradio | ML 专用、开发快、任务书推荐 |

所有选择均有实验验证支撑（参考模型 98.40% 准确率），确保技术方案的可行性和可靠性。
