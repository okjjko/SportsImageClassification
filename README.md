# Sports Image Classification

基于 PyTorch 的运动图片分类系统，使用 EfficientNet-B0 进行全网络微调，支持 100 种运动项目的图片识别。

## 数据集

来自 Kaggle：[Sports Image Classification](https://www.kaggle.com/datasets/saurabhshahane/sports-image-classification)

- 100 个运动类别
- 训练集 ~13,493 张，验证集 500 张，测试集 500 张

## 环境配置

```bash
pip install -r requirements.txt
```

## 项目结构

```
SportsImageClassification/
├── archive/                    # 数据集
├── src/
│   ├── config.py               # 全局配置
│   ├── dataset.py              # 数据加载与预处理
│   ├── model.py                # 模型定义
│   ├── train.py                # 训练与验证循环
│   ├── evaluate.py             # 评估与可视化
│   ├── predict.py              # 推理模块
│   └── app.py                  # Gradio 界面
├── outputs/
│   ├── models/                 # 模型权重
│   └── figures/                # 图表
├── run_train.py                # 训练入口
├── run_app.py                  # Gradio 启动入口
├── requirements.txt
└── README.md
```

## 训练模型

```bash
python run_train.py
```

训练完成后，最佳模型保存在 `outputs/models/best_model.pth`，训练曲线图保存在 `outputs/figures/training_history.png`。

## 评估模型

```python
from src.evaluate import evaluate_on_test, plot_confusion_matrix, analyze_errors

labels, preds, probs, class_names = evaluate_on_test()
plot_confusion_matrix(labels, preds, class_names)
analyze_errors(labels, preds, class_names)
```

## 启动 Gradio Web 界面

```bash
python run_app.py
```

浏览器打开显示的本地地址，上传图片即可查看 Top-3 预测结果。

## 模型说明

- **架构**: EfficientNet-B0（torchvision 预训练权重）
- **策略**: 全网络微调（Fine-tuning），所有层参与训练
- **分类头**: Dropout(0.3) → Linear(1280, 512) → ReLU → Dropout(0.2) → Linear(512, 100)
- **优化器**: Adam (lr=1e-4)
- **学习率调度**: StepLR（step_size=5, gamma=0.5）
- **输入尺寸**: 224×224
- **训练轮数**: 15

## 结果

| 指标 | 值 |
|------|-----|
| Top-1 准确率 | ~95%+ |
| Top-3 准确率 | ~99%+ |

> 具体结果以实际训练为准。
