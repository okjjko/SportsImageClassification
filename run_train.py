"""训练入口脚本

使用方法:
    python run_train.py
"""

from src.train import train_model
from src.evaluate import (
    analyze_errors,
    evaluate_on_test,
    plot_confusion_matrix,
    plot_training_history,
)


def main():
    # Step 1: 训练
    print("=" * 70)
    print("Sports Image Classification - Training")
    print("=" * 70)

    history = train_model()

    # Step 2: 绘制训练曲线
    print("\n绘制训练曲线...")
    plot_training_history(history)

    # Step 3: 测试集评估
    print("\n在测试集上评估模型...")
    all_labels, all_preds, all_probs, class_names = evaluate_on_test()

    # Step 4: 混淆矩阵
    print("\n绘制混淆矩阵...")
    plot_confusion_matrix(all_labels, all_preds, class_names)

    # Step 5: 错误分析
    analyze_errors(all_labels, all_preds, class_names)

    print("\n全部完成!")


if __name__ == "__main__":
    main()
