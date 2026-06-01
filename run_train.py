"""训练入口脚本

使用方法:
    python run_train.py
"""

import sys

from src.dataset import analyze_dataset, visualize_samples
from src.train import train_model
from src.evaluate import (
    analyze_errors,
    evaluate_on_test,
    plot_confusion_matrix,
    plot_training_history,
    visualize_errors,
)


def main():
    """训练完整流水线：数据分析 → 训练 → 绘制曲线 → 测试评估 → 混淆矩阵 → 错误分析。"""
    try:
        # Step 0: 数据集分析
        print("\n分析数据集...")
        analyze_dataset()
        visualize_samples()

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

        # Step 6: 错误样本可视化
        print("\n生成错误样本展示图...")
        visualize_errors(all_labels, all_preds, class_names)

        print("\n全部完成!")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n[错误] {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n[中断] 用户取消操作")
        sys.exit(0)
    except Exception as e:
        print(f"\n[错误] 程序异常退出: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
