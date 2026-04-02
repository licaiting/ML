# 可视化模块
# 负责绘制各种评估图表

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np


def plot_model_comparison(results, metrics=['accuracy', 'precision', 'recall', 'f1']):
    """
    可视化模型性能比较
    
    Args:
        results: 包含各模型结果的字典
        metrics: 要比较的指标列表
    """
    plt.figure(figsize=(12, 10))
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        model_names = list(results.keys())
        scores = [results[model][metric] for model in model_names]
        
        bars = plt.bar(model_names, scores, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title(f'{metric.capitalize()} comparision')
        plt.ylabel('score')
        plt.ylim(0, 1.1)
        
        # 在柱子上添加数值
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('evaluation/results/model_comparison.png', dpi=300)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name, class_names=None):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        model_name: 模型名称
        class_names: 类别名称列表
    """
    if class_names is None:
        class_names = ['消极', '积极']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Ground Label')
    plt.xlabel('Predicted Label')
    plt.savefig('evaluation/results/confusion_matrix.png', dpi=300)
    plt.show()


def plot_roc_curves(results, y_true):
    """
    绘制 ROC 曲线比较
    
    Args:
        results: 包含各模型结果的字典
        y_true: 真实标签
    """
    plt.figure(figsize=(8, 6))
    
    for name, result in results.items():
        if result.get('y_prob') is not None:
            fpr, tpr, _ = roc_curve(y_true, result['y_prob'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random guess')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('evaluation/results/roc_curves.png', dpi=300)
    plt.show()


def plot_training_history(history):
    """
    绘制训练历史曲线（适用于深度学习模型）
    
    Args:
        history: 训练历史记录
    """
    # TODO: 实现深度学习模型的训练历史可视化
    pass


if __name__ == "__main__":
    print("可视化模块测试")
    print("=" * 50)
    
    # 示例：创建模拟数据
    results = {
        'Model A': {'accuracy': 0.85, 'precision': 0.83, 'recall': 0.87, 'f1': 0.85},
        'Model B': {'accuracy': 0.88, 'precision': 0.86, 'recall': 0.89, 'f1': 0.87},
        'Model C': {'accuracy': 0.82, 'precision': 0.80, 'recall': 0.84, 'f1': 0.82}
    }
    
    print("\n示例：模型性能对比图")
    plot_model_comparison(results)
    
    print("\n可视化说明:")
    print("- 所有图表自动保存到 evaluation/results/ 目录")
    print("- 支持 PNG 格式，分辨率 300dpi")
