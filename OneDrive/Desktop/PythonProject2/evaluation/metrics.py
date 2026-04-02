# 评估指标模块
# 负责计算各种性能评价指标

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)


def calculate_metrics(y_true, y_pred, average='weighted'):
    """
    计算多个评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        average: 平均方式 ('macro', 'micro', 'weighted')
        
    Returns:
        包含各指标的字典
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average),
        'recall': recall_score(y_true, y_pred, average=average),
        'f1': f1_score(y_true, y_pred, average=average)
    }
    
    return metrics


def print_classification_report(y_true, y_pred, target_names=None):
    """
    打印详细分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        target_names: 类别名称列表
    """
    print(classification_report(y_true, y_pred, target_names=target_names))


def get_confusion_matrix(y_true, y_pred):
    """
    计算混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        混淆矩阵
    """
    return confusion_matrix(y_true, y_pred)


def explain_metrics():
    """
    解释各个评估指标的含义
    """
    print("=" * 60)
    print("机器学习性能评价指标说明")
    print("=" * 60)
    
    explanations = {
        '准确率 (Accuracy)': '正确分类的样本占总样本的比例',
        '精确率 (Precision)': '预测为正的样本中，实际为正的比例',
        '召回率 (Recall)': '实际为正的样本中，被正确预测为正的比例',
        'F1 分数 (F1-Score)': '精确率和召回率的调和平均数，综合评价指标',
        '混淆矩阵 (Confusion Matrix)': '展示预测结果与真实标签的对比矩阵'
    }
    
    for metric, explanation in explanations.items():
        print(f"\n{metric}:")
        print(f"  {explanation}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 示例：计算评估指标
    y_true = [1, 0, 1, 1, 0, 1]
    y_pred = [1, 0, 0, 1, 0, 1]
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print("示例指标计算结果:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    print("\n")
    explain_metrics()
