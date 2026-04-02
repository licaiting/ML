# 基础模型模块
# 定义多个机器学习模型

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_base_models():
    """
    获取基础模型字典
    
    Returns:
        包含多个模型的字典
    """
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    return models


def create_custom_model(model_type, **params):
    """
    创建自定义参数的模型
    
    Args:
        model_type: 模型类型 ('naive_bayes', 'logistic_regression', 'svm')
        **params: 模型参数
        
    Returns:
        配置好的模型
    """
    if model_type == 'naive_bayes':
        return MultinomialNB(**params)
    elif model_type == 'logistic_regression':
        return LogisticRegression(**params)
    elif model_type == 'svm':
        return SVC(**params)
    else:
        raise ValueError(f"未知的模型类型：{model_type}")


if __name__ == "__main__":
    # 测试模型创建
    models = get_base_models()
    
    print("可用的基础模型:")
    for name in models.keys():
        print(f"- {name}")
    
    print("\n示例：创建自定义 SVM 模型")
    custom_svm = create_custom_model('svm', C=1.0, kernel='rbf', probability=True)
    print(custom_svm)
