# 超参数调优模块
# 负责模型超参数搜索和优化

from sklearn.model_selection import GridSearchCV
from models.base_models import get_base_models


def tune_hyperparameters(model, X_train, y_train, param_grid, cv=5):
    """
    使用网格搜索调优超参数
    
    Args:
        model: 基础模型
        X_train: 训练特征
        y_train: 训练标签
        param_grid: 参数网格
        cv: 交叉验证折数
        
    Returns:
        最佳模型和参数
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数：{grid_search.best_params_}")
    print(f"最佳 F1 分数：{grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def get_param_grids():
    """
    获取各模型的参数搜索空间
    
    Returns:
        参数字典
    """
    param_grids = {
        'Naive Bayes': {
            'alpha': [0.1, 0.5, 1.0, 2.0]
        },
        'Logistic Regression': {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        },
        'SVM': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    }
    
    return param_grids


if __name__ == "__main__":
    print("超参数调优模块")
    print("=" * 50)
    
    param_grids = get_param_grids()
    
    print("\n各模型参数搜索空间:")
    for model_name, params in param_grids.items():
        print(f"\n{model_name}:")
        for param, values in params.items():
            print(f"  {param}: {values}")
