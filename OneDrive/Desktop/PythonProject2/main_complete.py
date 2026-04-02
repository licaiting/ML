# 电影评论情感分析 - 完整版（包含所有改进功能）
# 集成：5 折交叉验证 + 集成学习 + 超参数调优

import sys
import warnings
from pathlib import Path
import time

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings('ignore')

# 导入必要的库
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, 
    VotingClassifier, 
    StackingClassifier
)
from sklearn.model_selection import cross_validate, GridSearchCV


def get_base_models():
    """获取基础模型"""
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    return models


def get_ensemble_models():
    """获取集成学习模型"""
    ensemble_models = {}
    
    # 1. Voting Classifier
    estimators = [
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('nb', MultinomialNB()),
        ('svm', SVC(probability=True, random_state=42))
    ]
    
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting='soft',
        n_jobs=-1
    )
    ensemble_models['Voting Classifier'] = voting_clf
    
    # 2. Stacking Classifier
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1
    )
    ensemble_models['Stacking Classifier'] = stacking_clf
    
    return ensemble_models


def cross_validation_evaluate(model, X_train, y_train, model_name, cv=5):
    """5 折交叉验证评估"""
    print(f"\n{'='*60}")
    print(f"{model_name} - {cv}折交叉验证")
    print(f"{'='*60}")
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0
    )
    
    results = {}
    for metric in scoring.keys():
        mean_score = scores[f'test_{metric}'].mean()
        std_score = scores[f'test_{metric}'].std()
        results[metric] = {'mean': mean_score, 'std': std_score}
        print(f"{metric:12s}: {mean_score:.4f} (±{std_score:.4f})")
    
    return results


def tune_hyperparameters(X_train, y_train, model_name="Model"):
    """网格搜索超参数调优"""
    print(f"\n{'='*60}")
    print(f"{model_name} - 超参数调优")
    print(f"{'='*60}")
    
    # 定义各模型的参数网格
    param_grids = {
        'Naive Bayes': {'alpha': [0.1, 0.5, 1.0, 2.0], 'fit_prior': [True, False]},
        'Logistic Regression': {'C': [0.1, 1.0, 10.0], 'penalty': ['l2'], 'solver': ['lbfgs']},
        'SVM': {'C': [0.1, 1.0, 10.0], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
        'Random Forest': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    }
    
    if model_name not in param_grids:
        print(f"警告：{model_name} 没有定义参数网格，使用默认参数")
        return None, None, None
    
    # 创建基础模型
    if model_name == 'Naive Bayes':
        model = MultinomialNB()
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(max_iter=2000, random_state=42)
    elif model_name == 'SVM':
        model = SVC(probability=True, random_state=42)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # 网格搜索
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[model_name],
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=0
    )
    
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    
    print(f"✓ 调优完成！用时：{end_time - start_time:.2f}秒")
    print(f"最佳参数：{grid_search.best_params_}")
    print(f"最佳 F1 分数：{grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def calculate_metrics(y_true, y_pred):
    """计算评估指标"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted')
    }


def plot_results(results, use_cv=False, use_tuning=False):
    """绘制结果对比图"""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # 分离不同类型模型
        baseline_models = ['Naive Bayes', 'Logistic Regression', 'SVM', 'Random Forest']
        
        # 准备数据
        names = []
        f1_scores = []
        colors = []
        
        for name, result in results.items():
            names.append(name)
            
            if use_cv and 'cv_results' in result:
                f1_scores.append(result['cv_results']['f1']['mean'])
            else:
                f1_scores.append(result.get('f1', 0))
            
            # 设置颜色
            if '(Tuned)' in name:
                colors.append('lightgreen')
            elif 'Classifier' in name:
                colors.append('gold')
            else:
                colors.append('skyblue')
        
        # 绘图
        fig, ax = plt.subplots(figsize=(14, 7))
        x_pos = np.arange(len(names))
        bars = ax.bar(x_pos, f1_scores, color=colors)
        
        ax.set_xlabel('model', fontsize=12)
        ax.set_ylabel('F1 score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=15, ha='right')
        ax.set_ylim(0, 1.1)
        
        # 添加数值标签
        for bar, score in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('evaluation/results/model_comparison_complete.png', dpi=300)
        plt.show()
        
        print("\n✓ 结果对比图已保存到 evaluation/results/model_comparison_complete.png")
        
    except Exception as e:
        print(f"\n警告：无法绘制图表 - {e}")


def main():
    """主程序 - 交互式菜单"""
    print("="*60)
    print("电影评论情感分析 - 完整版")
    print("="*60)
    print("\n本程序包含以下改进功能:")
    print("1. ✓ 5 折交叉验证")
    print("2. ✓ 集成学习（随机森林、Voting、Stacking）")
    print("3. ✓ 超参数网格搜索调优")
    print("="*60)
    
    # ========== 配置选项 ==========
    print("\n【配置选项】")
    print("-"*60)
    
    # 选项 1: 数据量
    print("\n1. 选择数据量:")
    print("   [1] 5,000 条（快速测试，推荐首次运行）")
    print("   [2] 10,000 条（标准实验）")
    print("   [3] 25,000 条（完整实验，耗时较长）")
    choice_data = input("   请选择 (1/2/3) [默认 1]: ").strip() or "1"
    
    sample_sizes = {'1': 5000, '2': 10000, '3': 25000}
    sample_size = sample_sizes.get(choice_data, 5000)
    
    # 选项 2: 是否使用集成学习
    print("\n2. 是否包含集成学习模型？")
    print("   - 随机森林、Voting 分类器、Stacking 分类器")
    use_ensemble = input("   输入 y 启用 [默认 n]: ").strip().lower() == 'y'
    
    # 选项 3: 是否使用交叉验证
    print("\n3. 是否使用 5 折交叉验证？")
    print("   - 更可靠的性能评估，但会增加运行时间")
    use_cv = input("   输入 y 启用 [默认 y]: ").strip().lower() != 'n'
    
    # 选项 4: 是否进行超参数调优
    print("\n4. 是否进行超参数网格搜索调优？")
    print("   - 为每个模型寻找最优参数（预计需要 5-10 分钟）")
    use_tuning = input("   输入 y 启用 [默认 n]: ").strip().lower() == 'y'
    
    # ========== 开始执行 ==========
    print("\n" + "="*60)
    print("开始执行...")
    print("="*60)
    
    # 步骤 1: 加载数据
    print("\n【步骤 1】加载数据...")
    from data.loader import load_imdb_data, explore_data
    
    reviews, labels = load_imdb_data(sample_size=sample_size)
    
    print(f"\n✓ 加载了 {len(reviews):,} 条评论")
    print(f"  积极评论：{sum(labels):,} 条")
    print(f"  消极评论：{len(labels) - sum(labels):,} 条")
    
    # 探索数据
    print("\n数据探索:")
    explore_data(reviews, labels)
    
    # 步骤 2: 数据预处理
    print("\n【步骤 2】数据预处理和划分...")
    from features.preprocessing import preprocess_data
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(reviews, labels)
    
    # 步骤 3: 特征提取
    print("\n【步骤 3】特征提取...")
    from features.extraction import extract_features
    
    # 根据数据量调整特征数
    max_features = min(5000, sample_size // 10)
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = extract_features(
        X_train, X_val, X_test, max_features=max_features
    )
    
    # 步骤 4: 模型训练
    print("\n【步骤 4】模型训练和评估...")
    
    # 准备模型字典
    all_models = get_base_models()
    
    if use_ensemble:
        print("\n✓ 加载集成学习模型...")
        ensemble_models = get_ensemble_models()
        all_models.update(ensemble_models)
    
    # 存储结果
    results = {}
    
    # 训练每个模型
    for name, model in all_models.items():
        print(f"\n{'-'*60}")
        print(f"训练模型：{name}")
        print(f"{'-'*60}")
        
        start_time = time.time()
        
        # 训练模型
        model.fit(X_train_tfidf, y_train)
        
        train_time = time.time() - start_time
        print(f"✓ 训练完成！用时：{train_time:.2f}秒")
        
        # 在验证集上预测
        y_pred = model.predict(X_val_tfidf)
        y_prob = model.predict_proba(X_val_tfidf)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # 计算指标
        metrics = calculate_metrics(y_val, y_pred)
        
        # 保存结果
        results[name] = {
            'model': model,
            **metrics,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'train_time': train_time
        }
        
        print(f"准确率：{metrics['accuracy']:.4f}")
        print(f"精确率：{metrics['precision']:.4f}")
        print(f"召回率：{metrics['recall']:.4f}")
        print(f"F1 分数：{metrics['f1']:.4f}")
        
        # 如果启用交叉验证
        if use_cv:
            cv_results = cross_validation_evaluate(
                model, X_train_tfidf, y_train, name, cv=5
            )
            results[name]['cv_results'] = cv_results
    
    # 步骤 5: 超参数调优（如果启用）
    tuned_models = {}
    if use_tuning:
        print("\n" + "="*60)
        print("【步骤 5】超参数网格搜索调优...")
        print("="*60)
        print("提示：这可能需要 5-10 分钟，请耐心等待\n")
        
        # 只对基础模型进行调优
        base_model_names = ['Naive Bayes', 'Logistic Regression', 'SVM', 'Random Forest']
        
        for model_name in base_model_names:
            if model_name in all_models:
                best_model, best_params, best_score = tune_hyperparameters(
                    X_train_tfidf, y_train, model_name
                )
                
                if best_model is not None:
                    tuned_name = f"{model_name} (Tuned)"
                    
                    # 在验证集上评估
                    y_pred = best_model.predict(X_val_tfidf)
                    metrics = calculate_metrics(y_val, y_pred)
                    
                    # 如果需要交叉验证
                    if use_cv:
                        cv_results = cross_validation_evaluate(
                            best_model, X_train_tfidf, y_train, tuned_name, cv=5
                        )
                        results[tuned_name] = {
                            'model': best_model,
                            **metrics,
                            'cv_results': cv_results,
                            'params': best_params,
                            'tuned_f1': best_score
                        }
                    else:
                        results[tuned_name] = {
                            'model': best_model,
                            **metrics,
                            'params': best_params,
                            'tuned_f1': best_score
                        }
                    
                    print(f"\n✓ {tuned_name}:")
                    print(f"  最佳参数：{best_params}")
                    print(f"  验证集 F1: {metrics['f1']:.4f}")
        
        tuned_models = {k: v for k, v in results.items() if '(Tuned)' in k}
    
    # 步骤 6: 可视化结果
    print("\n【步骤 6】可视化结果...")
    
    # 确保目录存在
    import os
    os.makedirs('evaluation/results', exist_ok=True)
    
    plot_results(results, use_cv=use_cv, use_tuning=use_tuning)
    
    # 步骤 7: 混淆矩阵和 ROC 曲线
    print("\n生成混淆矩阵和 ROC 曲线...")
    try:
        from evaluation.visualization import plot_confusion_matrix, plot_roc_curves
        
        # 找出最佳模型
        best_model_name = max(results, key=lambda x: results[x].get('f1', 0))
        best_result = results[best_model_name]
        
        # 混淆矩阵
        plot_confusion_matrix(y_val, best_result['y_pred'], best_model_name)
        
        # ROC 曲线
        plot_roc_curves(results, y_val)
        
    except Exception as e:
        print(f"警告：无法生成详细图表 - {e}")
    
    # 步骤 8: 测试集评估
    print("\n【步骤 7】测试集最终评估...")
    print("="*60)
    
    # 找出最佳模型
    if use_cv:
        # 使用交叉验证的 F1 均值
        best_model_name = max(
            results, 
            key=lambda x: results[x].get('cv_results', {}).get('f1', {}).get('mean', 0)
        )
    else:
        # 使用验证集 F1
        best_model_name = max(results, key=lambda x: results[x].get('f1', 0))
    
    best_model = results[best_model_name]['model']
    
    # 在测试集上评估
    y_pred_test = best_model.predict(X_test_tfidf)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    print(f"\n最佳模型：{best_model_name}")
    print(f"测试集准确率：{test_metrics['accuracy']:.4f}")
    print(f"测试集 F1 分数：{test_metrics['f1']:.4f}")
    
    # 步骤 9: 总结报告
    print("\n" + "="*60)
    print("【最终总结报告】")
    print("="*60)
    
    print(f"\n配置信息:")
    print(f"  数据量：{sample_size:,} 条")
    print(f"  特征维度：{X_train_tfidf.shape[1]}")
    print(f"  集成学习：{'是' if use_ensemble else '否'}")
    print(f"  交叉验证：{'是' if use_cv else '否'}")
    print(f"  超参数调优：{'是' if use_tuning else '否'}")
    
    print(f"\n最佳模型：{best_model_name}")
    if use_tuning and '(Tuned)' in best_model_name:
        print(f"最佳参数：{results[best_model_name].get('params', 'N/A')}")
    
    print(f"\n测试集性能:")
    print(f"  准确率：{test_metrics['accuracy']:.4f}")
    print(f"  F1 分数：{test_metrics['f1']:.4f}")
    
    print(f"\n所有模型性能排名（按验证集 F1）:")
    print("-"*60)
    
    # 排序并显示
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].get('cv_results', {}).get('f1', {}).get('mean', x[1].get('f1', 0)),
        reverse=True
    )
    
    for i, (name, result) in enumerate(sorted_models[:10], 1):
        if use_cv and 'cv_results' in result:
            f1_mean = result['cv_results']['f1']['mean']
            f1_std = result['cv_results']['f1']['std']
            f1_str = f"{f1_mean:.4f}(±{f1_std:.4f})"
        else:
            f1_str = f"{result.get('f1', 0):.4f}"
        
        marker = " ★" if name == best_model_name else ""
        print(f"{i:2d}. {name:30s}: F1={f1_str}{marker}")
    
    # 保存结果摘要
    print("\n" + "="*60)
    print("提示：")
    print(f"  - 结果图已保存到 evaluation/results/")
    print(f"  - 可以运行 python main.py 使用简化版本")
    print("="*60)
    
    return results, best_model, best_model_name


if __name__ == "__main__":
    try:
        results, best_model, best_model_name = main()
    except KeyboardInterrupt:
        print("\n\n程序中断。")
    except Exception as e:
        print(f"\n发生错误：{e}")
        import traceback
        traceback.print_exc()
