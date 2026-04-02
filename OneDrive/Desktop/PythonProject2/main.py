# 电影评论情感分析主程序
# 集成所有模块，实现完整的机器学习流程

import sys
import warnings
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

warnings.filterwarnings('ignore')

# 导入各模块
from data.loader import load_imdb_data, explore_data
from features.preprocessing import preprocess_data
from features.extraction import extract_features
from models.base_models import get_base_models
from evaluation.metrics import calculate_metrics, print_classification_report
from evaluation.visualization import (
    plot_model_comparison, 
    plot_confusion_matrix, 
    plot_roc_curves
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


class SentimentAnalyzer:
    """情感分析器类"""
    
    def __init__(self):
        self.models = {
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'SVM': SVC(probability=True, random_state=42)
        }
        self.results = {}
    
    def train_and_evaluate(self, X_train, X_val, y_train, y_val):
        """训练并评估多个模型"""
        for name, model in self.models.items():
            print(f"\n{'=' * 50}")
            print(f"训练模型：{name}")
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 在验证集上预测
            y_pred = model.predict(X_val)
            y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # 计算评估指标
            metrics = calculate_metrics(y_val, y_pred)
            
            # 保存结果
            self.results[name] = {
                'model': model,
                **metrics,
                'y_pred': y_pred,
                'y_prob': y_prob
            }
            
            print(f"准确率：{metrics['accuracy']:.4f}")
            print(f"精确率：{metrics['precision']:.4f}")
            print(f"召回率：{metrics['recall']:.4f}")
            print(f"F1 分数：{metrics['f1']:.4f}")
    
    def get_best_model(self):
        """返回最佳模型"""
        best_model_name = max(self.results, key=lambda x: self.results[x]['f1'])
        return self.results[best_model_name]['model'], best_model_name


def evaluate_on_test(best_model, X_test, y_test, model_name):
    """在测试集上评估最佳模型"""
    print(f"\n{'=' * 50}")
    print(f"在测试集上评估最佳模型：{model_name}")
    
    y_pred_test = best_model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred_test)
    
    print(f"测试集准确率：{metrics['accuracy']:.4f}")
    print(f"测试集精确率：{metrics['precision']:.4f}")
    print(f"测试集召回率：{metrics['recall']:.4f}")
    print(f"测试集 F1 分数：{metrics['f1']:.4f}")
    
    # 详细分类报告
    print("\n分类报告:")
    print_classification_report(y_test, y_pred_test, target_names=['消极', '积极'])
    
    return metrics['accuracy'], metrics['f1']


def predict_new_reviews(model, vectorizer, new_reviews):
    """对新评论进行预测"""
    print("\n新评论预测示例:")
    print("-" * 50)
    
    for i, review in enumerate(new_reviews, 1):
        # 特征提取
        features = vectorizer.transform([review])
        
        # 预测
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
        
        sentiment = "积极" if prediction == 1 else "消极"
        
        print(f"评论 {i}: {review[:50]}...")
        print(f"情感：{sentiment}")
        if probability is not None:
            print(f"置信度：积极={probability[1]:.2%}, 消极={probability[0]:.2%}")
        print("-" * 30)


def main():
    """主程序"""
    print("电影评论情感分析实验")
    print("=" * 50)
    
    # 步骤 1: 加载数据
    print("步骤 1: 加载数据...")
    
    # 选项 A: 加载少量数据用于快速测试（推荐首次运行）
    reviews, labels = load_imdb_data(sample_size=5000)
    
    # 选项 B: 加载完整数据集（需要更长时间）
    # from data.loader import load_full_imdb_dataset
    # train_reviews, train_labels, test_reviews, test_labels = load_full_imdb_dataset()
    # reviews = train_reviews + test_reviews
    # labels = train_labels + test_labels
    
    print(f"\n加载了 {len(reviews):,} 条评论")
    print(f"积极评论：{sum(labels):,} 条")
    print(f"消极评论：{len(labels) - sum(labels):,} 条")
    
    # 探索数据
    print("\n数据探索:")
    explore_data(reviews, labels)
    
    # 步骤 2: 数据预处理
    print("\n步骤 2: 数据预处理和划分...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(reviews, labels)
    
    # 步骤 3: 特征提取
    print("\n步骤 3: 特征提取...")
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = extract_features(
        X_train, X_val, X_test, max_features=500
    )
    
    # 步骤 4: 模型训练和验证
    print("\n步骤 4: 模型训练和验证...")
    analyzer = SentimentAnalyzer()
    analyzer.train_and_evaluate(X_train_tfidf, X_val_tfidf, y_train, y_val)
    
    # 步骤 5: 可视化结果
    print("\n步骤 5: 可视化结果...")
    plot_model_comparison(analyzer.results)
    
    best_model_name = max(analyzer.results, key=lambda x: analyzer.results[x]['f1'])
    plot_confusion_matrix(
        y_val, 
        analyzer.results[best_model_name]['y_pred'], 
        best_model_name
    )
    plot_roc_curves(analyzer.results, y_val)
    
    # 步骤 6: 选择最佳模型并在测试集上评估
    print("\n步骤 6: 测试集评估...")
    best_model, best_model_name = analyzer.get_best_model()
    test_accuracy, test_f1 = evaluate_on_test(
        best_model, X_test_tfidf, y_test, best_model_name
    )
    
    # 步骤 7: 新评论预测
    print("\n步骤 7: 新评论预测示例...")
    new_reviews = [
        "I really enjoyed this film, the actors were amazing!",
        "Not worth watching, complete waste of money.",
        "The plot was interesting but the ending was weak.",
        "One of the best movies I've seen this year!"
    ]
    predict_new_reviews(best_model, vectorizer, new_reviews)
    
    # 步骤 8: 总结
    print("\n" + "=" * 50)
    print("实验总结")
    print("=" * 50)
    print(f"最佳模型：{best_model_name}")
    print(f"测试集 F1 分数：{test_f1:.4f}")
    print(f"测试集准确率：{test_accuracy:.4f}")
    
    # 各模型性能比较
    print("\n各模型性能比较:")
    for name, result in analyzer.results.items():
        print(f"{name}: 准确率={result['accuracy']:.4f}, F1={result['f1']:.4f}")


if __name__ == "__main__":
    main()
