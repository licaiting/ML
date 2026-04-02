# 特征提取模块
# 负责将文本转换为数值特征

from sklearn.feature_extraction.text import TfidfVectorizer


def extract_features(X_train, X_val, X_test, max_features=5000, ngram_range=(1, 2), use_idf=True):
    """
    使用 TF-IDF 提取文本特征
    
    Args:
        X_train: 训练集文本
        X_val: 验证集文本
        X_test: 测试集文本
        max_features: 最大特征数（默认 5000，适合大数据集）
        ngram_range: N-gram 范围（默认 (1,2)，即 unigram 和 bigram）
        use_idf: 是否使用 IDF 加权（默认 True）
        
    Returns:
        X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        ngram_range=ngram_range,
        use_idf=use_idf,
        min_df=2,  # 至少在 2 个文档中出现的词才保留
        max_df=0.95  # 去除在 95% 以上文档中都出现的词
    )
    
    # 在训练集上拟合
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"特征维度：{X_train_tfidf.shape}")
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer


def compare_feature_methods():
    """
    比较不同特征提取方法
    TODO: 实现以下方法的对比
    1. TF-IDF
    2. Word2Vec
    3. BERT embeddings
    """
    pass


if __name__ == "__main__":
    # 测试特征提取
    from data.exploration import load_imdb_data
    from features.preprocessing import preprocess_data
    
    reviews, labels = load_imdb_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(reviews, labels)
    
    X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = extract_features(
        X_train, X_val, X_test, max_features=500
    )
    
    print("\n特征提取说明:")
    print("- 使用 TF-IDF 方法")
    print("- 去除英文停用词")
    print("- 使用 unigram 和 bigram (1-2 个词的组合)")
    print(f"- 最多保留 500 个特征")
