# 数据预处理模块
# 负责数据清洗、划分训练集/验证集/测试集

from sklearn.model_selection import train_test_split


def preprocess_data(texts, labels, test_size=0.2, random_state=42, clean=False):
    """
    数据预处理和划分
    
    Args:
        texts: 文本列表
        labels: 标签列表
        test_size: 测试集比例（默认 0.2）
        random_state: 随机种子
        clean: 是否清洗文本（默认 False，因为 IMDB 数据集已经比较干净）
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # 可选的文本清洗
    if clean:
        print("正在清洗文本...")
        texts = clean_texts(texts)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    # 进一步划分验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=random_state, stratify=y_train
    )
    
    print(f"训练集大小：{len(X_train)}")
    print(f"验证集大小：{len(X_val)}")
    print(f"测试集大小：{len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def clean_text(text):
    """
    清洗单个文本
    
    Args:
        text: 原始文本
        
    Returns:
        清洗后的文本
    """
    import re
    
    # 1. 去除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. 去除特殊字符和标点（保留字母和空格）
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 3. 转换为小写
    text = text.lower()
    
    # 4. 去除多余空格
    text = ' '.join(text.split())
    
    return text


def clean_texts(texts):
    """
    批量清洗文本
    
    Args:
        texts: 文本列表
        
    Returns:
        清洗后的文本列表
    """
    cleaned = []
    for i, text in enumerate(texts):
        if i % 1000 == 0:
            print(f"  正在清洗第 {i} 条文本...")
        cleaned.append(clean_text(text))
    return cleaned


if __name__ == "__main__":
    # 测试数据划分
    from data.exploration import load_imdb_data
    
    reviews, labels = load_imdb_data()
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(reviews, labels)
    
    print("\n数据集划分说明:")
    print("- 首先按 8:2 划分训练集和测试集（分层抽样）")
    print("- 然后将训练集按 3:1 划分为训练集和验证集")
    print("- 最终比例约为 6:2:2")
