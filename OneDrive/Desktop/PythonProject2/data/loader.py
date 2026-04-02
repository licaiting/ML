# 数据加载模块
# 负责加载和初步探索数据集

import os
import numpy as np
from pathlib import Path


def load_imdb_data(data_dir=None, sample_size=None):
    """
    加载 IMDB 电影评论数据集（Stanford aclImdb_v1）
    
    Args:
        data_dir: 数据集目录路径，默认在 data/raw/aclImdb_v1/aclImdb
        sample_size: 采样数量（可选），None 表示加载全部
        
    Returns:
        reviews (list): 评论文本列表
        labels (list): 情感标签列表（1=积极，0=消极）
    """
    if data_dir is None:
        # 默认路径
        data_dir = Path(__file__).parent / "raw" / "aclImdb_v1" / "aclImdb"
    
    # 检查目录是否存在
    if not data_dir.exists():
        print(f"错误：找不到数据集目录 {data_dir}")
        print("请确保已解压 aclImdb_v1.tar.gz 到 data/raw/ 目录")
        return [], []
    
    reviews = []
    labels = []
    
    # 只加载训练集用于演示
    print("正在加载训练集...")
    for sentiment, label in [('pos', 1), ('neg', 0)]:
        folder_path = data_dir / 'train' / sentiment
        
        if not folder_path.exists():
            print(f"警告：找不到文件夹 {folder_path}")
            continue
        
        # 读取该类别下的所有 txt 文件
        txt_files = list(folder_path.glob('*.txt'))
        
        # 计算该类别需要加载的数量
        per_class_limit = None
        if sample_size:
            per_class_limit = sample_size // 2  # 平均分配给两个类别
        
        count = 0
        for txt_file in txt_files:
            if per_class_limit and count >= per_class_limit:
                break
            
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    review = f.read().strip()
                    reviews.append(review)
                    labels.append(label)
                    count += 1
            except Exception as e:
                print(f"读取文件失败 {txt_file}: {e}")
        
        print(f"  {sentiment}类：{count} 条")
    
    total_loaded = len(reviews)
    print(f"\n✓ 成功加载 {total_loaded} 条评论")
    print(f"  积极评论：{sum(labels)} 条")
    print(f"  消极评论：{total_loaded - sum(labels)} 条")
    
    return reviews, labels


def load_full_imdb_dataset(data_dir=None):
    """
    加载完整的训练集和测试集
    
    Returns:
        train_reviews, train_labels, test_reviews, test_labels
    """
    if data_dir is None:
        data_dir = Path(__file__).parent / "raw" / "aclImdb_v1" / "aclImdb"
    
    def load_split(split_name):
        reviews = []
        labels = []
        
        for sentiment, label in [('pos', 1), ('neg', 0)]:
            folder_path = data_dir / split_name / sentiment
            
            if not folder_path.exists():
                continue
            
            txt_files = list(folder_path.glob('*.txt'))
            
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        review = f.read().strip()
                        reviews.append(review)
                        labels.append(label)
                except Exception as e:
                    continue
        
        return reviews, labels
    
    print("加载训练集...")
    train_reviews, train_labels = load_split('train')
    
    print("加载测试集...")
    test_reviews, test_labels = load_split('test')
    
    print(f"\n✓ 训练集：{len(train_reviews)} 条")
    print(f"✓ 测试集：{len(test_reviews)} 条")
    
    return train_reviews, train_labels, test_reviews, test_labels


def load_data_from_file(file_path, encoding='utf-8'):
    """
    从 CSV 或其他格式文件加载数据
    
    Args:
        file_path: 数据文件路径
        encoding: 文件编码
        
    Returns:
        reviews, labels
    """
    import pandas as pd
    
    if not Path(file_path).exists():
        print(f"错误：找不到文件 {file_path}")
        return [], []
    
    # 根据文件扩展名选择读取方式
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, encoding=encoding)
        # 假设列名为 'review' 和 'label' 或 'sentiment'
        if 'review' in df.columns:
            reviews = df['review'].fillna('').tolist()
        elif 'text' in df.columns:
            reviews = df['text'].fillna('').tolist()
        else:
            reviews = df.iloc[:, 0].fillna('').tolist()
        
        if 'label' in df.columns:
            labels = df['label'].tolist()
        elif 'sentiment' in df.columns:
            labels = df['sentiment'].map({'positive': 1, 'negative': 0}).tolist()
        else:
            labels = df.iloc[:, 1].tolist()
        
        return reviews, labels
    else:
        print("不支持的文件格式，请使用 CSV 格式")
        return [], []


def explore_data(reviews, labels):
    """
    数据探索性分析
    
    Args:
        reviews: 评论列表
        labels: 标签列表
    """
    if not reviews:
        print("没有数据可分析")
        return
    
    print("\n" + "="*50)
    print("IMDB 数据探索分析")
    print("="*50)
    
    print(f"\n总样本数：{len(reviews):,} 条")
    print(f"积极评论数：{sum(labels):,} 条 ({sum(labels)/len(labels)*100:.1f}%)")
    print(f"消极评论数：{len(labels) - sum(labels):,} 条 ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")
    
    # 评论长度统计
    lengths = [len(review.split()) for review in reviews]
    print(f"\n平均评论长度：{np.mean(lengths):.1f} 词")
    print(f"最短评论：{min(lengths)} 词")
    print(f"最长评论：{max(lengths)} 词")
    
    # 显示前 5 个样本
    print("\n前 5 个样本示例:")
    for i in range(min(5, len(reviews))):
        sentiment = "积极" if labels[i] == 1 else "消极"
        preview = reviews[i][:80].replace('\n', ' ')
        print(f"{i+1}. [{sentiment}] {preview}...")
    
    # 数据分布建议
    print("\n" + "="*50)
    print("数据集划分建议")
    print("="*50)
    print("推荐划分比例：")
    print("  - 训练集：60% (约 15,000 条)")
    print("  - 验证集：20% (约 5,000 条)")
    print("  - 测试集：20% (约 5,000 条)")
    print("\n或者直接使用官方划分：")
    print("  - 训练集：25,000 条")
    print("  - 测试集：25,000 条")


if __name__ == "__main__":
    # 测试数据加载
    print("开始加载 IMDB 数据集...")
    reviews, labels = load_imdb_data(sample_size=100)  # 先测试加载 100 条
    
    if reviews:
        explore_data(reviews, labels)
