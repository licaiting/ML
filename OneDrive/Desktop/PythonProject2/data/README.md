# 数据集使用说明

## 📦 当前使用的数据集

**名称**: Stanford IMDB Movie Reviews Dataset (aclImdb_v1)

**来源**: https://ai.stanford.edu/~amaas/data/sentiment/

**文件格式**: `aclImdb_v1.tar.gz` (约 80MB)

---

## 📊 数据集统计

| 类别 | 训练集 | 测试集 | 总计 |
|------|--------|--------|------|
| **积极评论** | 12,500 | 12,500 | 25,000 |
| **消极评论** | 12,500 | 12,500 | 25,000 |
| **总计** | 25,000 | 25,000 | **50,000** |

**特点**:
- ✅ 平衡数据集（正负样本各占 50%）
- ✅ 每条评论至少包含 100 个词
- ✅ 人工标注，质量高
- ✅ 学术界广泛使用的标准数据集

---

## 📁 目录结构

```
data/raw/aclImdb_v1/aclImdb/
├── train/              # 训练集
│   ├── pos/           # 积极评论（12,500 个.txt 文件）
│   └── neg/           # 消极评论（12,500 个.txt 文件）
├── test/              # 测试集
│   ├── pos/          # 积极评论（12,500 个.txt 文件）
│   └── neg/          # 消极评论（12,500 个.txt 文件）
├── README            # 官方说明文档
├── imdb.vocab        # 词汇表（可选使用）
└── imdbEr.txt        # 错误率信息（可选）
```

---

## 🔧 数据加载方式

### 方式一：加载部分数据（推荐用于快速测试）

```python
from data.loader import load_imdb_data

# 加载 5,000 条用于测试
reviews, labels = load_imdb_data(sample_size=5000)
```

### 方式二：加载完整数据集

```python
from data.loader import load_full_imdb_dataset

# 加载全部 50,000 条数据
train_reviews, train_labels, test_reviews, test_labels = load_full_imdb_dataset()

# 合并使用
all_reviews = train_reviews + test_reviews
all_labels = train_labels + test_labels
```

### 方式三：使用官方划分

```python
from data.loader import load_full_imdb_dataset

# 保持官方的训练/测试划分
train_reviews, train_labels, test_reviews, test_labels = load_full_imdb_dataset()

# 从训练集中再划分验证集
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    train_reviews, train_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=train_labels
)
```

---

## 🎯 推荐的数据处理流程

### 对于课程作业（快速完成）

```python
# 1. 加载 5,000-10,000 条数据用于快速实验
reviews, labels = load_imdb_data(sample_size=5000)

# 2. 数据预处理和划分
from features.preprocessing import preprocess_data
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
    reviews, labels, 
    test_size=0.2, 
    random_state=42
)

# 3. 特征提取
from features.extraction import extract_features
X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = extract_features(
    X_train, X_val, X_test, 
    max_features=5000  # 5000 个特征
)

# 4. 模型训练和评估
# ... 后续代码 ...
```

### 对于完整实验（追求更好效果）

```python
# 1. 加载完整训练集（25,000 条）
train_reviews, train_labels, test_reviews, test_labels = load_full_imdb_dataset()

# 2. 只使用训练集进行实验
from features.preprocessing import preprocess_data
X_train, X_val, y_train, y_val = preprocess_data(
    train_reviews, train_labels, 
    test_size=0.2, 
    random_state=42
)

# 3. 特征提取（使用更多特征）
from features.extraction import extract_features
X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = extract_features(
    X_train, X_val, X_test, 
    max_features=10000  # 10000 个特征
)

# 4. 在测试集上最终评估
# ... 使用 test_reviews 和 test_labels ...
```

---

## ⚙️ 参数调优建议

### 针对不同数据量的参数配置

#### 小数据集（1,000-5,000 条）
```python
max_features=1000      # 1000 个特征
ngram_range=(1, 2)     # unigram + bigram
min_df=1              # 至少出现 1 次
```

#### 中等数据集（5,000-25,000 条）
```python
max_features=5000      # 5000 个特征
ngram_range=(1, 2)     # unigram + bigram
min_df=2              # 至少出现 2 次
```

#### 大数据集（25,000+ 条）
```python
max_features=10000     # 10000 个特征
ngram_range=(1, 2)     # unigram + bigram
min_df=5              # 至少出现 5 次
max_df=0.9            # 去除 90% 以上都出现的词
```

---

## 📝 文本清洗（可选）

IMDB 数据集已经比较干净，但如果你想进一步清洗：

```python
from features.preprocessing import clean_texts

# 清洗文本
cleaned_reviews = clean_texts(reviews)

# 然后继续后续流程
X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(
    cleaned_reviews, labels, 
    test_size=0.2
)
```

**清洗步骤**：
1. 去除 HTML 标签（如 `<br />`）
2. 去除特殊字符和标点
3. 转换为小写
4. 去除多余空格

---

## 🐛 常见问题

### Q1: 数据加载失败？
**A**: 检查路径是否正确：
```python
from pathlib import Path
data_dir = Path("data/raw/aclImdb_v1/aclImdb")
print(data_dir.exists())  # 应该输出 True
```

### Q2: 内存不足怎么办？
**A**: 减少采样数量：
```python
reviews, labels = load_imdb_data(sample_size=2000)  # 只加载 2000 条
```

### Q3: 运行太慢？
**A**: 
- 减少 `max_features` 参数
- 减少 `sample_size`
- 使用更简单的 ngram_range，如 `(1,1)`

### Q4: 如何查看数据样例？
**A**:
```python
reviews, labels = load_imdb_data(sample_size=10)

for i, (review, label) in enumerate(zip(reviews, labels), 1):
    sentiment = "积极" if label == 1 else "消极"
    print(f"{i}. [{sentiment}] {review[:100]}...")
```

---

## 📊 数据统计脚本

```python
from data.loader import load_full_imdb_dataset
import numpy as np

# 加载数据
train_reviews, train_labels, test_reviews, test_labels = load_full_imdb_dataset()

# 统计分析
print("=" * 50)
print("数据集详细统计")
print("=" * 50)

print(f"\n训练集:")
print(f"  总数：{len(train_reviews):,} 条")
print(f"  积极：{sum(train_labels):,} 条 ({sum(train_labels)/len(train_labels)*100:.1f}%)")
print(f"  消极：{len(train_labels)-sum(train_labels):,} 条")

print(f"\n测试集:")
print(f"  总数：{len(test_reviews):,} 条")
print(f"  积极：{sum(test_labels):,} 条 ({sum(test_labels)/len(test_labels)*100:.1f}%)")
print(f"  消极：{len(test_labels)-sum(test_labels):,} 条")

# 评论长度分析
all_reviews = train_reviews + test_reviews
lengths = [len(review.split()) for review in all_reviews]

print(f"\n评论长度统计:")
print(f"  平均：{np.mean(lengths):.1f} 词")
print(f"  最短：{min(lengths)} 词")
print(f"  最长：{max(lengths)} 词")
print(f"  中位数：{np.median(lengths):.1f} 词")
```

---

## 🎓 参考文献

```bibtex
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L. and Daly, Raymond E. and Pham, Peter T. and Huang, Dan and Ng, Andrew Y. and Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  pages     = {142--150},
  publisher = {Association for Computational Linguistics}
}
```

---

## ✅ 检查清单

使用前请确认：
- [x] 数据集已解压到 `data/raw/aclImdb_v1/`
- [x] 文件夹结构正确（包含 train/和 test/）
- [ ] 测试加载少量数据（如 sample_size=10）
- [ ] 确认可正常运行后再加载大量数据

---

**最后更新**: 2026-04-02  
**数据集版本**: aclImdb_v1  
**适用项目**: PythonProject2 - 电影评论情感分析
