# 特征工程报告

## 1. 特征提取方法

### 1.1 TF-IDF (Term Frequency-Inverse Document Frequency)

**原理**:
- 词频（TF）：某个词在文档中出现的频率
- 逆文档频率（IDF）：衡量词的普遍重要性
- 公式：TF-IDF = TF × IDF

**参数设置**:
```python
TfidfVectorizer(
    max_features=1000,      # 最大特征数
    stop_words='english',   # 去除停用词
    ngram_range=(1, 2)      # 使用 unigram 和 bigram
)
```

**优点**:
- 简单高效，易于实现
- 能捕捉词的重要性
- 适合短文本分类

**缺点**:
- 忽略词序信息
- 无法处理同义词
- 特征维度可能很高

### 1.2 其他特征方法（扩展方向）

#### Word2Vec
- 词向量表示
- 能捕捉语义关系
- 维度固定（通常 100-300 维）

#### BERT Embeddings
- 上下文相关的词向量
- 效果最好但计算成本高
- 维度较高（768 维）

## 2. 特征选择

### 2.1 最大特征数选择

| max_features | 准确率 | F1 分数 | 训练时间 |
|-------------|--------|--------|---------|
| 100         | -      | -      | -       |
| 500         | -      | -      | -       |
| 1000        | -      | -      | -       |
| 5000        | -      | -      | -       |

*注：表格数据需通过实验填充*

### 2.2 N-gram 范围

- **Unigram (1,1)**: 单个词
- **Bigram (1,2)**: 单词 + 双词组合
- **Trigram (1,3)**: 单词 + 双词 + 三词组合

**建议**: 
- 短文本使用 (1,2) 或 (1,3)
- 长文本使用 (1,1) 或 (1,2)

## 3. 特征预处理

### 3.1 停用词处理

**常见停用词**:
- the, a, an, and, or, but, in, on, at, to, for...

**影响**:
- 减少特征维度
- 提高训练速度
- 可能丢失部分信息

### 3.2 词干提取与词形还原（可选）

**词干提取 (Stemming)**:
- running → run
- better → bet

**词形还原 (Lemmatization)**:
- running → run
- better → good

## 4. 特征可视化

### 4.1 词云图
展示高频词汇

### 4.2 特征重要性
展示对分类贡献最大的词

## 5. 实验建议

1. **基线实验**: 先用默认参数建立基线
2. **参数调优**: 调整 max_features 和 ngram_range
3. **对比实验**: 对比不同特征提取方法
4. **误差分析**: 分析错误分类的样本

## 6. 代码示例

```python
from features.extraction import extract_features

# 提取特征
X_train_tfidf, X_val_tfidf, X_test_tfidf, vectorizer = extract_features(
    X_train, X_val, X_test, 
    max_features=1000
)

# 查看特征词
feature_names = vectorizer.get_feature_names_out()
print(f"前 20 个特征词：{feature_names[:20]}")
```

## 7. 总结

当前使用的 TF-IDF 方法配合 (1,2) n-gram 是一个合理的起点。后续可以尝试：
1. 增加更多特征维度
2. 引入词性标注特征
3. 尝试深度学习特征
4. 集成多种特征表示

---

**负责人**: 成员 3  
**更新日期**: 2026-04-02
