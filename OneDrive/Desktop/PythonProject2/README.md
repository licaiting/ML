# 电影评论情感分析系统

## 项目简介

本项目实现了一个完整的电影评论情感分析系统，使用机器学习方法对电影评论进行二分类（积极/消极）。

## 项目结构

```
PythonProject2/
├── data/                     # 数据组
│   ├── raw/                  # 原始数据
│   ├── processed/            # 清洗后数据
│   └── exploration.ipynb     # 数据分析笔记
├── features/                 # 特征工程组
│   ├── preprocessing.py      # 文本预处理
│   ├── extraction.py         # 特征提取
│   └── features_report.md    # 特征工程报告
├── models/                   # 模型组
│   ├── base_models.py        # 基础模型
│   ├── hyperparameter_tuning.py  # 超参数调优
│   └── model_comparison.ipynb    # 模型对比笔记
├── evaluation/               # 评估组
│   ├── metrics.py           # 评估指标
│   ├── visualization.py     # 可视化工具
│   └── results/             # 结果图表
├── main.py                   # 主程序
├── requirements.txt          # 依赖包
└── README.md                # 项目说明
```

## 小组成员分工

### 成员 1：数据收集与加载
- **负责模块**: `data/exploration.ipynb`
- **主要任务**:
  - 数据集选择与调研（IMDB、SST-2 等）
  - 数据加载函数实现
  - 数据探索性分析
  - 数据标注形式说明

### 成员 2：数据预处理
- **负责模块**: `features/preprocessing.py`
- **主要任务**:
  - 数据清洗逻辑
  - 训练集/验证集/测试集划分
  - 分层抽样实现
  - 数据集划分合理性分析

### 成员 3：特征工程
- **负责模块**: `features/extraction.py`
- **主要任务**:
  - TF-IDF 特征提取
  - 特征参数调优
  - 不同特征方法对比（可选扩展）
  - 特征维度分析

### 成员 4：模型训练与调优
- **负责模块**: `models/base_models.py`, `models/hyperparameter_tuning.py`
- **主要任务**:
  - 基础模型实现（朴素贝叶斯、逻辑回归、SVM）
  - 超参数调优
  - 模型对比实验

### 成员 5：性能评估
- **负责模块**: `evaluation/metrics.py`, `evaluation/visualization.py`
- **主要任务**:
  - 评估指标计算（准确率、精确率、召回率、F1 值）
  - 可视化实现（混淆矩阵、ROC 曲线、性能对比图）
  - 各指标含义解释

### 成员 6：系统集成与测试
- **负责模块**: `main.py`
- **主要任务**:
  - 集成所有模块
  - 完整流程实现
  - 测试集最终评估
  - 新评论预测演示
  - 实验报告汇总

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行项目

```bash
python main.py
```

## 性能评价指标

本系统使用以下指标评估模型性能：

- **准确率 (Accuracy)**: 正确分类的样本占总样本的比例
- **精确率 (Precision)**: 预测为正的样本中，实际为正的比例
- **召回率 (Recall)**: 实际为正的样本中，被正确预测为正的比例
- **F1 分数 (F1-Score)**: 精确率和召回率的调和平均数

## 数据集划分

采用分层抽样方式：
- 训练集：60%
- 验证集：20%
- 测试集：20%

## 实验结果

运行主程序后，自动生成以下可视化结果：
- `evaluation/results/model_comparison.png` - 模型性能对比图
- `evaluation/results/confusion_matrix.png` - 最佳模型混淆矩阵
- `evaluation/results/roc_curves.png` - ROC 曲线对比

## 扩展方向

1. 尝试深度学习方法（LSTM、BERT 等）
2. 增加更多特征提取方法（Word2Vec、GloVe 等）
3. 实现更复杂的超参数搜索（随机搜索、贝叶斯优化等）
4. 部署为 Web 应用或 API 服务

## 许可证

MIT License
