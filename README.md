# 基于机器学习的供水管网漏损检测
# ML-based Leak Detection in Water Distribution Networks

> 数据集：[BattLeDIM / L-Town](https://github.com/KIOS-Research/BattLeDIM)（KIOS 研究中心发布的世界银行级基准：
> 782 条管道的真实感供水管网，SCADA 流量/压力监测数据，含带标准答案的漏损事件）

## 项目简介

供水管网的漏损（爆管前的暗漏）是水务公司产销差损失的主要来源之一。
本项目使用 BattLeDIM 基准数据集，构建从 SCADA 监测数据到漏损识别的完整机器学习流水线：

- 数据解析与探索（EDA）：真实感数据的缺失段、传感器异常处理
- 特征工程：夜间最小流量、滚动偏差、压力-流量残差等漏损检测经典特征
- 建模：阈值报警基线 → 随机森林 / XGBoost 漏损分类
- 评估：对照数据集自带的标准答案，量化检出率与误报率

## 技术栈

Python（Pandas / NumPy / scikit-learn / Matplotlib）· SQLite · MySQL · WSL2 (Ubuntu) · Git

## 目录结构

```
├── README.md            项目说明（本文件）
├── .gitignore           数据集大文件不入库
├── data/                数据集（本地下载，不入库）
├── notebooks/           探索性分析
├── src/                 分析与建模脚本
└── 出图/                图表产出
```

## 进度

- [ ] 数据集下载与解析
- [ ] EDA：传感器全景、缺失与异常盘点
- [ ] 特征工程
- [ ] 基线模型（阈值报警）
- [ ] 机器学习模型与评估
- [ ] 项目总结

> 本仓库是给排水工程背景 + 数据分析的学习实践项目，分析代码与图表全部开源。
