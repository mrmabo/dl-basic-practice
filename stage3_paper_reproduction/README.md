# 第三阶段：论文复现与实验研究

该目录为未来的论文复现阶段预留。第二阶段完成后，再将每篇论文作为一个独立子目录加入。

建议结构：

```text
paper_name/
├── README.md              # 论文问题、贡献、模型结构和复现结论
├── configs/               # 可复现实验配置
├── models/                # 核心模型与模块
├── data/                  # Dataset/DataLoader（数据本身不提交）
├── train.py
├── evaluate.py
├── inference.py
├── tests/                 # shape 与核心模块测试
└── results/               # 曲线、表格和可视化
```

每次复现至少完成：baseline、核心模块、相同数据划分、实验配置、指标、训练曲线、预测可视化、与论文结果的差异分析，以及一个自己的消融或改进实验。

建议从 TCN 开始完整实现，再实现 TimesNet 或 MICN 的核心模块，最后选择其中一个做完整论文级复现。
