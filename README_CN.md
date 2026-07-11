# AntennaOpt — 多端口天线优化与电流特征提取工具

## 项目概述

- **run_extraction.py** — 电流特征提取：基于 PCA + K-Means，从 CST 仿真图像中提取辐射/接地结构的二值化特征图。
- **run_optimization.py** — 多端口天线优化：GPU 加速遗传算法，联合优化 S11 参数和三个频率（1.8 / 2.3 / 2.8 GHz）的远场方向图。

## 目录结构

```
AntennaOpt/
├── run_extraction.py          # 电流特征提取入口
├── run_optimization.py        # 多端口优化入口
├── lib/                       # 共享库（3 个模块）
│   ├── cutter.py              # AntennaCutter — PCA+KMeans 图像处理
│   ├── optimizers.py          # GPU 遗传算法优化器
│   └── utils.py               # .ffs / .s1p 读写、绘图工具
├── data/                      # 输入数据（用户放入）
│   ├── curimg/                #   电流分布图像
│   ├── ffdata/                #   远场 .ffs 文件
│   └── sNpdata/               #   Touchstone .sNp / .s1p 文件
├── output/                    # 输出结果
│   ├── charadata/             #   特征提取结果 (.npy)
│   └── MultiportResult/       #   优化结果 (.npy)
├── README_CN.md
└── README_EN.md
```

## 环境依赖

```bash
pip install numpy matplotlib scikit-rf deap torch opencv-python pillow
```

## 模块 1 — 电流特征提取

```bash
python run_extraction.py
```

**流程**：PCA 降维（36 相位 → 第 1 主成分） → K-Means 二值化 → 三频率 max 合成 → 四象限对称化。

结果：`output/charadata/p_chara.npy`（辐射）、`g_chara.npy`（接地）。

## 模块 2 — 多端口天线优化

```bash
python run_optimization.py
```

**损失函数**：`cost = 0.6 × cost_S11 + 0.4 × cost_farfield`

- S11 在 1.68–3.14 GHz 内超过 -10.5 dB 的部分用 ReLU 惩罚。
- 远场在 1.8 / 2.3 / 2.8 GHz 分别计算分段加权差异后取平均。

**可调参数**（在 `run_optimization.py` 顶部）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `population_size` | 500 | 种群大小 |
| `n_generations` | 240 | 遗传代数 |
| `ifs1p` | False | True=仅 S 参数（vs 参考 .s1p） |

## 数据准备

- **data/curimg/** — 每频点 36 张相位电流图（`img_0.png` … `img_35.png`）。
- **data/ffdata/ffX.X/** — 每端口独立激励的远场 .ffs（`ff1.ffs` … `ffN.ffs`）。
- **data/ffdata/ori_X.X.ffs** — 参考天线远场。
- **data/sNpdata/multiport_sNp.s501p** — 多端口 Z 参数矩阵。
