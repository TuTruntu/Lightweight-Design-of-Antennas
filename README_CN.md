# 轻量化天线设计 — 电流特征提取与多端口天线优化

## 项目概述

- **run_curExtraction.py** — 电流特征提取：基于 PCA + K-Means，从 CST 仿真图像中提取辐射/接地结构的二值化特征图。
- **run_multiportOptimization.py** — 多端口天线优化：GPU 加速遗传算法，联合优化 S11 参数和三个频率（1.8 / 2.3 / 2.8 GHz）的远场方向图，支持实时可视化收敛过程。
- **run_ultEvaluation.py** — 优化结果评估：对比优化后 S11 与 MED.s1p 参考，并排展示优化前后 3D 天线模型。

## 目录结构

```
AntennaOpt/
├── run_curExtraction.py          # 电流特征提取入口
├── run_multiportOptimization.py  # 多端口优化入口
├── run_ultEvaluation.py          # 优化结果评估入口
├── lib/                          # 共享库（3 个模块）
│   ├── cutter.py                 # AntennaCutter — PCA+KMeans 图像处理
│   ├── optimizers.py             # GPU 遗传算法优化器
│   └── utils.py                  # .ffs / .s1p 读写、GA 实时绘图工具
├── data/                         # 输入数据（用户放入）
│   ├── curimg/                   #   电流分布图像
│   ├── ffdata/                   #   远场 .ffs 文件
│   ├── modeldata/                #   3D 天线模型 (.stl)
│   └── sNpdata/                  #   Touchstone .sNp / .s1p 文件
├── output/                       # 输出结果
│   ├── charadata/                #   特征提取结果 (.npy)
│   └── MultiportResult/          #   优化结果、收敛图、S11 对比图
├── README_CN.md
└── README_EN.md
```

## 环境依赖

```bash
pip install numpy matplotlib scikit-rf deap torch opencv-python pillow
```

可选（3D 模型查看）：
```bash
pip install pyqt6 vtk
```

## 模块 1 — 电流特征提取

```bash
python run_curExtraction.py
```

**流程**：PCA 降维（36 相位 → 第 1 主成分） → K-Means 二值化 → 三频率 max 合成 → 四象限对称化。

结果：`output/charadata/p_chara.npy`（辐射）、`g_chara.npy`（接地）。

## 模块 2 — 多端口天线优化

```bash
python run_multiportOptimization.py
```

**损失函数**：`cost = 0.6 × cost_S11 + 0.4 × cost_farfield`

- S11 在 1.8–2.8 GHz 内超过 −11 dB 的部分用 ReLU 惩罚。
- 远场在 1.8 / 2.3 / 2.8 GHz 分别计算分段加权差异后取平均。
- 每代遗传后实时更新收敛曲线（最低损失 + 平均损失），优化结束自动保存 `convergence.png`。

**可调参数**（在 `run_multiportOptimization.py` 顶部）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `population_size` | 500 | 种群大小 |
| `n_generations` | 240 | 遗传代数 |
| `ifs1p` | False | True=仅 S 参数（vs 参考 .s1p） |
| `S11_FREQ_BAND` | (1.8e9, 2.8e9) | S11 目标频段 |
| `S11_THRESHOLD` | −11.0 dB | S11 惩罚阈值 |

## 模块 3 — 优化结果评估

```bash
python run_ultEvaluation.py
```

**功能**：
- 加载优化解 `result_v1.npy`，计算优化后 S11 并与 `MED.s1p` 对比绘图（保存 `S11_comparison.png`）。
- 控制台输出目标频段 [1.8–2.8] GHz 内的 S11 统计摘要。
- 3D 并排展示优化前（`A_Original.stl`）与优化后（`D_Final.stl`）天线模型。

## 数据准备

- **data/curimg/** — 每频点 36 张相位电流图（`img_0.png` … `img_35.png`）。
- **data/ffdata/ffX.X/** — 每端口独立激励的远场 .ffs（`ff1.ffs` … `ffN.ffs`）。
- **data/ffdata/ori_X.X.ffs** — 参考天线远场。
- **data/sNpdata/multiport_sNp.s501p** — 多端口 Z 参数矩阵。
- **data/sNpdata/MED.s1p** — 参考 S11 用于对比评估。
- **data/modeldata/A_Original.stl** — 优化前天线 3D 模型。
- **data/modeldata/D_Final.stl** — 优化后天线 3D 模型。
