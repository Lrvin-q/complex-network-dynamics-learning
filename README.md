# 复杂网络动力学学习与拓扑重构

**Learning Network Dynamics and Reconstructing Network Topology from Time Series**

从节点状态时间序列出发，学习复杂网络上的局部演化规律，并在网络结构未知时联合恢复潜在连接。
本项目以网络 SIR 动力学为验证案例，在 ER 随机网络、WS 小世界网络和 BA 无标度网络上开展实验。
方法面向一般网络动力学建模；对其他理论模型和真实系统的验证是后续研究方向。

作者：Lrvin-q · [GitHub](https://github.com/Lrvin-q) · [MIT 许可证](LICENSE)

## 论文与主要材料

- **[阅读中文论文 PDF](output/pdf/复杂网络动力学学习与拓扑重构.pdf)**
- [LaTeX 主文件](paper/main.tex)与[分章节源码](paper/sections/)
- [图 1–3：黑白方法示意图 Notebook](paper/notebooks/01_method_diagrams.ipynb)
- [图 4–6：实验结果图 Notebook](paper/notebooks/02_experiment_figures.ipynb)
- [论文编辑、绘图与 PDF 构建说明](paper/README.md)

论文题目为《复杂网络动力学学习与拓扑重构：从观测时间序列到演化规律与潜在连接》。
仓库同时提供研究代码、已保存的结果 JSON、合成数据、图表与阶段报告。
旧阶段报告保留研究过程，其中部分早期解释和自回归结果已在后续修正；正式结论以论文及其引用的 phase4/phase6 结果为准。

## 研究内容

已知拓扑时，通过自身信息与邻居信息分离的双通道表示，学习局部状态转移规律，
并与不同深度、方向及聚合方式的 GraphConv 模型比较。

未知拓扑时，将可微网络生成器与双通道动力学学习器结合，采用交替优化、
硬图学习器和按感染活动分层的采样机制，减少结构与动力学参数之间的补偿，
再依据验证交叉熵进行训练随机种子间的模型选择。

在本项目保存的实验中，验证选出的 ER/WS/BA 模型的结构重构 ACC 为 0.987–1.000。
完整指标、全部九次运行及参数校准和多步预测分析见论文。
每类网络使用一个固定的 100 节点图实例，训练与验证按状态转移对随机划分；
多随机种子对应优化初始化，不能视为独立网络重复实验。

## 目录结构

```text
.
├── two_channel_experiment/        # 动力学学习、联合重构与评估脚本
│   ├── common.py                 # SIR 模拟器、模型与通用诊断
│   ├── dgsl_common.py            # 图生成器与交替训练
│   └── results/                  # phase0–phase6 的 JSON、合成数据与图表
├── paper/
│   ├── main.tex                  # 中文论文主文件
│   ├── sections/                 # 分章节正文，语义换行
│   ├── references.bib            # 顺序编码制参考文献
│   ├── notebooks/                # 六幅论文主图的唯一绘图维护入口
│   ├── figures/                  # 已导出的 PDF 矢量图与 PNG 预览
│   ├── scripts/                  # 绘图执行、数据一致性检查等
│   ├── build.ps1                 # Windows PDF 构建入口
│   └── README.md
├── output/pdf/                   # 最终论文 PDF
├── requirements.txt              # 实验环境依赖
├── requirements-paper.txt        # 论文绘图依赖
├── THIRD_PARTY_NOTICES.md
└── LICENSE
```

## 快速使用

只阅读论文和保存的结果，无须安装训练环境。
如需修改论文或重绘图表，推荐使用 Python 3.13 的独立环境：

```powershell
git clone https://github.com/Lrvin-q/complex-network-dynamics-learning.git
cd complex-network-dynamics-learning
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-paper.txt
.\.venv\Scripts\python.exe paper/scripts/verify_results.py
```

依赖版本记录自本机实际运行环境（Windows、Python 3.13.7）。
修改图表时运行 `paper/open_notebooks.cmd`，在 Notebook 的集中参数区调参并“全部运行”。
只修改正文时，双击 `paper/build_pdf.cmd` 即可生成 PDF；默认编译使用已保存的图片。

论文使用 XeLaTeX、latexmk、biber、ctex 和 `biblatex-gb7714-2015`。
当前模板使用 Windows 的宋体、微软雅黑、Times New Roman 与 Consolas，
因此原样重建 PDF 以 Windows + TeX Live 为基准。
其他系统需要调整字体配置及构建入口；不影响直接阅读已提供的 PDF。
构建脚本优先使用项目 `.venv`，也支持 `PAPER_PYTHON`、`TEXLIVE_BIN` 环境变量：

```powershell
# 可选：指定其他已有环境，用自己的实际路径替换示例
$env:PAPER_PYTHON = 'C:\path\to\python.exe'
$env:TEXLIVE_BIN = 'C:\path\to\texlive\bin\windows'
powershell -ExecutionPolicy Bypass -File .\paper\build.ps1
```

## 实验复现

训练环境安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

论文图表可直接从保存的 phase4/phase6 JSON 重绘，无须重新训练。
`paper/scripts/verify_results.py` 检查结果来源 SHA-256、关键数值、表格和选模规则。
源 JSON 按原始字节提交，避免换行转换破坏哈希校验。

重新训练已知拓扑实验时，先运行 phase3 生成 phase4 复用的权重，再运行 phase4：

```powershell
.\.venv\Scripts\python.exe -u two_channel_experiment/phase3_eval.py
.\.venv\Scripts\python.exe -u two_channel_experiment/phase4_five_models.py
```

未知拓扑最终方法入口为 `phase6_l1b.py`，三类网络各运行三个训练随机种子：

```powershell
foreach ($trainingSeed in 7, 21, 42) {
    .\.venv\Scripts\python.exe -u two_channel_experiment/phase6_l1b.py --nets ER WS BA --seed $trainingSeed
}
```

`results/phase6/data_*.npz` 包含固定图与训练/验证划分；脚本发现已有数据时会直接加载，
从而保持网络实例与数据划分一致。训练命令会写入相应结果路径，运行前请备份要保留的结果。
`phase6_eval_extended.py`、`phase6_l1b_recover.py`、`phase6_l1b_5x5.py` 等记录了补充评估流程；
论文使用的数据来源以 `paper/metrics_manifest.json` 为准。
模型检查点未提交，需要重新训练后才能执行依赖权重的补充评估；
本仓库不承诺不同硬件或库版本下重新训练得到逐位相同的结果。

## 来源与引用

联合建模思路参考 Gumbel Graph Network（GGN）；论文与上游仓库链接见
[第三方来源说明](THIRD_PARTY_NOTICES.md)。
本仓库不包含本地保存的第三方 `GGN-master` 代码副本。
论文参考文献列于 [references.bib](paper/references.bib)。

若使用本项目，请注明作者、仓库地址及所使用的提交版本。
