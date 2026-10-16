# 论文编辑与 PDF 生成

## 最常用的操作

1. **只改正文**：编辑 `sections/` 下相应 `.tex` 文件，保存后双击 `build_pdf.cmd`。
2. **调整图片**：双击 `open_notebooks.cmd`，在浏览器中打开相应 notebook，修改开头的集中参数区，选择“运行 → 运行所有单元格”，再保存 notebook。
3. **图片调整完成后**：双击 `build_pdf.cmd`，把已导出的图片编入论文。

成功时终端显示最终 PDF 的完整路径：项目根目录下 `output/pdf/复杂网络动力学学习与拓扑重构.pdf`。
失败时窗口保留错误信息，不更新交付 PDF；详细编译日志在 `build/main.log`。
若 PDF 正被阅读器锁定，请关闭该文件后重新构建。

**默认构建不执行绘图，不覆盖任何手动调整的图片。** Notebook 导出的 PDF 才是正文使用的版本；PNG 仅用于预览。
仅修改 notebook 参数但没有运行单元格，不会改变图片。

## 绘图 Notebook

- `notebooks/01_method_diagrams.ipynb`：图 1–3，黑白方法图。开头集中设置字号、画布尺寸、方块尺寸、坐标、箭头和文字换行。
- `notebooks/02_experiment_figures.ipynb`：图 4–6，结果图。用 `FONT_SCALE` 统一缩放全部文字，也可分别修改标题、坐标轴、刻度、图例和数值字号；子图间距、图例位置、柱宽、点大小以及随机种子标签偏移均在参数区。

两份 notebook 均包含完整绘图代码，是六幅主图的唯一维护入口。
`scripts/build_figures.py` 只负责用当前 Python 解释器依次执行已保存的 notebook，不再保留另一套绘图代码。
`figures/` 中保留的旧 `.tex` 示意图仅作历史参考，正文不再引用。

每份 notebook 的“全部运行”会生成该组全部 PDF/PNG 图，并显示预览。
第一次使用或重启内核后应“全部运行”；之后可先运行参数单元格，再运行需要调整的绘图函数定义与对应逐图生成单元格。
图 4–6 只读 phase4/phase6 JSON；不会训练模型或修改实验结果。

JupyterLab 启动后，保持终端窗口运行；使用结束后可在终端按 Ctrl+C 停止服务。
推荐选择 Python 3 内核。启动脚本与自动绘图执行器共用同一环境查找规则：
优先采用 `PAPER_PYTHON` 环境变量，其次为项目根目录的 `.venv\Scripts\python.exe`，
再尝试作者本机的已验证环境，最后查找 PATH 中的 Python。
新电脑可先在项目根目录建立 `.venv` 并安装 `requirements-paper.txt`。

## 命令行方式

在项目根目录执行：

```powershell
# 使用已保存图片编译正文（默认）
powershell -ExecutionPolicy Bypass -File .\paper\build.ps1

# 清理论文编译中间文件后重新编译；不改图片和 notebook
powershell -ExecutionPolicy Bypass -File .\paper\build.ps1 -Clean

# 执行两份已保存 notebook，重绘六幅图，再编译
powershell -ExecutionPolicy Bypass -File .\paper\build.ps1 -RebuildFigures
```

`-Clean` 与 `-RebuildFigures` 可以组合。其他目录启动时，将脚本路径替换为完整路径即可。
构建优先使用 `TEXLIVE_BIN` 环境变量指定的目录；未设置时，
优先采用作者本机已验证的 `D:\SelfDownloadSoftware\texlive\2025\bin\windows`，
再查找 PATH 中同时含有 XeLaTeX、latexmk 与 biber 的目录。
可用 `-PythonPath` 和 `-TexLiveBin` 参数显式指定环境；显式路径无效时构建停止并给出错误。
默认步骤为：只读数值检查 → LaTeX 编译 → 日志检查 → 更新交付 PDF。
原样重建论文需要 Windows 字体（宋体、微软雅黑、Times New Roman、Consolas）；其他系统需调整字体设置。

## 正文编辑

`main.tex` 是主文件，`sections/` 为分章节正文。
每个章节首行含编辑器根文件提示；通常一句一行，长句在自然分句处换行，段落之间保留空行。
普通源码换行不强制 PDF 换行。不要为了源码可读性额外添加 `\\`。
数值检查会忽略表格源码中的空白和换行差异。

## 数据来源与辅助文件

- 已知拓扑仅使用 phase4；未知拓扑按最低验证 CE 选取 ER 训练随机种子 42、WS 7、BA 7，图 5 保留全部九次运行。
- `metrics_manifest.json`：第二份 notebook 从原始结果计算的指标与来源哈希。
- `scripts/verify_results.py`：检查来源哈希、表格关键数值、选模规则和六幅 PDF 是否存在。
- `references.bib`：GB/T 7714 参考文献数据库。
- `build/`：编译中间文件与日志；`qa/`：全页渲染及视觉检查预览。

原始实验文件保持不变。Notebook 执行只更新 `figures/`、`metrics_manifest.json` 和 notebook 自身的运行输出。

## 本次交付验收

交付稿为 37 页 A4。两份 notebook 已分别在新内核中从上至下运行，未出现执行错误或缺字警告。
清洁构建、数据核对、字体嵌入和全页渲染检查均已完成；六幅主图 PDF 在默认构建前后 SHA-256 不变。
以后手动编辑后页数可能变化，这不属于构建错误。
