# 实验覆盖说明

这份完整源码包合并了两部分内容：

1. 基础 paper-driven 复现链：Fig.4-Fig.12
2. SOSP 扩展链：安全性、LLM 稳定性、secure tradeoff、end-to-end case study、扩展消融与对比

## 基础论文实验

- Fig.4 延迟对比：`scripts/run_benchmarks.py` + `scripts/plot_paper_figures.py`
- Fig.5 能效对比：`scripts/run_benchmarks.py` + `scripts/plot_paper_figures.py`
- Fig.6 混合任务端到端：`scripts/run_benchmarks.py` + `scripts/plot_paper_figures.py`
- Fig.7 面积/功耗：`scripts/run_benchmarks.py` + `scripts/plot_paper_figures.py`
- Fig.8 复用与路径分解：`scripts/run_benchmarks.py` + `scripts/plot_paper_figures.py`
- Fig.9 退化设计/消融：`scripts/run_benchmarks.py` + `scripts/plot_paper_figures.py`
- Fig.10 片上驻留与冗余访问：`scripts/run_benchmarks.py` + `scripts/plot_paper_figures.py`
- Fig.11 调度与切换开销：`scripts/run_benchmarks.py` + `scripts/plot_paper_figures.py`
- Fig.12 重配置延迟/能耗/状态保持：`scripts/run_benchmarks.py` + `scripts/plot_paper_figures.py`

基础结果 CSV 位于 `results/fig4_latency.csv` 到 `results/fig12_reconfig.csv`，原始 trace 与 metrics 位于 `results/raw/`。

## SOSP 扩展实验

- Fig.13 安全攻击与泄露：`scripts/run_security_eval.py` 或 `scripts/run_sosp_plus_security.py`
- Fig.14 LLM 稳定性：`scripts/run_sosp_suite.py`
- Fig.15 secure tradeoff：`scripts/run_sosp_suite.py`
- Fig.16 case study：`scripts/run_sosp_suite.py` 或 `scripts/run_sosp_plus_cases.py`
- Fig.17 扩展消融：`scripts/run_sosp_suite.py` 或 `scripts/run_sosp_plus_ablations.py`
- 额外对比、负载扫描、secure 组件拆解、dashboard 与中文报告：`scripts/run_extended_compare.py`、`scripts/run_load_sweep.py`、`scripts/run_secure_components.py`、`scripts/export_dashboard.py`、`scripts/generate_cn_report.py`

## 说明

这是按论文描述和图表目标重建的完整工程链，包含源码、脚本、绘图和结果文件。它不是作者内部私有 exact RTL/raw trace 的泄露版本，而是可运行的 paper-driven full reimplementation。
