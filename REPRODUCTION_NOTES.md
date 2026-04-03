# Reproduction Notes

本工程在前一版复现基础上继续升级，目标是把论文方法链扩展成更接近系统论文提交的工程形态。

## 本次升级

- 新增 LLM prompt-to-IR 前端
- 新增 Canonical IR 与 verifier
- 新增 secure-ct 与 secure-randomized 模式
- 新增 Transformer / hybrid / RAG 类 workload
- 新增安全评测、攻击评测、prompt stability 评测
- 新增 case study、扩展消融、扩展对比
- 新增 Fig.13–Fig.17 与 Table 1 风格结果表

## 结果文件

- `results/fig13_security.csv`
- `results/fig14_llm_stability.csv`
- `results/fig15_secure_tradeoff.csv`
- `results/fig16_case_studies.csv`
- `results/fig17_ablation_plus.csv`
- `results/table1_sosp_summary.csv`
- `results/raw/fig13_security_trials.csv`
- `results/raw/fig16_case_latencies.csv`

## 关键命令

```bash
python scripts/build_cpp.py
python scripts/train_calibration.py --output configs/calibration.json --engine cpp
python scripts/run_sosp_suite.py --output-dir results --engine cpp --calibration configs/calibration.json
python scripts/plot_sosp_figures.py --input-dir results --output-dir figures
PYTHONPATH=src pytest -q
```

## 说明

- 当前版本是论文驱动的完整重建，不是作者私有实现泄露版。
- raw CSV、raw event、raw metrics、图表脚本都已包含。
- 安全实验面覆盖 latency leakage、reuse leakage、schedule 稳定性、prompt variation、secure mode tradeoff。
