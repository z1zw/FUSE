# FUSE

FUSE 是当前仓库的统一对外名称。仓库内部仍保留 `src/morphaccel` 作为兼容包路径，但所有对外实验、图表、报告、dashboard 和脚本入口都以 `FUSE` 呈现。

## 项目结构

```text
fuse/
├── cpp/
├── rtl/
├── scripts/
├── src/morphaccel/
│   ├── llm/
│   ├── security/
│   ├── sim/
│   ├── workloads/
│   ├── benchmarks.py
│   ├── dashboard.py
│   ├── hardware.py
│   ├── report.py
│   ├── scenarios.py
│   ├── sosp.py
│   ├── sosp_plus.py
│   ├── sosp_viz.py
│   └── viz.py
├── tests/
├── docs/
├── results/
└── figures/
```

## 当前能力

- FUSE simulator 与 C++ timing core 双路径
- GPU 本机实测路径
- ZKP、CNN、AES、Transformer、Hybrid workload
- `VGG16`、`ResNet-50`、`ResNet-101`
- `AES-128`、`AES-256`
- 场景化输入流水线
- 安全评估、消融、mixed-trace、SOSP/SOSP+ 图表
- dashboard、中文报告、投稿版图表输出

## 环境

推荐 Python `>=3.10`。

```powershell
conda activate zw
cd C:\Users\16320\Desktop\Experiments\morphaccel_complete_full_source\morphaccel_sosp
$env:PYTHONPATH='src'
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 构建 C++

```powershell
$env:PYTHONPATH='src'
python scripts\build_cpp.py
```

## 基础评估

```powershell
$env:PYTHONPATH='src'
python scripts\run_benchmarks.py --output-dir results --engine cpp
python scripts\plot_paper_figures.py --input-dir results --output-dir figures
```

## GPU 硬件实测

默认会同时覆盖 `VGG16`、`ResNet-50`、`ResNet-101` 与 `AES-128`、`AES-256`。

```powershell
$env:PYTHONPATH='src'
python scripts\run_hardware_benchmarks.py --output-dir runs\local_gpu\hardware --device auto
```

## 场景实验

```powershell
$env:PYTHONPATH='src'
python scripts\init_fuse_data_layout.py
python scripts\run_scenarios.py --data-root data\scenarios --output-root runs\local_gpu\scenarios --mode fuse --engine gpu
```

## 真实数据入口

预留了 3 类真实数据槽位：

- `data/scenarios/medical_csv`
- `data/scenarios/image_dataset`
- `data/scenarios/llm_small_dataset`

扩展批量实验目录：

- `data/large_runs/medical_csv`
- `data/large_runs/image_dataset`
- `data/large_runs/llm_small_dataset`

推荐配置：

- 医学表格：Breast Cancer Wisconsin (Diagnostic)
- 图像：CIFAR-10 或 Imagenette
- 小规模大模型训练数据：Dolly-15k subset 或 UltraChat subset
- 紧凑开源模型：Qwen2.5-1.5B-Instruct

## SOSP 与扩展评估

```powershell
$env:PYTHONPATH='src'
python scripts\run_sosp_suite.py --output-dir results_sosp --engine cpp
python scripts\plot_sosp_figures.py --input-dir results_sosp --output-dir figures_sosp

python scripts\run_extended_compare.py --output-dir results_sosp_plus --engine cpp
python scripts\plot_sosp_plus_figures.py --input-dir results_sosp_plus --output-dir figures_sosp_plus
```

## 输出物

- `results/*.csv`
- `results/raw/*.csv`
- `figures/*.png`
- `runs/local_gpu/*`
- `runs/cloud_gpu/*`

## 说明

这个仓库当前定位是可运行、可扩展、可上本地 GPU 和云 GPU 的 FUSE 实验基线。内部仍保留部分 `morphaccel` 路径和兼容类名，用于兼容已有脚本和结果，不影响对外交付名称。
