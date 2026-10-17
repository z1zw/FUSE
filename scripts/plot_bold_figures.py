# -*- coding: utf-8 -*-
"""FUSE paper figures — BOLD paper style.

Standard: DejaVu Sans bold throughout; axis labels 21 / ticks 18 / legend 15;
line width 3.2, markers 11 with a black edge at mew 1.4; spines 3.0; major grid
1.3 alpha .55, minor 0.6 alpha .35; single panel (4.7, 4.2), 2x3 grid
(14.1, 8.4); Okabe-Ito palette with BLUE for ours against RED; no figure-level
title; bars always vertical.

Two grids:

  fig_fuse_domains   a Groth16     b VGG16       c AES-128
                     d Language models   e Overhead vs specialist   f Energy

  fig_llm_secure     a Answer length     b Energy vs baselines      c Plan quality
                     d Array scaling     e Sharing vs partitioning  f Per dataset

  fig_concurrent     a Cost per domain   b What security adds       c Request shapes
                     d Array occupancy   e Makespan                 f Energy

Panel (d) of the first grid is the language-model domain added to what was a
three-domain figure; panels (e) and (f) summarise all four domains together.
Every panel is also written on its own.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------------------------------------------------- style ----
BLUE, RED = "#1f77b4", "#d62728"
ORANGE, GREEN, VERM, PURPLE, SKY, GREY = (
    "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#9aa0a6")

FIGSIZE = (4.7, 4.2)
GRID6 = (14.1, 8.4)
LW, MS, MEW = 3.2, 11, 1.4
LAB, TICK, LEG, PANEL = 21, 18, 15, 18
TICK_DENSE, LEG_DENSE = 13, 12

rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 17, "font.weight": "bold",
    "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.linewidth": 3.0,
    "axes.unicode_minus": False, "figure.dpi": 150, "savefig.dpi": 300,
})


def dress(ax, tick=TICK):
    ax.grid(True, which="major", linewidth=1.3, alpha=0.55, color="#999")
    ax.grid(True, which="minor", linewidth=0.6, alpha=0.35, color="#bbb")
    ax.tick_params(axis="both", which="major", width=2.6, length=8, labelsize=tick)
    ax.tick_params(axis="both", which="minor", width=1.5, length=4)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    for s in ax.spines.values():
        s.set_linewidth(3.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend(ax, loc="best", size=LEG, ncol=1):
    ax.legend(frameon=True, framealpha=0.96, edgecolor="#666", loc=loc, ncol=ncol,
              prop={"weight": "bold", "size": size}, handlelength=1.7,
              borderpad=0.32, labelspacing=0.3, handletextpad=0.5, columnspacing=0.9)


def xlab(ax, s, size=LAB):
    ax.set_xlabel(s, fontsize=size, fontweight="bold")


def ylab(ax, s, size=LAB):
    ax.set_ylabel(s, fontsize=size, fontweight="bold")


def line(ax, x, y, color, marker="o", ls="-", label=None, ms=MS, lw=LW, z=3):
    ax.plot(x, y, ls, marker=marker, color=color, lw=lw, ms=ms, markerfacecolor=color,
            markeredgecolor="black", markeredgewidth=MEW, label=label, zorder=z)


def bars(ax, x, h, colors, w=0.68, yerr=None, hatch=None, lw=2.2):
    return ax.bar(x, h, width=w, color=colors, edgecolor="black", linewidth=lw,
                  yerr=yerr, hatch=hatch, zorder=3,
                  error_kw=dict(ecolor="black", elinewidth=2.4, capsize=5, capthick=2.4))


def ratio_line(ax, y=1.0):
    """Parity marker for every normalized panel."""
    ax.axhline(y, color="black", ls="--", lw=2.4, zorder=4)


# ----------------------------------------------------------------- data ----
class Data:
    """Every figure reads from here, so a rerun of any experiment flows through."""

    def __init__(self, root: Path):
        self.root = root
        self.fig4 = self._csv("results/raw/fig4_raw.csv")
        self.zoo = self._csv("runs/llm_zoo/llm_summary.csv")
        self.scaling = self._csv("runs/concurrent/rfu_scaling.csv")
        self.conc = self._csv("runs/concurrent/concurrent_results.csv")
        self.shapes = self._csv("runs/concurrent/request_shapes.csv")
        self.ablation = self._csv("runs/llm_zoo/llm_ablation.csv")
        self.simulated = self._csv("runs/llm_zoo/llm_simulated.csv")
        self.census = self._csv("runs/census/template_census.csv")
        self.granularity = self._csv("runs/census/granularity.csv")
        self.fig11a = self._csv("results/fig11a_sched.csv")
        self.fig11b = self._csv("results/fig11b_switch.csv")
        self.fig11c = self._csv("results/fig11c_summary.csv")
        self.chain = self._csv("runs/census/reuse_breakdown.csv")
        self.frontend = self._csv("runs/llm_frontend/frontend_summary.csv")
        self.answers = self._json("runs/concurrent/answer_lengths.json")

    def _csv(self, rel):
        path = self.root / rel
        # Some result files carry a provenance header as # comments.
        return pd.read_csv(path, comment="#") if path.exists() else None

    def _json(self, rel):
        path = self.root / rel
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def domain(self, name):
        """Latency and energy for one classic domain, FUSE against its specialist."""
        part = self.fig4[self.fig4["domain"] == name]
        # The stored data still uses the old name in its mode column.
        fuse = part[part["mode"] == "FUSE"].sort_values("x")
        base = part[part["mode"] != "FUSE"].sort_values("x")
        return fuse, base, str(base["baseline_name"].iloc[0])

    def zoo_sorted(self):
        return self.zoo.sort_values("parameter_count").reset_index(drop=True)


#: Names long enough to be unambiguous but short enough to sit under a tick.
SHORT = {
    "Qwen2.5-0.5B": "Qwen-0.5B", "TinyLlama-1.1B": "TinyLl-1.1B", "OLMoE-1B-7B": "OLMoE",
    "Qwen2.5-7B": "Qwen-7B", "Qwen2.5-14B": "Qwen-14B", "Qwen2.5-32B": "Qwen-32B",
    "gpt-5.4": "GPT-5.4", "deepseek-v4-pro": "DeepSeek", "kimi-k2.6": "Kimi",
    "minimax-m3": "MiniMax", "gemini-3.1-pro-preview": "Gemini",
}


def short(name):
    return SHORT.get(str(name), str(name))


DOMAIN_STYLE = {
    "Groth16": ("Constraints", lambda v: f"{int(v) // 1024}k", "Static ZKP core"),
    "VGG16": ("Batch size", lambda v: str(int(v)), "Eyeriss-style"),
    "AES-128": ("Blocks", lambda v: f"{int(v) // 1024}k", "AES engine"),
}


def scaled(values):
    """Microseconds when a domain never reaches a millisecond, else milliseconds.

    Three orders of magnitude separate an AES round from a VGG16 batch, and a
    y-axis reading 0.005 to 0.008 is unreadable whatever it is measuring.
    """
    top = float(np.max(values))
    return (values * 1000.0, r"Latency ($\mu$s)") if top < 1.0 else (values, "Latency (ms)")


# ------------------------------------------------- (a-c) classic domains ---
def _domain_panel(ax, D, name, P):
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE + 1, LEG_DENSE)
    fuse, base, _ = D.domain(name)
    axis_label, fmt, short = DOMAIN_STYLE[name]
    x = np.arange(len(fuse))
    base_y, unit = scaled(base["latency_ms"].values)
    fuse_y, _ = scaled(fuse["latency_ms"].values)
    line(ax, x, base_y, RED, marker="s", label=short, ms=10 if not P else 8, lw=2.8)
    line(ax, x, fuse_y, BLUE, marker="P", label="FUSE", ms=13 if not P else 10, lw=3.6, z=9)
    ax.set_xticks(x)
    ax.set_xticklabels([fmt(v) for v in fuse["x"]], fontsize=tick)
    xlab(ax, axis_label, LAB - 3 if not P else LAB - 6)
    ylab(ax, unit, LAB - 3 if not P else LAB - 6)
    ax.margins(y=0.22)
    dress(ax, tick)
    legend(ax, "upper left", leg - 2)


def p_groth(ax, D, P=False):
    _domain_panel(ax, D, "Groth16", P)


def p_vgg(ax, D, P=False):
    _domain_panel(ax, D, "VGG16", P)


def p_aes(ax, D, P=False):
    _domain_panel(ax, D, "AES-128", P)


# --------------------------------------------- (d) the language-model domain
def p_llm_domain(ax, D, P=False):
    """FUSE against three baselines across model scale.

    Absolute curves would lie on top of each other -- a five per cent gap is
    invisible next to a forty-fold range of latency -- so the ratio is plotted
    instead, which is both the result and legible.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE + 1, LEG_DENSE)
    z = D.zoo_sorted()
    x = np.arange(len(z))
    for col_name, label, col, mk in (
            ("transformer_asic_latency_ratio", "vs Transformer ASIC", RED, "s"),
            ("gpu_tensor_latency_ratio", "vs GPU-Tensor", PURPLE, "D"),
            ("multicore_latency_ratio", "vs Multi-core", GREEN, "^")):
        if col_name in z.columns:
            line(ax, x, z[col_name], col, marker=mk, label=label,
                 ms=9 if not P else 7.5, lw=2.8)
    ratio_line(ax, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p / 1e9:.2f}" if p < 1e9 else f"{p / 1e9:.1f}"
                        for p in z["parameter_count"]], fontsize=tick - 2)
    xlab(ax, "Model size (B params)", LAB - 4 if not P else LAB - 7)
    ylab(ax, "Latency / baseline", LAB - 4 if not P else LAB - 7)
    ax.set_ylim(0.96, 1.12)
    ax.set_yticks([1.0, 1.04, 1.08])
    dress(ax, tick)
    legend(ax, "upper left", leg - 4)


# --------------------------------- (e) overhead against the specialist -----
DOM_ORDER = ["AES-128", "VGG16", "LLM", "Groth16"]
DOM_COLOR = {"AES-128": GREEN, "VGG16": ORANGE, "LLM": BLUE, "Groth16": VERM}


def _overheads(D):
    out = {}
    for name in ("AES-128", "VGG16", "Groth16"):
        fuse, base, _ = D.domain(name)
        r = fuse["latency_ms"].values / base["latency_ms"].values
        out[name] = (float(r.mean()), float(r.min()), float(r.max()))
    z = D.zoo_sorted()["transformer_asic_latency_ratio"]
    out["LLM"] = (float(z.mean()), float(z.min()), float(z.max()))
    return out


def p_overhead(ax, D, P=False):
    tick = TICK if not P else TICK_DENSE + 1
    o = _overheads(D)
    x = np.arange(len(DOM_ORDER))
    mean = [o[d][0] for d in DOM_ORDER]
    lo = [o[d][0] - o[d][1] for d in DOM_ORDER]
    hi = [o[d][2] - o[d][0] for d in DOM_ORDER]
    bars(ax, x, mean, [DOM_COLOR[d] for d in DOM_ORDER], w=0.62, yerr=[lo, hi])
    ratio_line(ax, 1.0)
    for i, v in enumerate(mean):
        ax.text(i, v + hi[i] + 0.025, f"{v:.2f}", ha="center", va="bottom",
                fontsize=tick - 2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(DOM_ORDER, rotation=20, ha="right", fontsize=tick)
    ylab(ax, "Latency / specialist", LAB - 4 if not P else LAB - 7)
    ax.set_ylim(0.95, 1.45)
    ax.set_yticks([1.0, 1.1, 1.2, 1.3, 1.4])
    dress(ax, tick)


# ------------------------------------------------ (f) energy across domains
def p_energy_domains(ax, D, P=False):
    tick = TICK if not P else TICK_DENSE + 1
    out = {}
    for name in ("AES-128", "VGG16", "Groth16"):
        fuse, base, _ = D.domain(name)
        out[name] = float((fuse["energy_nj"].values / base["energy_nj"].values).mean())
    out["LLM"] = float((1.0 / D.zoo_sorted()["transformer_asic_energy_gain"]).mean())
    x = np.arange(len(DOM_ORDER))
    vals = [out[d] for d in DOM_ORDER]
    bars(ax, x, vals, [DOM_COLOR[d] for d in DOM_ORDER], w=0.62)
    ratio_line(ax, 1.0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.2f}", ha="center", va="bottom",
                fontsize=tick - 2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(DOM_ORDER, rotation=20, ha="right", fontsize=tick)
    ylab(ax, "Energy / specialist", LAB - 4 if not P else LAB - 7)
    ax.set_ylim(0.9, 2.1)
    ax.set_yticks([1.0, 1.4, 1.8])
    dress(ax, tick)


# -------------------------------------- (a) answer length, hosted models ---
ANSWER_STYLE = [("gpt-5.4", "GPT-5.4", BLUE, "P"),
                ("deepseek-v4-pro", "DeepSeek-v4", ORANGE, "s"),
                ("kimi-k2.6", "Kimi-k2.6", GREEN, "^"),
                ("minimax-m3", "MiniMax-m3", PURPLE, "D"),
                ("gemini-3.1-pro-preview", "Gemini-3.1", VERM, "v")]


def p_answer_len(ax, D, P=False):
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    ds = D.answers["datasets"]
    names = [k for k in ("gsm8k", "mmlu", "humaneval") if k in ds]
    labels = [ds[k]["label"] for k in names]
    x = np.arange(len(names))
    width = 0.15
    for i, (key, name, col, _) in enumerate(ANSWER_STYLE):
        vals = [ds[k]["per_model"].get(key, {}).get("median", np.nan) for k in names]
        ax.bar(x + (i - 2) * width, vals, width=width, color=col, edgecolor="black",
               linewidth=1.6, zorder=3, label=name)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=tick)
    ylab(ax, "Answer length (tok)", LAB - 4 if not P else LAB - 7)
    ax.set_ylim(40, 4000)
    dress(ax, tick)
    legend(ax, "upper left", leg - 4, ncol=2)


# --------------------------------------- (b) energy against the baselines --
ZOO_BASES = [("transformer_asic_energy_gain", "T-ASIC", RED),
             ("gpu_tensor_energy_gain", "GPU-Tensor", PURPLE),
             ("multicore_energy_gain", "Multi-core", GREEN)]


def p_llm_energy(ax, D, P=False):
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    z = D.zoo_sorted()
    x = np.arange(len(z))
    width = 0.26
    for i, (col_name, label, col) in enumerate(ZOO_BASES):
        ax.bar(x + (i - 1) * width, z[col_name], width=width, color=col,
               edgecolor="black", linewidth=1.8, zorder=3, label=label)
    ratio_line(ax, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in z["model_label"]], rotation=30, ha="right",
                       fontsize=tick - 3)
    ylab(ax, "Energy of baseline / FUSE", LAB - 6 if not P else LAB - 9)
    ax.set_ylim(0, 1.65)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    dress(ax, tick)
    legend(ax, "upper left", leg - 3, ncol=3)


# ---------------------------------------------- (c) plan quality -----------
def p_plan_quality(ax, D, P=False):
    """How often a compiler returns a plan, and how often it is the right plan.

    The second bar is the one that matters: a reply can parse cleanly and still
    have lost the sizes the request stated, and the distance between the two
    bars is exactly that failure.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    f = D.frontend.copy()
    f["_k"] = f["kind"].map({"hosted": 0, "local": 1}).fillna(9)
    f = f.sort_values(["_k", "plan_fidelity"], ascending=[True, False]).reset_index(drop=True)
    x = np.arange(len(f))
    width = 0.38
    parses = 100.0 * f["parse_rate"].values
    sizes = 100.0 * f["plan_fidelity"].values
    ax.bar(x - width / 2, parses, width=width, color=ORANGE,
           edgecolor="black", linewidth=1.8, zorder=3, label="Returns a plan")
    ax.bar(x + width / 2, sizes, width=width, color=BLUE,
           edgecolor="black", linewidth=1.8, zorder=3, label="Right sizes")
    split = int((f["_k"] == 0).sum())
    if 0 < split < len(f):
        ax.axvline(split - 0.5, color="black", ls=":", lw=2.4, zorder=4)
        ax.text(split / 2 - 0.5, 104, "hosted", ha="center", fontsize=tick - 2,
                fontweight="bold", color="#444")
        ax.text((split + len(f)) / 2 - 0.5, 104, "local", ha="center", fontsize=tick - 2,
                fontweight="bold", color="#444")
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in f["model"]], rotation=38, ha="right",
                       fontsize=tick - 4)
    ylab(ax, "Requests compiled (%)", LAB - 5 if not P else LAB - 8)
    ax.set_ylim(0, 152)
    ax.set_yticks([0, 25, 50, 75, 100])
    dress(ax, tick)
    legend(ax, "upper center", leg - 4, ncol=2)


# ------------------------------------------------- (d) array scaling -------
SCALE_STYLE = [("GPT-2", BLUE, "P"), ("Qwen2.5-0.5B", ORANGE, "s"),
               ("TinyLlama-1.1B", GREEN, "^"), ("Mistral-7B", PURPLE, "D"),
               ("Qwen2.5-14B", VERM, "v"), ("Qwen2.5-32B", SKY, "<")]
SCALE_LABEL = {name: None for name, _, _ in SCALE_STYLE}


def p_scaling(ax, D, P=False):
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    s = D.scaling
    s = s[s["dataset"] == s["dataset"].iloc[0]]
    limit = int(s["rfus"].max())
    for label, col, mk in SCALE_STYLE:
        part = s[s["model_label"] == label].sort_values("rfus")
        if part.empty:
            continue
        base = float(part[part["rfus"] == 1]["makespan_ms"].iloc[0])
        ax.plot(part["rfus"], base / part["makespan_ms"], "-", color=col, lw=2.8,
                marker=mk, markevery=4, ms=10 if not P else 8, markerfacecolor=col,
                markeredgecolor="black", markeredgewidth=MEW, label=short(label), zorder=3)
    ax.plot([1, limit], [1, limit], ":", color="#555", lw=2.4, zorder=2, label="Linear")
    xlab(ax, "RFUs given to the model", LAB - 5 if not P else LAB - 8)
    ylab(ax, "Speedup over 1 RFU", LAB - 5 if not P else LAB - 8)
    ax.set_xlim(0, limit + 1)
    ax.set_ylim(0, limit + 2)
    dress(ax, tick)
    legend(ax, "upper left", leg - 5, ncol=2)


# ---------------------------------- (e) sharing against fixed partitioning -
def p_sharing(ax, D, P=False):
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    c = D.conc
    level = sorted(c["security_per_request"].unique())[0]
    c = c[c["security_per_request"] == level]
    for label, col, mk in SCALE_STYLE:
        part = c[c["model_label"] == label]
        if part.empty:
            continue
        g = part.groupby("rfus")["speedup_vs_partitioned"].mean().reset_index()
        line(ax, g["rfus"], g["speedup_vs_partitioned"], col, marker=mk, label=short(label),
             ms=8 if not P else 6.5, lw=2.8)
    ratio_line(ax, 1.0)
    ax.axvline(16, color="#555", ls=":", lw=2.4, zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32])
    ax.set_xticklabels(["4", "8", "16", "32"], fontsize=tick)
    xlab(ax, "Total RFUs on chip", LAB - 5 if not P else LAB - 8)
    ylab(ax, "FUSE / best partition", LAB - 5 if not P else LAB - 8)
    ax.set_ylim(0.95, 2.15)
    ax.set_yticks([1.0, 1.4, 1.8])
    dress(ax, tick)
    legend(ax, "upper right", leg - 6, ncol=1)


# --------------------------------------- (f) per dataset at the evaluated --
DATASET_COLOR = {"GSM8K": BLUE, "MMLU": ORANGE, "HumanEval": GREEN}


def p_per_dataset(ax, D, P=False):
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    c = D.conc
    c = c[(c["rfus"] == 16) & (c["security_per_request"] == sorted(
        c["security_per_request"].unique())[0])]
    models = [m for m, _, _ in SCALE_STYLE if m in set(c["model_label"])]
    names = [n for n in ("GSM8K", "MMLU", "HumanEval") if n in set(c["dataset_label"])]
    x = np.arange(len(models))
    width = 0.26
    for i, name in enumerate(names):
        vals = [float(c[(c["model_label"] == m) & (c["dataset_label"] == name)]
                      ["speedup_vs_partitioned"].iloc[0]) for m in models]
        ax.bar(x + (i - 1) * width, vals, width=width, color=DATASET_COLOR[name],
               edgecolor="black", linewidth=1.8, zorder=3, label=name)
    ratio_line(ax, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in models], rotation=30, ha="right",
                       fontsize=tick - 3)
    ylab(ax, "FUSE / best partition", LAB - 5 if not P else LAB - 8)
    ax.set_ylim(0.95, 1.38)
    ax.set_yticks([1.0, 1.1, 1.2, 1.3])
    dress(ax, tick)
    legend(ax, "upper left", leg - 4, ncol=3)


# ----------------------------------------------------------------- driver --
GRID_DOMAINS = [("groth", p_groth, "(a) Groth16"),
                ("vgg", p_vgg, "(b) VGG16"),
                ("aes", p_aes, "(c) AES-128"),
                ("llm", p_llm_domain, "(d) Language models"),
                ("overhead", p_overhead, "(e) Overhead vs specialist"),
                ("energy", p_energy_domains, "(f) Energy vs specialist")]

GRID_SECURE = [("answerlen", p_answer_len, "(a) Answer length"),
               ("llmenergy", p_llm_energy, "(b) Energy vs baselines"),
               ("planquality", p_plan_quality, "(c) Plan quality"),
               ("scaling", p_scaling, "(d) Array scaling"),
               ("sharing", p_sharing, "(e) Sharing vs partition"),
               ("perdataset", p_per_dataset, "(f) At the 16-RFU array")]


# ===================================================================== #
# fig_concurrent -- the three domains on one array                      #
# ===================================================================== #
REF_RFUS = 16
#: One dataset carries the per-model panels; the others differ only in how many
#: tokens are generated, and panel (c) shows the whole sweep anyway.
REF_DATASET = "gsm8k"


def _ref(D):
    """Every model at the evaluated array, in order of size."""
    c = D.conc
    r = c[(c["rfus"] == REF_RFUS) & (c["dataset"] == REF_DATASET)]
    order = {name: i for i, name in enumerate(
        ["GPT-2", "Qwen2.5-0.5B", "TinyLlama-1.1B", "Mistral-7B",
         "Qwen2.5-14B", "Qwen2.5-32B"])}
    return r.assign(_o=r["model_label"].map(order)).sort_values("_o").reset_index(drop=True)


# ------------------------------------------ (a) what each domain costs -----
def p_domain_cost(ax, D, P=False):
    """The three concurrent domains, on a log axis because that is the point.

    A Groth16 proof and an AES payload are two to three orders of magnitude
    cheaper than the decode they protect. That is the whole reason the security
    work can be absorbed, and it is not visible on a linear axis.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    r = _ref(D)
    x = np.arange(len(r))
    w = 0.27
    for dx, col, key, lab in ((-w, BLUE, "fuse_llm_latency_ms", "Language model"),
                              (0.0, ORANGE, "fuse_aes_latency_ms", "AES-128"),
                              (w, GREEN, "fuse_zkp_latency_ms", "Groth16")):
        ax.bar(x + dx, r[key].values, width=w, color=col, edgecolor="black",
               linewidth=1.8, zorder=3, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in r["model_label"]], rotation=38, ha="right",
                       fontsize=tick - 4)
    ylab(ax, "Busy time (ms)", LAB - 4 if not P else LAB - 7)
    ax.set_ylim(0.1, 9e4)
    dress(ax, tick)
    legend(ax, "upper left", leg - 4, ncol=3)


# --------------------------------------- (b) the obligation is absorbed ----
def p_absorbed(ax, D, P=False):
    """Makespan with the security work, against the same trace without it.

    FUSE lands on the language model's own latency; the partition pays for the
    units it set aside whether or not the security path is using them.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    r = _ref(D)
    x = np.arange(len(r))
    w = 0.38
    alone = r["llm_only_ms"].values
    ax.bar(x - w / 2, r["fuse_ms"].values / alone, width=w, color=BLUE,
           edgecolor="black", linewidth=1.8, zorder=3, label="FUSE")
    ax.bar(x + w / 2, r["partitioned_ms"].values / alone, width=w, color=RED,
           edgecolor="black", linewidth=1.8, zorder=3, label="Best partition")
    ratio_line(ax, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in r["model_label"]], rotation=38, ha="right",
                       fontsize=tick - 4)
    ylab(ax, "Makespan / model alone", LAB - 6 if not P else LAB - 9)
    ax.set_ylim(0.9, 1.42)
    dress(ax, tick)
    legend(ax, "upper left", leg - 3, ncol=2)


# ------------------------------------------------ (c) the request shapes ---
def p_request_shapes(ax, D, P=False):
    """What one request of each shape actually demands.

    The point of the panel is which bars move together. The answer length is
    measured, not assumed, and the AES payload follows it; the proof is a fixed
    circuit and does not move at all, which is why longer answers make the
    encryption grow and leave the attestation where it was.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    shapes = D.shapes
    x = np.arange(len(shapes))
    w = 0.27
    for dx, col, key, lab in ((-w, GREEN, "prompt_tokens", "Prompt (tok)"),
                              (0.0, BLUE, "gen_tokens", "Answer (tok)"),
                              (w, ORANGE, "aes_blocks", "AES (blocks)")):
        ax.bar(x + dx, shapes[key].values, width=w, color=col, edgecolor="black",
               linewidth=1.8, zorder=3, label=lab)
    zkp = int(shapes["zkp_constraints"].iloc[0])
    ax.axhline(zkp, color=RED, ls="--", lw=2.6, zorder=4,
               label=f"Groth16 ({zkp} constr.)")
    ax.set_yscale("log")
    ax.set_ylim(20, 2.4e4)
    ax.set_xticks(x)
    ax.set_xticklabels(shapes["label"].values, fontsize=tick - 2)
    ylab(ax, "Per request", LAB - 3 if not P else LAB - 6)
    dress(ax, tick)
    legend(ax, "upper left", leg - 4, ncol=2)


# ------------------------------------------------- (d) why it is absorbed --
def p_occupancy(ax, D, P=False):
    """How busy FUSE keeps the array, against what a partition has to set aside.

    The two curves cross: while the array is scarce FUSE is saturated and the
    reserved units are a large fraction of it, and by 32 RFUs neither is true.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    c = D.conc
    g = c.groupby("rfus")
    sizes = np.array(sorted(c["rfus"].unique()), dtype=float)
    util = 100.0 * g["fuse_rfu_utilization"].mean().values
    reserved = 100.0 * (g["partitioned_reserved"].mean().values / sizes)
    line(ax, sizes, util, BLUE, "P", label="FUSE array busy")
    line(ax, sizes, reserved, RED, "s", ls="--", label="Partition reserved")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(int(v)) for v in sizes], fontsize=tick)
    ax.minorticks_off()
    xlab(ax, "Total RFUs on chip", LAB - 3 if not P else LAB - 6)
    ylab(ax, "Share of the array (%)", LAB - 5 if not P else LAB - 8)
    ax.set_ylim(0, 105)
    dress(ax, tick)
    legend(ax, "lower left", leg - 3)


# ------------------------------------------------ (e) the trace in full ----
def p_makespan(ax, D, P=False):
    """Absolute makespan for one model as the array grows.

    Ratios are elsewhere; this is what the service actually waits, and it shows
    the three designs converging rather than one of them winning outright.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    c = D.conc
    r = c[(c["model_label"] == "Mistral-7B") & (c["dataset"] == REF_DATASET)]
    r = r.sort_values("rfus")
    x = r["rfus"].values.astype(float)
    line(ax, x, r["fuse_ms"].values, BLUE, "P", label="FUSE")
    line(ax, x, r["partitioned_ms"].values, RED, "s", label="Best partition")
    line(ax, x, r["dedicated_ms"].values, ORANGE, "^", ls="--", label="Dedicated")
    ax.axvline(REF_RFUS, color=GREY, ls=":", lw=2.6, zorder=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x], fontsize=tick)
    ax.minorticks_off()
    xlab(ax, "Total RFUs on chip", LAB - 3 if not P else LAB - 6)
    ylab(ax, "Makespan (ms)", LAB - 3 if not P else LAB - 6)
    dress(ax, tick)
    legend(ax, "upper right", leg - 3)


# ------------------------------------------------------------ (f) energy ---
def p_conc_energy(ax, D, P=False):
    """Energy against both baselines as the array grows.

    The margin over a partition decays with the array for the same reason the
    latency margin does. The gap to engines that also get their domain's
    datapath does not move, and the panel is drawn to show both.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    g = D.conc.groupby("rfus")
    sizes = np.array(sorted(D.conc["rfus"].unique()), dtype=float)
    part = g["energy_gain_vs_partitioned"].mean().values
    ded = g["energy_gain_vs_dedicated"].mean().values
    line(ax, sizes, part, BLUE, "P", label="vs best partition")
    line(ax, sizes, ded, RED, "s", ls="--", label="vs dedicated")
    ratio_line(ax, 1.0)
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(int(v)) for v in sizes], fontsize=tick)
    ax.minorticks_off()
    xlab(ax, "Total RFUs on chip", LAB - 3 if not P else LAB - 6)
    ylab(ax, "Baseline energy / FUSE", LAB - 6 if not P else LAB - 9)
    ax.set_ylim(0.8, 1.12)
    dress(ax, tick)
    legend(ax, "center right", leg - 3)


GRID_CONCURRENT = [("domaincost", p_domain_cost, "(a) Cost of each domain"),
                   ("absorbed", p_absorbed, "(b) Cost of the obligation"),
                   ("shapes", p_request_shapes, "(c) Request composition"),
                   ("occupancy", p_occupancy, "(d) Array allocation"),
                   ("makespan", p_makespan, "(e) Makespan, Mistral-7B"),
                   ("energy", p_conc_energy, "(f) Energy")]


# ===================================================================== #
# fig11d -- the mechanism counters behind the scheduling result         #
# ===================================================================== #
#: Counter, label, and whether a lower value is the good direction.
MECHANISM = [
    ("lrc_switch_events", "Reconfigs"),
    ("switch_cycles_total", "Switch cyc."),
    ("fvm_redundant_reads", "Redund. rds"),
    ("buffer_ops", "Buffer ops"),
    ("energy_nj", "Energy"),
]


def _pair(root):
    """The two sides of the paired Figure 11 trace, as {counter: value}."""
    out = {}
    for tag, name in (("fuse", "morph"), ("multicore", "multicore")):
        path = root / "results" / "raw" / f"fig11_{name}_metrics.csv"
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        out[tag] = {k: v for k, v in zip(frame["key"], pd.to_numeric(frame["value"], errors="coerce"))}
    return out


def p_mechanisms(ax, D, P=False):
    """Each counter under FUSE as a percentage of the same counter unscheduled.

    The panel is the 'via TARS and FVM' half of the figure's caption: the
    scheduling result is elsewhere, and this is where it comes from.
    """
    tick = TICK if not P else TICK_DENSE
    pair = _pair(D.root)
    x = np.arange(len(MECHANISM))
    pct = np.array([100.0 * pair["fuse"][k] / pair["multicore"][k] for k, _ in MECHANISM])
    ax.bar(x, pct, width=0.66, color=BLUE, edgecolor="black", linewidth=2.2, zorder=3)
    ax.axhline(100.0, color=RED, ls="--", lw=2.8, zorder=4, label="Multi-core")
    for xi, v in zip(x, pct):
        ax.text(xi, v + 3.0, f"{v:.0f}", ha="center", va="bottom",
                fontsize=tick - 2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in MECHANISM], rotation=28, ha="right",
                       fontsize=tick - 5)
    ylab(ax, "FUSE, % of multi-core", LAB - 5 if not P else LAB - 8)
    ax.set_ylim(0, 142)
    ax.set_yticks([0, 25, 50, 75, 100])
    dress(ax, tick)
    legend(ax, "upper left", (LEG if not P else LEG_DENSE) - 3)


# --------------------------------------------- granularity: the real axis --
GRAN_STYLE = [("Groth16", VERM, "s"), ("AES-128", GREEN, "^"),
              ("MLP", PURPLE, "D"), ("LLM", BLUE, "P"), ("VGG16", ORANGE, "o")]


def p_granularity(ax, D, P=False):
    """What the reuse machinery is worth, against the size of a fragment.

    Reconfiguration is a fixed cost per fragment, so removing it matters in
    proportion to how little else the fragment does. Across five workload
    families the two track each other over six orders of magnitude.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    g = D.granularity
    for domain, col, mk in GRAN_STYLE:
        d = g[g["domain"] == domain].sort_values("ops_per_fragment")
        if d.empty:
            continue
        ax.plot(d["ops_per_fragment"], d["speedup"], marker=mk, ls="none",
                color=col, ms=13 if not P else 11, markerfacecolor=col,
                markeredgecolor="black", markeredgewidth=MEW, label=domain, zorder=4)
    ratio_line(ax, 1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.9, 13)
    ax.set_yticks([1, 2, 5, 10])
    ax.set_yticklabels(["1", "2", "5", "10"])
    xlab(ax, "Operations per fragment", LAB - 5 if not P else LAB - 8)
    ylab(ax, "Degraded / FUSE", LAB - 5 if not P else LAB - 8)
    dress(ax, tick)
    legend(ax, "upper right", leg - 4, ncol=2)


def p_zkp_sweep(ax, D, P=False):
    """The same ablation as one workload grows, where granularity is fixed.

    Fragments multiply but each stays the same size, so the trend here is not
    the one in (b): it is that FUSE's reconfiguration cost stops growing.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    d = D.granularity
    d = d[d["domain"] == "Groth16"].sort_values("fragments")
    line(ax, d["fragments"], d["degraded_switch"] / 1e3, GREY, "s", label="Degraded")
    line(ax, d["fragments"], d["fuse_switch"] / 1e3, BLUE, "P", label="FUSE")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(5, 600)
    xlab(ax, "Fragments in the circuit", LAB - 5 if not P else LAB - 8)
    ylab(ax, "Reconfig. cycles (k)", LAB - 5 if not P else LAB - 8)
    dress(ax, tick)
    legend(ax, "lower right", leg - 3)


def p_zkp_speedup(ax, D, P=False):
    """And what that saturation costs the ratio, as the circuit grows."""
    tick = TICK if not P else TICK_DENSE
    d = D.granularity
    d = d[d["domain"] == "Groth16"].sort_values("fragments")
    x = np.arange(len(d))
    ax.bar(x, d["speedup"], width=0.66, color=VERM, edgecolor="black",
           linewidth=2.0, zorder=3)
    ratio_line(ax, 1.0)
    for xi, v in zip(x, d["speedup"]):
        ax.text(xi, v + 0.25, f"{v:.1f}", ha="center", va="bottom",
                fontsize=tick - 3, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(d["point"], rotation=30, ha="right", fontsize=tick - 3)
    xlab(ax, "Circuit constraints", LAB - 5 if not P else LAB - 8)
    ylab(ax, "Degraded / FUSE", LAB - 5 if not P else LAB - 8)
    ax.set_ylim(0, 10.4)
    dress(ax, tick)


# ===================================================================== #
# fig_ablation -- FUSE against a degraded build of itself               #
# ===================================================================== #
#: The three classic domains, read from the metrics the ablation run wrote.
CLASSIC = [("Groth16", "groth16"), ("MLP", "mlp"), ("VGG16", "vgg16")]


def _metrics(root, task, mode):
    path = root / "results" / "raw" / f"fig9_{task}_{mode}_metrics.csv"
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return dict(zip(frame["key"], pd.to_numeric(frame["value"], errors="coerce")))


def _ablation(D):
    """One row per workload: the classic three, then the model zoo.

    Returns label, degraded/FUSE latency, degraded/FUSE reconfiguration cycles,
    and which of the two reuse mechanisms fired under FUSE.
    """
    rows = []
    for label, task in CLASSIC:
        deg, fuse = _metrics(D.root, task, "degraded"), _metrics(D.root, task, "morph")
        if not (deg and fuse):
            return None
        rows.append({
            "label": label, "domain": label,
            "deg_ms": deg["makespan_ms"], "fuse_ms": fuse["makespan_ms"],
            "deg_switch": deg["switch_cycles_total"],
            "fuse_switch": fuse["switch_cycles_total"],
            "structural": 100.0 * fuse["structural_reuse_rate"],
            "result": 100.0 * fuse["result_reuse_rate"],
        })
    if D.ablation is None:
        return None
    zoo = D.zoo.set_index("model_label") if D.zoo is not None else None
    for _, r in D.ablation.sort_values("parameter_count").iterrows():
        rows.append({
            "label": short(r["model_label"]), "domain": "LLM",
            "deg_ms": r["degraded_latency_ms"], "fuse_ms": r["fuse_latency_ms"],
            "deg_switch": r["degraded_switch_cycles"],
            "fuse_switch": r["fuse_switch_cycles"],
            "structural": 100.0 * r["fuse_structural_reuse"],
            "result": 100.0 * r["fuse_result_reuse"],
        })
    frame = pd.DataFrame(rows)
    frame["speedup"] = frame["deg_ms"] / frame["fuse_ms"]
    frame["switch_gain"] = frame["deg_switch"] / frame["fuse_switch"]
    return frame


DOMAIN_COLOR = {"Groth16": VERM, "MLP": PURPLE, "VGG16": ORANGE, "LLM": BLUE}


# ------------------------------------------------------------ (a) latency --
def p_abl_latency(ax, D, P=False):
    """Absolute latency with the reuse machinery off, then on."""
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    f = _ablation(D)
    x = np.arange(len(f))
    w = 0.38
    ax.bar(x - w / 2, f["deg_ms"], width=w, color=GREY, edgecolor="black",
           linewidth=1.8, zorder=3, label="Degraded")
    ax.bar(x + w / 2, f["fuse_ms"], width=w, color=BLUE, edgecolor="black",
           linewidth=1.8, zorder=3, label="FUSE")
    ax.set_yscale("log")
    ax.set_ylim(3e-3, 2e4)
    ax.set_xticks(x)
    ax.set_xticklabels(f["label"], rotation=40, ha="right", fontsize=tick - 5)
    ylab(ax, "Latency (ms)", LAB - 3 if not P else LAB - 6)
    dress(ax, tick)
    legend(ax, "upper left", leg - 3, ncol=2)


# ------------------------------------------------------------ (b) speedup --
def p_abl_speedup(ax, D, P=False):
    """What the reuse machinery is worth, per workload.

    Only Groth16 moves. The panel is drawn on a log axis so that the domains
    which do not move are visibly at parity rather than merely short.
    """
    tick = TICK if not P else TICK_DENSE
    f = _ablation(D)
    x = np.arange(len(f))
    ax.bar(x, f["speedup"], width=0.66,
           color=[DOMAIN_COLOR[d] for d in f["domain"]],
           edgecolor="black", linewidth=1.8, zorder=3)
    ratio_line(ax, 1.0)
    ax.set_yscale("log")
    ax.set_ylim(0.9, 14)
    ax.set_yticks([1, 2, 5, 10])
    ax.set_yticklabels(["1", "2", "5", "10"])
    ax.set_xticks(x)
    ax.set_xticklabels(f["label"], rotation=40, ha="right", fontsize=tick - 5)
    ylab(ax, "Degraded / FUSE", LAB - 4 if not P else LAB - 7)
    dress(ax, tick)


# ---------------------------------------------------------- (c) mechanism --
def p_abl_mechanism(ax, D, P=False):
    """Which of the two kinds of reuse actually fires, per domain.

    They are different quantities with different correctness conditions, and
    the domains separate on them: Groth16 and the language models reuse
    instruction templates and never a result, the MLP does the opposite.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    f = _ablation(D)
    agg = f.groupby("domain")[["structural", "result"]].mean()
    agg = agg.reindex([d for d in ("Groth16", "MLP", "VGG16", "LLM") if d in agg.index])
    x = np.arange(len(agg))
    w = 0.38
    ax.bar(x - w / 2, agg["structural"], width=w, color=BLUE, edgecolor="black",
           linewidth=1.8, zorder=3, label="Template reuse")
    ax.bar(x + w / 2, agg["result"], width=w, color=ORANGE, edgecolor="black",
           linewidth=1.8, zorder=3, label="Result reuse")
    for i, (a, b) in enumerate(zip(agg["structural"], agg["result"])):
        for dx, v in ((-w / 2, a), (w / 2, b)):
            if v <= 1e-9:
                ax.text(x[i] + dx, 2.0, "0", ha="center", va="bottom",
                        fontsize=tick - 4, fontweight="bold", color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, fontsize=tick - 2)
    ylab(ax, "Fragments reused (%)", LAB - 5 if not P else LAB - 8)
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 25, 50, 75, 100])
    dress(ax, tick)
    legend(ax, "upper center", leg - 4, ncol=2)


# ------------------------------------------------------ (d) reconfiguration -
def p_abl_switch(ax, D, P=False):
    """The reconfiguration cycles the machinery removes, whatever latency does."""
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    f = _ablation(D)
    x = np.arange(len(f))
    w = 0.38
    ax.bar(x - w / 2, f["deg_switch"] / 1e3, width=w, color=GREY, edgecolor="black",
           linewidth=1.8, zorder=3, label="Degraded")
    ax.bar(x + w / 2, f["fuse_switch"] / 1e3, width=w, color=BLUE, edgecolor="black",
           linewidth=1.8, zorder=3, label="FUSE")
    ax.set_yscale("log")
    ax.set_ylim(5, 1.2e3)
    ax.set_xticks(x)
    ax.set_xticklabels(f["label"], rotation=40, ha="right", fontsize=tick - 5)
    ylab(ax, "Reconfig. cycles (k)", LAB - 5 if not P else LAB - 8)
    dress(ax, tick)
    legend(ax, "upper right", leg - 3, ncol=2)


# -------------------------------------------------------- (e) the decoupling
def p_abl_decoupling(ax, D, P=False):
    """Fewer reconfiguration cycles does not imply a shorter request.

    Every workload sits to the right of parity on the horizontal axis, and only
    one of them rises off parity on the vertical. Where a fragment occupies the
    array for a long time, the reconfiguration it saves is hidden behind
    compute; where fragments are small, the same saving is the whole result.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    f = _ablation(D)
    for domain, group in f.groupby("domain"):
        ax.scatter(group["switch_gain"], group["speedup"], s=190 if not P else 150,
                   color=DOMAIN_COLOR[domain], edgecolor="black", linewidth=1.8,
                   zorder=4, label=domain)
    ax.axhline(1.0, color="black", ls="--", lw=2.4, zorder=3)
    ax.axvline(1.0, color=GREY, ls=":", lw=2.4, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.45, 22)
    ax.set_ylim(0.95, 14)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_yticks([1, 2, 5, 10])
    ax.set_yticklabels(["1", "2", "5", "10"])
    ax.minorticks_off()
    xlab(ax, r"Reconfig. cycles saved ($\times$)", LAB - 6 if not P else LAB - 9)
    ylab(ax, r"Latency speedup ($\times$)", LAB - 6 if not P else LAB - 9)
    dress(ax, tick)
    legend(ax, "upper left", leg - 4, ncol=2)


# --------------------------------------------------------- (f) compression --
def p_abl_paths(ax, D, P=False):
    """How many distinct execution paths the fragments of one model collapse to."""
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    sim = D.simulated[D.simulated["mode"] == "fuse"]
    z = (sim.groupby("model_label")
            .agg(params=("parameter_count", "first"),
                 fragments=("fragments", "mean"),
                 paths=("representative_paths", "mean"))
            .sort_values("params").reset_index())
    x = np.arange(len(z))
    w = 0.38
    ax.bar(x - w / 2, z["fragments"], width=w, color=GREY, edgecolor="black",
           linewidth=1.8, zorder=3, label="Fragments")
    ax.bar(x + w / 2, z["paths"], width=w, color=BLUE,
           edgecolor="black", linewidth=1.8, zorder=3, label="Representative paths")
    ax.set_yscale("log")
    ax.set_ylim(5, 2.2e4)
    ax.set_xticks(x)
    ax.set_xticklabels([short(m) for m in z["model_label"]], rotation=40, ha="right",
                       fontsize=tick - 5)
    ylab(ax, "Count per request", LAB - 5 if not P else LAB - 8)
    dress(ax, tick)
    legend(ax, "upper left", leg - 4)


GRID_ABLATION = [("latency", p_abl_latency, "(a) Ablation latency"),
                 ("granularity", p_granularity, "(b) Gain vs fragment size"),
                 ("mechanism", p_abl_mechanism, "(c) Reuse by kind"),
                 ("switch", p_abl_switch, "(d) Reconfig. cycles"),
                 ("zkpsweep", p_zkp_sweep, "(e) Groth16 scaling"),
                 ("zkpspeedup", p_zkp_speedup, "(f) Groth16 ablation")]


# ===================================================================== #
# fig_reuse -- what the four domains share, and at which level          #
# ===================================================================== #
FAMILY_STYLE = [("linear_map", "Linear map", BLUE),
                ("nonlinear_lookup", "Nonlinear lookup", ORANGE),
                ("permutation", "Permutation", GREEN),
                ("reduction", "Reduction", VERM)]
CENSUS_DOMAINS = ["Groth16", "VGG16", "AES-128", "LLM"]


def _census(D):
    c = D.census
    return None if c is None else c[c["domain"] != "_fanout"]


# ------------------------------------------------- (a) template families ---
def p_family_share(ax, D, P=False):
    """Every domain draws on the same handful of RFU template families.

    This is the sharing the fabric actually exploits: a fragment is mapped by
    the family its operator belongs to, so four domains that look nothing alike
    are asking for four configurations of the same datapath.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    c = _census(D)
    wide = c.pivot(index="domain", columns="template_family",
                   values="share").reindex(CENSUS_DOMAINS).fillna(0.0)
    x = np.arange(len(wide))
    w = 0.2
    for i, (key, label, col) in enumerate(FAMILY_STYLE):
        if key not in wide:
            continue
        ax.bar(x + (i - 1.5) * w, wide[key].values, width=w, color=col,
               edgecolor="black", linewidth=1.6, zorder=3, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(wide.index, rotation=28, ha="right", fontsize=tick - 3)
    ylab(ax, "Fragments (%)", LAB - 3 if not P else LAB - 6)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    dress(ax, tick)
    legend(ax, "upper center", leg - 5, ncol=2)


# ----------------------------------------------------- (b) three counts ----
def p_class_counts(ax, D, P=False):
    """Fragments, the structural classes they fall into, and the semantic ones.

    The middle bar is what the fabric configures for; the right bar is what it
    would have to match to reuse a value, and it never repeats.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    c = _census(D)
    g = c.groupby("domain").first().reindex(CENSUS_DOMAINS)
    x = np.arange(len(g))
    w = 0.27
    for dx, col, key, lab in ((-w, GREY, "fragments", "Fragments"),
                              (0.0, BLUE, "structural_classes", "Structural classes"),
                              (w, ORANGE, "semantic_classes", "Semantic classes")):
        ax.bar(x + dx, g[key].values, width=w, color=col, edgecolor="black",
               linewidth=1.6, zorder=3, label=lab)
    ax.set_yscale("log")
    ax.set_ylim(8, 2.2e4)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=28, ha="right", fontsize=tick - 3)
    ylab(ax, "Count per request", LAB - 4 if not P else LAB - 7)
    dress(ax, tick)
    legend(ax, "upper left", leg - 5)


# ------------------------------------------------------ (c) compression ----
def p_compression(ax, D, P=False):
    """How far the structural hash collapses a domain's fragments."""
    tick = TICK if not P else TICK_DENSE
    c = _census(D)
    g = c.groupby("domain").first().reindex(CENSUS_DOMAINS)
    ratio = g["fragments"] / g["structural_classes"]
    x = np.arange(len(g))
    ax.bar(x, ratio.values, width=0.64,
           color=[DOMAIN_COLOR.get(d, BLUE) for d in g.index],
           edgecolor="black", linewidth=2.0, zorder=3)
    for xi, v in zip(x, ratio.values):
        ax.text(xi, v * 1.1, f"{v:.0f}$\\times$", ha="center", va="bottom",
                fontsize=tick - 3, fontweight="bold")
    ax.set_yscale("log")
    ax.set_ylim(1, 420)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, rotation=28, ha="right", fontsize=tick - 3)
    ylab(ax, "Fragments per class", LAB - 5 if not P else LAB - 8)
    dress(ax, tick)


# ------------------------------------------------- (d) how far each shares -
def p_family_reach(ax, D, P=False):
    """How many of the four domains use each template family, and each class.

    The families are shared; the structural classes are not, because a
    structural hash carries its operator and an operator belongs to one domain.
    The distinction is the paper's, and this is where it is measured.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    c = _census(D)
    wide = c.pivot(index="domain", columns="template_family",
                   values="share").reindex(CENSUS_DOMAINS).fillna(0.0)
    labels, reach, cols = [], [], []
    for key, label, col in FAMILY_STYLE:
        if key in wide:
            labels.append(label.split()[0])
            reach.append(int((wide[key] > 0).sum()))
            cols.append(col)
    x = np.arange(len(labels) + 1)
    ax.bar(x[:-1], reach, width=0.62, color=cols, edgecolor="black",
           linewidth=2.0, zorder=3)
    ax.bar(x[-1:], [1], width=0.62, color=GREY, edgecolor="black",
           linewidth=2.0, hatch="//", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels + ["Struct.\nclass"], rotation=28, ha="right",
                       fontsize=tick - 4)
    ylab(ax, "Domains that share it", LAB - 5 if not P else LAB - 8)
    ax.set_ylim(0, 4.9)
    ax.set_yticks([0, 1, 2, 3, 4])
    dress(ax, tick)


GRID_REUSE = [("families", p_family_share, "(a) Template families"),
              ("classes", p_class_counts, "(b) Fragments and classes"),
              ("compression", p_compression, "(c) Compression"),
              ("reach", p_family_reach, "(d) Cross-domain sharing")]


# ===================================================================== #
# fig_sched -- cross-task scheduling overhead, all four panels in style #
# ===================================================================== #
#: The three panels read the frames _raw_fig11 derives from the engine run that
#: scripts/verify_cpp_repro.py reproduces. An earlier version read a
#: hand-written results/fig11_paper_run.csv instead, whose own header said the
#: engine run disagreed with it; the figure now follows the run, and so does
#: the text.
def _fig11(D, panel):
    return {"a": D.fig11a, "b": D.fig11b, "c": D.fig11c}[panel]


def p_sched_filter(ax, D, P=False):
    """What fraction of candidate fragments each stage of TARS lets through."""
    tick = TICK if not P else TICK_DENSE
    d = _fig11(D, "a")
    x = np.arange(len(d))
    ax.bar(x, d["value"], width=0.62, color=[BLUE, GREEN, ORANGE],
           edgecolor="black", linewidth=2.2, zorder=3)
    for xi, v in zip(x, d["value"]):
        ax.text(xi, v + 2.5, f"{v:.1f}", ha="center", va="bottom",
                fontsize=tick - 3, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([m.split()[0] for m in d["bucket"]], rotation=22,
                       ha="right", fontsize=tick - 3)
    ylab(ax, "Match ratio (%)", LAB - 4 if not P else LAB - 7)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    dress(ax, tick)


def p_sched_latency(ax, D, P=False):
    """Dispatch latency and how much of the array sits idle, against multi-core."""
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    d = _fig11(D, "b")
    lat = d[d["metric"] == "Scheduling Latency (ms)"].set_index("platform")["value"]
    idle = d[d["metric"] == "Idle Ratio (%)"].set_index("platform")["value"]
    order = ["Multi-core", "FUSE"]
    x = np.arange(2)
    w = 0.36
    # Idle is plotted on the latency axis at a fixed scale so both fit one
    # panel; the scale is chosen from the data and written in the legend.
    scale = 10 ** np.ceil(np.log10(max(idle[k] for k in order) / max(lat[k] for k in order)))
    ax.bar(x - w / 2, [lat[k] for k in order], width=w, color=BLUE,
           edgecolor="black", linewidth=2.0, zorder=3, label="Latency (ms)")
    ax.bar(x + w / 2, [idle[k] / scale for k in order], width=w, color=ORANGE,
           edgecolor="black", linewidth=2.0, zorder=3, label=f"Idle ratio (%/{scale:g})")
    top = max(max(lat[k] for k in order), max(idle[k] / scale for k in order))
    for xi, k in zip(x, order):
        ax.text(xi - w / 2, lat[k] + top * 0.02, f"{lat[k]:.3f}", ha="center",
                va="bottom", fontsize=tick - 3, fontweight="bold")
        ax.text(xi + w / 2, idle[k] / scale + top * 0.02, f"{idle[k]:.1f}", ha="center",
                va="bottom", fontsize=tick - 3, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Multi-core", "FUSE"], fontsize=tick - 1)
    ylab(ax, "Latency (ms)", LAB - 4 if not P else LAB - 7)
    ax.set_ylim(0, top * 1.45)
    dress(ax, tick)
    legend(ax, "upper right", leg - 4)


def p_sched_norm(ax, D, P=False):
    """The same quantities normalized to the multi-core baseline."""
    tick = TICK if not P else TICK_DENSE
    d = _fig11(D, "c")
    f = d[d["platform"] == "FUSE"]
    x = np.arange(len(f))
    ax.bar(x, f["value"], width=0.62, color=[BLUE, VERM, GREEN],
           edgecolor="black", linewidth=2.2, zorder=3)
    ratio_line(ax, 100.0)
    for xi, v in zip(x, f["value"]):
        ax.text(xi, v + 3.0, f"{v:.0f}", ha="center", va="bottom",
                fontsize=tick - 3, fontweight="bold")
    ax.set_xticks(x)
    labels = {"Scheduling latency": "Latency", "Idle ratio": "Idle",
              "Redundant reads": "Redundant reads"}
    ax.set_xticklabels([labels.get(m, m) for m in f["metric"]], rotation=22,
                       ha="right", fontsize=tick - 3)
    ylab(ax, "% of multi-core", LAB - 4 if not P else LAB - 7)
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 25, 50, 75, 100])
    dress(ax, tick)


GRID_SCHED = [("filter", p_sched_filter, "(a) TARS fragment filter"),
              ("latency", p_sched_latency, "(b) Latency and idle"),
              ("norm", p_sched_norm, "(c) Normalized to multi-core"),
              ("mechanisms", p_mechanisms, "(d) Mechanism counters")]


# ===================================================================== #
# fig_chain -- configuration reuse, then value reuse                    #
# ===================================================================== #
def _chain(D, attributable=None):
    c = D.chain
    if c is None:
        return None
    if attributable is None:
        return c
    return c[c["attributable"] == attributable]


def _chain_label(row):
    """Short enough for four of them to sit under a quarter-width panel."""
    name = {"AES-128": "AES", "LLM (same prompt)": "LLM same",
            "LLM (new prompt)": "LLM new"}.get(row["workload"], row["workload"])
    reqs = 1 if row["regime"] == "single request" else 2
    return "%s x%d" % (name, reqs)


def p_chain_steps(ax, D, P=False):
    """The two steps as absolute latency, for the cases where they are visible."""
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    c = _chain(D, True)
    x = np.arange(len(c))
    w = 0.27
    for dx, col, key, lab in ((-w, GREY, "degraded_ms", "Degraded"),
                              (0.0, ORANGE, "template_only_ms", "+ template reuse"),
                              (w, BLUE, "fuse_ms", "+ value reuse")):
        ax.bar(x + dx, c[key].values * 1000.0, width=w, color=col,
               edgecolor="black", linewidth=1.8, zorder=3, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([_chain_label(r) for _, r in c.iterrows()],
                       rotation=32, ha="right", fontsize=tick - 4)
    ylab(ax, r"Latency ($\mu$s)", LAB - 4 if not P else LAB - 7)
    ax.set_ylim(0, 105)
    dress(ax, tick)
    legend(ax, "upper left", leg - 5)


def p_chain_share(ax, D, P=False):
    """Which link the gain came from."""
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    c = _chain(D, True)
    x = np.arange(len(c))
    cfg = c["config_share"].values
    val = c["value_share"].values
    ax.bar(x, cfg, width=0.62, color=ORANGE, edgecolor="black", linewidth=1.8,
           zorder=3, label="Configuration")
    ax.bar(x, val, width=0.62, bottom=cfg, color=BLUE, edgecolor="black",
           linewidth=1.8, zorder=3, label="Value")
    for xi, (a, b) in enumerate(zip(cfg, val)):
        if a > 12:
            ax.text(xi, a / 2, f"{a:.0f}", ha="center", va="center",
                    fontsize=tick - 3, fontweight="bold")
        if b > 12:
            ax.text(xi, a + b / 2, f"{b:.0f}", ha="center", va="center",
                    fontsize=tick - 3, fontweight="bold", color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([_chain_label(r) for _, r in c.iterrows()],
                       rotation=32, ha="right", fontsize=tick - 4)
    ylab(ax, "Share of the gain (%)", LAB - 5 if not P else LAB - 8)
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 25, 50, 75, 100])
    dress(ax, tick)
    legend(ax, "upper center", leg - 5, ncol=2)


def p_chain_config(ax, D, P=False):
    """Link one: the reconfiguration cycles template reuse removes."""
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    c = _chain(D)
    x = np.arange(len(c))
    w = 0.38
    ax.bar(x - w / 2, c["degraded_switch"].values / 1e3, width=w, color=GREY,
           edgecolor="black", linewidth=1.8, zorder=3, label="Degraded")
    ax.bar(x + w / 2, c["template_only_switch"].values / 1e3, width=w, color=ORANGE,
           edgecolor="black", linewidth=1.8, zorder=3, label="+ template reuse")
    ax.set_yscale("log")
    ax.set_ylim(1, 2e3)
    ax.set_xticks(x)
    ax.set_xticklabels([_chain_label(r) for _, r in c.iterrows()],
                       rotation=32, ha="right", fontsize=tick - 5)
    ylab(ax, "Reconfig. cycles (k)", LAB - 5 if not P else LAB - 8)
    dress(ax, tick)
    legend(ax, "lower left", leg - 5)


def p_chain_cache(ax, D, P=False):
    """Link two: value reuse needs a cache hit, and a hit needs a repeat."""
    tick = TICK if not P else TICK_DENSE
    c = _chain(D)
    x = np.arange(len(c))
    hit = c["fuse_value"].values
    cols = [BLUE if v > 0 else GREY for v in hit]
    ax.bar(x, hit, width=0.62, color=cols, edgecolor="black", linewidth=1.8, zorder=3)
    for xi, v in zip(x, hit):
        if v <= 1e-9:
            ax.text(xi, 2.0, "0", ha="center", va="bottom", fontsize=tick - 4,
                    fontweight="bold", color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels([_chain_label(r) for _, r in c.iterrows()],
                       rotation=32, ha="right", fontsize=tick - 5)
    ylab(ax, "Fragments from cache (%)", LAB - 6 if not P else LAB - 9)
    ax.set_ylim(0, 66)
    dress(ax, tick)


GRID_CHAIN = [("steps", p_chain_steps, "(a) Ablation arms"),
              ("share", p_chain_share, "(b) Where the gain comes from"),
              ("config", p_chain_config, "(c) Reconfiguration cycles")]


# ===================================================================== #
# fig_mixed -- the mixed-task trace across scheduling windows           #
# ===================================================================== #
MIXED_WINDOWS = (1, 2, 4, 8)


def _mixed(root):
    """Makespan and array occupancy per scheduling window, both modes."""
    rows = []
    for w in MIXED_WINDOWS:
        rec = {"window": w}
        for tag, mode in (("isomer", "morph"), ("baseline", "multicore")):
            path = root / "results" / "raw" / f"fig6_window{w}_{mode}_metrics.csv"
            if not path.exists():
                return None
            m = dict(zip(*[pd.read_csv(path)[c] for c in ("key", "value")]))
            rec[f"{tag}_ms"] = float(m["makespan_ms"])
            rec[f"{tag}_util"] = 100.0 * float(m["rfu_utilization"])
            rec[f"{tag}_reuse"] = 100.0 * float(m["result_reuse_rate"])
        rows.append(rec)
    return pd.DataFrame(rows)


def p_mixed(ax, D, P=False):
    """Makespan and occupancy as the scheduling window grows.

    The two designs track each other on both axes; where they part is reuse,
    which the right-hand panel of this appendix reports.
    """
    tick, leg = (TICK, LEG) if not P else (TICK_DENSE, LEG_DENSE)
    f = _mixed(D.root)
    x = np.arange(len(f))
    w = 0.38
    ax.bar(x - w / 2, f["baseline_ms"], width=w, color=GREY, edgecolor="black",
           linewidth=2.0, zorder=3, label="Multi-core makespan")
    ax.bar(x + w / 2, f["isomer_ms"], width=w, color=BLUE, edgecolor="black",
           linewidth=2.0, zorder=3, label="FUSE makespan")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in f["window"]], fontsize=tick)
    xlab(ax, "Scheduling window (ms)", LAB - 4 if not P else LAB - 7)
    ylab(ax, "Makespan (ms)", LAB - 3 if not P else LAB - 6)
    ax.set_ylim(0, 86)
    dress(ax, tick)

    rhs = ax.twinx()
    line(rhs, x, f["baseline_util"], VERM, "s", ls="--", label="Multi-core util.")
    line(rhs, x, f["isomer_util"], ORANGE, "P", label="FUSE util.")
    rhs.set_ylim(0, 62)
    rhs.set_ylabel("Array occupancy (%)", fontsize=(LAB - 4 if not P else LAB - 7),
                   fontweight="bold")
    for lbl in rhs.get_yticklabels():
        lbl.set_fontweight("bold")
    rhs.tick_params(axis="y", which="major", width=2.6, length=8, labelsize=tick)
    rhs.spines["top"].set_visible(False)
    for sp in rhs.spines.values():
        sp.set_linewidth(3.0)

    handles = ax.get_legend_handles_labels()[0] + rhs.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + rhs.get_legend_handles_labels()[1]
    ax.legend(handles, labels, frameon=True, framealpha=0.96, edgecolor="#666",
              loc="upper left", ncol=2, prop={"weight": "bold", "size": leg - 5},
              handlelength=1.5, borderpad=0.3, labelspacing=0.25,
              handletextpad=0.45, columnspacing=0.8)


def save(fig, out_dir: Path, name: str):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_dir / f"{name}.pdf")


def build(grid, D, out_dir: Path, name: str):
    fig, axes = plt.subplots(2, 3, figsize=GRID6)
    for (tag, fn, title), ax in zip(grid, axes.ravel()):
        ax.set_title(title, loc="left", fontsize=PANEL, fontweight="bold", pad=9)
        fn(ax, D, P=True)
    save(fig, out_dir, name)
    for tag, fn, _ in grid:
        f_, ax = plt.subplots(figsize=(5.2, 4.4))
        fn(ax, D, P=False)
        save(f_, out_dir, f"{name}_{tag}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--out-dir", type=Path, default=root / "sosp")
    parser.add_argument("--only",
                        choices=["domains", "secure", "concurrent", "fig11d",
                                 "ablation", "reuse", "sched", "chain",
                                 "mixed"],
                        default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    D = Data(args.root)
    if args.only in (None, "domains"):
        if D.fig4 is None or D.zoo is None:
            print("! need results/raw/fig4_raw.csv and runs/llm_zoo/llm_summary.csv")
        else:
            build(GRID_DOMAINS, D, args.out_dir, "fig_fuse_domains")
    if args.only in (None, "secure"):
        missing = [n for n, v in (("concurrent", D.conc), ("scaling", D.scaling),
                                  ("frontend", D.frontend), ("answers", D.answers))
                   if v is None]
        if missing:
            print(f"! missing for the secure grid: {missing}")
        else:
            build(GRID_SECURE, D, args.out_dir, "fig_llm_secure")
    if args.only in (None, "concurrent"):
        if D.conc is None or D.shapes is None:
            print("! need runs/concurrent/{concurrent_results,request_shapes}.csv")
        else:
            build(GRID_CONCURRENT, D, args.out_dir, "fig_concurrent")
    if args.only in (None, "mixed"):
        if _mixed(D.root) is None:
            print("! need results/raw/fig6_window*_metrics.csv")
        else:
            f_, ax = plt.subplots(figsize=(6.6, 4.2))
            p_mixed(ax, D, P=False)
            save(f_, args.out_dir, "fig_mixed")
    if args.only in (None, "chain"):
        if D.chain is None:
            print("! need runs/census/reuse_breakdown.csv "
                  "(scripts/run_reuse_breakdown.py)")
        else:
            fig, axes = plt.subplots(1, 3, figsize=(14.1, 4.7))
            for (tag, fn, title), ax in zip(GRID_CHAIN, axes.ravel()):
                ax.set_title(title, loc="left", fontsize=PANEL,
                             fontweight="bold", pad=9)
                fn(ax, D, P=True)
            save(fig, args.out_dir, "fig_chain")
            for tag, fn, _ in GRID_CHAIN:
                f_, ax = plt.subplots(figsize=(5.2, 4.4))
                fn(ax, D, P=False)
                save(f_, args.out_dir, f"fig_chain_{tag}")
    if args.only in (None, "reuse"):
        if D.census is None:
            print("! need runs/census/template_census.csv "
                  "(scripts/run_template_census.py)")
        else:
            fig, axes = plt.subplots(1, 4, figsize=(14.1, 4.7))
            for (tag, fn, title), ax in zip(GRID_REUSE, axes.ravel()):
                ax.set_title(title, loc="left", fontsize=PANEL,
                             fontweight="bold", pad=9)
                fn(ax, D, P=True)
            save(fig, args.out_dir, "fig_reuse")
            for tag, fn, _ in GRID_REUSE:
                f_, ax = plt.subplots(figsize=(5.2, 4.4))
                fn(ax, D, P=False)
                save(f_, args.out_dir, f"fig_reuse_{tag}")
    if args.only in (None, "ablation"):
        if _ablation(D) is None or D.granularity is None:
            print("! need results/raw/fig9_*, runs/llm_zoo/llm_ablation.csv "
                  "and runs/census/granularity.csv")
        else:
            build(GRID_ABLATION, D, args.out_dir, "fig_ablation")
    if args.only in (None, "sched"):
        if _pair(D.root) is None or any(f is None for f in (D.fig11a, D.fig11b, D.fig11c)):
            print("! need results/fig11{a_sched,b_switch,c_summary}.csv and "
                  "results/raw/fig11_{morph,multicore}_metrics.csv")
        else:
            fig, axes = plt.subplots(1, 4, figsize=(14.1, 4.7))
            for (tag, fn, title), ax in zip(GRID_SCHED, axes.ravel()):
                ax.set_title(title, loc="left", fontsize=PANEL,
                             fontweight="bold", pad=9)
                fn(ax, D, P=True)
            save(fig, args.out_dir, "fig_sched")
            for tag, fn, _ in GRID_SCHED:
                f_, ax = plt.subplots(figsize=(5.2, 4.4))
                fn(ax, D, P=False)
                save(f_, args.out_dir, f"fig_sched_{tag}")


if __name__ == "__main__":
    main()
