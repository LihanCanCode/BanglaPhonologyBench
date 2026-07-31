# -*- coding: utf-8 -*-
"""Figure 2: distribution of (GTAD, STAD_bn, rho) across tokenizers on the
top-3000 wordlist, plus the violation-type decomposition (Spec C.6.1).

Input : results/metrics_top3000.csv  (from scripts/compute_metrics.py)
Output: figures/figure2.png (300 dpi) and figures/figure2.pdf
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- palette (validated reference categorical order; light mode) -------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # slots 1-5
ORDINAL3 = ["#86b6ef", "#2a78d6", "#104281"]   # blue seq 250/450/650 (ordered)
INK, MUTED, GRID, BASE = "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SURFACE = "#fcfcfb"

TOKENIZER_ORDER = ["llama3", "gpt2", "byt5", "tigerllm", "banglat5"]
LABELS = {"llama3": "Llama-3.1", "gpt2": "GPT-2", "byt5": "ByT5",
          "tigerllm": "TigerLLM-9B", "banglat5": "BanglaT5"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": BASE, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
})


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(BASE)
    ax.tick_params(length=0)


def ecdf(ax, values, color, label):
    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)
    ax.step(np.concatenate([[0], x, [1]]), np.concatenate([[0], y, [1]]),
            where="post", color=color, lw=1.6, label=label)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else "results/metrics_top3000.csv"
    df = pd.read_csv(csv)
    df = df[~df["quarantined"].astype(bool)]
    order = [t for t in TOKENIZER_ORDER if t in set(df["tokenizer"])]

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9))
    (ax_g, ax_s, ax_r, ax_d) = axes

    # (a) GTAD ECDF, (b) STAD ECDF
    for ax, col, title in ((ax_g, "gtad", "(a) GTAD"),
                           (ax_s, "stad", "(b) STAD$_{bn}$")):
        for i, tk in enumerate(order):
            vals = df.loc[df["tokenizer"] == tk, col].dropna()
            ecdf(ax, vals.to_numpy(), SERIES[i], LABELS[tk])
        ax.set_title(title, loc="left")
        ax.set_xlabel(col.upper() if col == "gtad" else "STAD$_{bn}$")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1.02)
        style(ax)
    ax_g.set_ylabel("cumulative share of words")
    ax_g.legend(frameon=False, fontsize=7.5, loc="lower right",
                handlelength=1.4, borderaxespad=0.2)

    # (c) rho — property of the script, identical across tokenizers
    rho = df[df["tokenizer"] == order[0]].set_index("word")["rho"].dropna()
    bins = np.linspace(0, 1, 11)
    ax_r.hist(rho, bins=bins, color=SERIES[0], edgecolor=SURFACE, lw=0.8,
              weights=np.full(len(rho), 1 / len(rho)))
    ax_r.set_title(r"(c) $\rho$  (script-intrinsic)", loc="left")
    ax_r.set_xlabel(r"$\rho$ per word"); ax_r.set_ylabel("share of words")
    ax_r.set_xlim(-0.02, 1.02)
    share_pos = float((rho > 0).mean())
    ax_r.annotate(f"{share_pos:.0%} of words have\n" r"$\rho>0$ (unrepresentable"
                  "\nsyllable boundaries)",
                  xy=(0.98, 0.55), xycoords="axes fraction",
                  ha="right", va="center", fontsize=7.5, color=INK)
    style(ax_r)

    # (d) violation-type decomposition: bar height = mean GTAD, split by type
    byte_r, matra_r, conj_r, n_bounds = [], [], [], []
    for tk in order:
        d = df[df["tokenizer"] == tk]
        nb = max(int(d["n_boundaries"].sum()), 1)
        n_bounds.append(int(d["n_boundaries"].sum()))
        byte_r.append(d["gtad_byte"].sum() / nb)
        matra_r.append(d["gtad_matra"].sum() / nb)
        conj_r.append(d["gtad_conjunct"].sum() / nb)
    x = np.arange(len(order))
    b0 = np.zeros(len(order))
    for vals, color, lab in ((byte_r, ORDINAL3[0], "byte-internal"),
                             (matra_r, ORDINAL3[1], "matra/diacritic"),
                             (conj_r, ORDINAL3[2], "conjunct")):
        ax_d.bar(x, vals, 0.62, bottom=b0, color=color, label=lab,
                 edgecolor=SURFACE, linewidth=1.2)
        b0 += np.asarray(vals)
    for xi, total, nb in zip(x, b0, n_bounds):
        ax_d.annotate(f"{total:.2f}", xy=(xi, total + 0.015), ha="center",
                      fontsize=7.5, color=INK)
        ax_d.annotate(f"n={nb:,}", xy=(xi, 0.015), ha="center", va="bottom",
                      fontsize=6.5, color=MUTED)
    ax_d.set_title("(d) violating boundaries by type", loc="left")
    ax_d.set_ylabel("share of token boundaries")
    ax_d.set_xticks(x, [LABELS[t] for t in order], rotation=30, ha="right",
                    fontsize=7.5)
    ax_d.set_ylim(0, 1.06)
    ax_d.legend(frameon=False, fontsize=7.5, loc="upper right",
                handlelength=1.0, borderaxespad=0.2)
    style(ax_d)

    fig.suptitle("")
    fig.tight_layout(w_pad=1.6)
    out = Path("figures"); out.mkdir(exist_ok=True)
    fig.savefig(out / "figure2.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "figure2.pdf", bbox_inches="tight")
    print(f"wrote {out/'figure2.png'} and .pdf")

    # console summary for the paper text
    print("\nper-tokenizer means on top-3000:")
    print(df.groupby("tokenizer")[["gtad", "stad", "rho"]].mean()
            .reindex(order).round(3))


if __name__ == "__main__":
    main()
