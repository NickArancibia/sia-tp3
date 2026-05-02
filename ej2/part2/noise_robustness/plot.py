"""EJ2 part2 — Plot de robustez al ruido."""
import os
import pickle
import sys

PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART2_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(PART2_DIR)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS_PART2
from shared.utils import save_fig

OUT_DIR = os.path.join(RESULTS_PART2, "noise_robustness")


def main():
    with open(os.path.join(OUT_DIR, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    sigmas = np.array([r["sigma"] for r in results])
    means = np.array([r["mean_acc"] for r in results])
    stds = np.array([r["std_acc"] for r in results])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(sigmas, means, yerr=stds, fmt="o-", color="darkred",
                capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel("σ del ruido gaussiano agregado al test")
    ax.set_ylabel("Test accuracy")
    ax.set_title("EJ2 — Robustez al ruido (modelo seleccionado, 5 seeds de ruido)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)
    for s, m in zip(sigmas, means):
        ax.annotate(f"{m:.3f}", (s, m), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    save_fig(fig, os.path.join(OUT_DIR, "noise_curve.png"))

    # Tabla resumen
    summary = "EJ2 — Robustez al ruido gaussiano N(0, σ²) sobre test:\n\n"
    summary += "σ      | acc_mean | acc_std  | degradación vs σ=0\n"
    summary += "-" * 56 + "\n"
    base_acc = means[0]
    for s, m, sd in zip(sigmas, means, stds):
        deg = base_acc - m
        summary += f"{s:.2f}   | {m:.4f}   | {sd:.4f}   | -{deg:.4f}\n"
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
