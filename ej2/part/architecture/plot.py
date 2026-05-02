"""Bar chart de test accuracy por arquitectura, ordenado de mejor a peor."""
import os
import sys

EJ2_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ2_DIR)
sys.path.insert(0, EJ2_DIR)
sys.path.insert(0, REPO_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join(EJ2_DIR, "results", "part", "architecture")


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_arch_bars(summary_df, path):
    df = summary_df.sort_values("mean_test_acc", ascending=True)
    labels = df["config_name"].tolist()
    means = df["mean_test_acc"].to_numpy()
    stds = df["std_test_acc"].to_numpy()
    n_params = df["n_params"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(labels)))
    bars = ax.barh(labels, means, xerr=stds, capsize=5, color=colors,
                   alpha=0.85, error_kw={"linewidth": 1.5})

    for bar, m, std, np_ in zip(bars, means, stds, n_params):
        ax.text(m + std + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{m:.4f} ± {std:.4f}  ({int(np_):,} params)",
                va="center", fontsize=8)

    ax.set_xlabel("Test accuracy")
    ax.set_title("Test accuracy por arquitectura (media ± std, 5 semillas)\nOrdenado de peor a mejor")
    margin = max(stds) * 3 + 0.02
    ax.set_xlim(max(0, means.min() - margin), min(1.02, means.max() + margin + 0.12))
    fig.tight_layout()
    _save(fig, path)
    print(f"Guardado: {path}")


def main():
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    if not os.path.exists(summary_path):
        print(f"ERROR: ejecutar run.py primero (falta {summary_path})")
        return

    summary_df = pd.read_csv(summary_path)
    summary_df["n_params"] = summary_df["n_params"].astype(int)

    plot_arch_bars(summary_df, path=os.path.join(RESULTS_DIR, "test_acc_bar.png"))

    best = summary_df.loc[summary_df["mean_test_acc"].idxmax()]
    print(f"\nMejor arquitectura:")
    print(f"  {best['config_name']}: {best['architecture']}")
    print(f"  test_acc = {best['mean_test_acc']:.4f} ± {best['std_test_acc']:.4f}")
    print(f"  params   = {int(best['n_params']):,}")


if __name__ == "__main__":
    main()
