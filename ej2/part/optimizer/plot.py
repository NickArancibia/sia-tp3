"""Gráficos del experimento 2: optimizador × learning rate.

Produce:
  - heatmap val_acc: optimizer (filas) × lr (columnas)
  - curvas de convergencia: un panel por optimizador, una curva por lr
"""
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

RESULTS_DIR = os.path.join(EJ2_DIR, "results", "part", "optimizer")

OPT_ORDER = ["gd", "momentum", "adam"]
LR_COLORS = {0.1: "#4e79a7", 0.01: "#f28e2b", 0.001: "#e15759", 0.0001: "#76b7b2"}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_heatmap(df, opt_order, lr_order, path):
    col_labels = [f"{lr:.0e}" for lr in lr_order]
    grid_mean = np.full((len(opt_order), len(lr_order)), np.nan)
    grid_std = np.full((len(opt_order), len(lr_order)), np.nan)

    for i, opt in enumerate(opt_order):
        for j, lr in enumerate(lr_order):
            mask = (df["optimizer"] == opt) & (df["learning_rate"].round(12) == round(lr, 12))
            if mask.any():
                grid_mean[i, j] = float(df.loc[mask, "mean_val_acc"].iloc[0])
                grid_std[i, j] = float(df.loc[mask, "std_val_acc"].iloc[0])

    fig, ax = plt.subplots(figsize=(1.8 * len(lr_order) + 2, 0.9 * len(opt_order) + 2))
    im = ax.imshow(grid_mean, cmap="viridis", aspect="auto",
                   vmin=float(np.nanmin(grid_mean)), vmax=float(np.nanmax(grid_mean)))

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(len(opt_order)))
    ax.set_yticklabels(opt_order, fontsize=10)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Optimizador")
    ax.set_title("Val accuracy — optimizador × learning rate (batch=32, 5 semillas)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Val accuracy")

    thresh = float(np.nanmean(grid_mean))
    for i in range(grid_mean.shape[0]):
        for j in range(grid_mean.shape[1]):
            if np.isnan(grid_mean[i, j]):
                continue
            color = "black" if grid_mean[i, j] > thresh else "white"
            ax.text(j, i, f"{grid_mean[i,j]:.4f}\n±{grid_std[i,j]:.4f}",
                    ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    _save(fig, path)
    print(f"Guardado: {path}")


def plot_convergence_single(curves_df, opt, lr_order, out_dir):
    fig, ax = plt.subplots(figsize=(7, 5))

    for lr in lr_order:
        cname = f"{opt}_lr{lr:.0e}"
        sub = curves_df[curves_df["config_name"] == cname]
        if sub.empty:
            continue

        grouped = sub.groupby("epoch")["val_loss"].agg(["mean", "std"]).reset_index()
        epochs = grouped["epoch"].values
        means = grouped["mean"].values
        stds = grouped["std"].values

        color = LR_COLORS.get(lr, "gray")
        ax.plot(epochs, means, linestyle="-", color=color, label=f"lr={lr:.0e}")
        ax.fill_between(epochs, means - stds, means + stds, alpha=0.15, color=color)

    ax.set_title(f"Convergencia — {opt} (batch=32)")
    ax.set_xlabel("Época")
    ax.set_ylabel("Val MSE (media ± std)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(out_dir, f"convergence_{opt}.png")
    _save(fig, path)
    print(f"Guardado: {path}")


def main():
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    curves_path = os.path.join(RESULTS_DIR, "curves.csv")
    if not os.path.exists(summary_path):
        print(f"ERROR: ejecutar run.py primero (falta {summary_path})")
        return

    df = pd.read_csv(summary_path)
    df["learning_rate"] = df["learning_rate"].astype(float)

    lr_order = sorted(df["learning_rate"].unique())
    opt_order = [o for o in OPT_ORDER if o in df["optimizer"].values]

    plot_heatmap(df, opt_order, lr_order,
                 path=os.path.join(RESULTS_DIR, "heatmap_val_acc.png"))

    if os.path.exists(curves_path):
        curves_df = pd.read_csv(curves_path)
        for opt in opt_order:
            plot_convergence_single(curves_df, opt, lr_order, RESULTS_DIR)

    print("\nMejor config por optimizador:")
    for opt in opt_order:
        sub = df[df["optimizer"] == opt]
        best = sub.loc[sub["mean_val_acc"].idxmax()]
        print(f"  {opt}: lr={best['learning_rate']:.0e}  "
              f"val_acc={best['mean_val_acc']:.4f} ± {best['std_val_acc']:.4f}")

    best_overall = df.loc[df["mean_val_acc"].idxmax()]
    print(f"\nMejor global: {best_overall['config_name']}  "
          f"val_acc={best_overall['mean_val_acc']:.4f} ± {best_overall['std_val_acc']:.4f}")


if __name__ == "__main__":
    main()
