"""Plot heatmap batch_size × LR con accuracy de validación (media ± std)
y gráficos de tiempo de ejecución."""
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

RESULTS_DIR = os.path.join(EJ2_DIR, "results", "part", "batch_lr")

BATCH_ORDER = ["online", "mini32", "mini64", "mini128", "full"]
BATCH_COLORS = {
    "online":   "#e15759",
    "mini32":   "#4e79a7",
    "mini64":   "#76b7b2",
    "mini128":  "#f28e2b",
    "full":     "#59a14f",
}


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _build_grid(df, value_col, row_labels, lr_order):
    grid = np.full((len(row_labels), len(lr_order)), np.nan)
    for i, bl in enumerate(row_labels):
        for j, lr in enumerate(lr_order):
            mask = (df["batch_label"] == bl) & (df["learning_rate"].round(12) == round(lr, 12))
            if mask.any():
                grid[i, j] = float(df.loc[mask, value_col].iloc[0])
    return grid


def plot_heatmap(values, stds, row_labels, col_labels, title, cbar_label, path, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(1.8 * len(col_labels) + 2, 0.9 * len(row_labels) + 2))
    im = ax.imshow(values, cmap=cmap, aspect="auto",
                   vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Batch size")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    thresh = float(np.nanmean(values))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                continue
            color = "black" if values[i, j] > thresh else "white"
            if values.max() <= 1.0:
                val_fmt = f"{values[i,j]:.4f}"
                std_txt = f"\n±{stds[i,j]:.4f}" if stds is not None else ""
            else:
                val_fmt = f"{values[i,j]:.1f}s"
                std_txt = f"\n±{stds[i,j]:.1f}s" if stds is not None else ""
            ax.text(j, i, f"{val_fmt}{std_txt}",
                    ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    _save(fig, path)
    print(f"Guardado: {path}")


def plot_time_bars(df, row_labels, lr_order, path):
    """Bar chart agrupado: eje x = batch strategy, barras = LR, altura = tiempo medio."""
    lr_labels = [f"{lr:.0e}" for lr in lr_order]
    x = np.arange(len(row_labels))
    n_lrs = len(lr_order)
    width = 0.8 / n_lrs
    offsets = np.linspace(-(n_lrs - 1) / 2, (n_lrs - 1) / 2, n_lrs) * width

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, n_lrs))

    for j, (lr, lr_lbl, color, offset) in enumerate(zip(lr_order, lr_labels, colors, offsets)):
        means, stds = [], []
        for bl in row_labels:
            mask = (df["batch_label"] == bl) & (df["learning_rate"].round(12) == round(lr, 12))
            if mask.any():
                means.append(float(df.loc[mask, "mean_elapsed_s"].iloc[0]))
                stds.append(float(df.loc[mask, "std_elapsed_s"].iloc[0]))
            else:
                means.append(0.0)
                stds.append(0.0)
        bars = ax.bar(x + offset, means, width, yerr=stds, label=f"lr={lr_lbl}",
                      color=color, alpha=0.85, capsize=4, error_kw={"linewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(row_labels, fontsize=11)
    ax.set_xlabel("Estrategia de batch")
    ax.set_ylabel("Tiempo de entrenamiento (s)")
    ax.set_title("Tiempo de entrenamiento por batch size y learning rate (GD, 5 semillas, media ± std)")
    ax.legend(title="Learning rate", fontsize=9)
    fig.tight_layout()
    _save(fig, path)
    print(f"Guardado: {path}")


def plot_pareto_scatter(df, row_labels, path):
    """Scatter accuracy vs tiempo con frontera de Pareto marcada."""
    fig, ax = plt.subplots(figsize=(9, 6))

    pareto_candidates = []

    for bl in row_labels:
        sub = df[df["batch_label"] == bl]
        color = BATCH_COLORS.get(bl, "gray")
        for _, row in sub.iterrows():
            x = row["mean_elapsed_s"]
            y = row["mean_val_acc"]
            xe = row["std_elapsed_s"]
            ye = row["std_val_acc"]
            lr_str = f"lr={row['learning_rate']:.0e}"
            ax.errorbar(x, y, xerr=xe, yerr=ye,
                        fmt="o", color=color, alpha=0.85,
                        capsize=4, markersize=7)
            ax.annotate(lr_str, (x, y),
                        textcoords="offset points", xytext=(6, 3),
                        fontsize=7, color=color)
            pareto_candidates.append((x, y, bl, lr_str))

    # Frontera de Pareto (mejor accuracy con menor tiempo)
    pareto = []
    sorted_by_time = sorted(pareto_candidates, key=lambda p: p[0])
    best_acc = -1.0
    for pt in sorted_by_time:
        if pt[1] >= best_acc:
            best_acc = pt[1]
            pareto.append(pt)
    if len(pareto) >= 2:
        px = [p[0] for p in pareto]
        py = [p[1] for p in pareto]
        ax.plot(px, py, "--", color="gray", linewidth=1.2,
                alpha=0.6, label="Frontera de Pareto")

    # Leyenda por batch strategy
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BATCH_COLORS[bl],
               markersize=9, label=bl)
        for bl in row_labels if bl in BATCH_COLORS
    ]
    legend_elements.append(
        Line2D([0], [0], linestyle="--", color="gray", alpha=0.6, label="Frontera de Pareto")
    )
    ax.legend(handles=legend_elements, fontsize=9)

    ax.set_xlabel("Tiempo medio de entrenamiento (s)")
    ax.set_ylabel("Val accuracy media")
    ax.set_title("Accuracy vs tiempo — batch size × learning rate\n(cada punto = una config, barras = ±std)")
    fig.tight_layout()
    _save(fig, path)
    print(f"Guardado: {path}")


def main():
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    if not os.path.exists(summary_path):
        print(f"ERROR: no existe {summary_path}. Ejecutar run.py primero.")
        return

    df = pd.read_csv(summary_path)
    df["batch_size"] = df["batch_size"].astype(int)
    df["learning_rate"] = df["learning_rate"].astype(float)

    lr_order = sorted(df["learning_rate"].unique())
    row_labels = [b for b in BATCH_ORDER if b in df["batch_label"].values]
    col_labels = [f"{lr:.0e}" for lr in lr_order]

    # Heatmap de accuracy
    mean_acc = _build_grid(df, "mean_val_acc", row_labels, lr_order)
    std_acc = _build_grid(df, "std_val_acc", row_labels, lr_order)
    plot_heatmap(
        mean_acc, std_acc, row_labels, col_labels,
        title="Val accuracy — batch size × learning rate (GD, 5 semillas)",
        cbar_label="Val accuracy",
        path=os.path.join(RESULTS_DIR, "heatmap_val_acc.png"),
        cmap="viridis",
    )

    if "mean_elapsed_s" in df.columns:
        plot_time_bars(df, row_labels, lr_order,
                       path=os.path.join(RESULTS_DIR, "bar_time.png"))
        plot_pareto_scatter(df, row_labels,
                            path=os.path.join(RESULTS_DIR, "pareto_acc_vs_time.png"))

    best_idx = np.unravel_index(np.nanargmax(mean_acc), mean_acc.shape)
    print(f"\nMejor config: batch={row_labels[best_idx[0]]}, "
          f"lr={col_labels[best_idx[1]]}  "
          f"val_acc={mean_acc[best_idx]:.4f} ± {std_acc[best_idx]:.4f}")


if __name__ == "__main__":
    main()
