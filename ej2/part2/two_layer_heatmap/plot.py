"""EJ2 part2 — Plots para two_layer_heatmap.

Lee results.pkl (31 configs únicas × 5 seeds = 155 rows) y arma una
matriz 6x6 espejando los pares (0, n) ↔ (n, 0). Genera 4 heatmaps
(val/test/train acc + n_params) y un summary.txt.

Si querés cambiar el estilo del gráfico (cmap, anotaciones, escala),
solo editá este archivo y re-ejecutá: NO requiere re-correr run.py.
"""
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

SIZES = [0, 8, 16, 32, 64, 128, 256, 512]

OUT_DIR = os.path.join(RESULTS_PART2, "two_layer_heatmap")


def aggregate(rows, key):
    vals = [r[key] for r in rows]
    return float(np.mean(vals)), float(np.std(vals))


def heatmap(matrix, title, path, cmap="viridis", fmt=".3f",
            std_matrix=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(SIZES)))
    ax.set_yticks(range(len(SIZES)))
    ax.set_xticklabels(SIZES)
    ax.set_yticklabels(SIZES)
    ax.set_xlabel("Capa 2 (neuronas)  —  0 = no existe")
    ax.set_ylabel("Capa 1 (neuronas)  —  0 = no existe")
    ax.set_title(title)
    mid = (matrix.max() + matrix.min()) / 2
    for i in range(len(SIZES)):
        for j in range(len(SIZES)):
            v = matrix[i, j]
            color = "white" if v < mid else "black"
            label = f"{v:{fmt}}"
            if std_matrix is not None:
                label += f"\n±{std_matrix[i,j]:{fmt}}"
            ax.text(j, i, label, ha="center", va="center",
                    color=color, fontsize=8)
    plt.colorbar(im, ax=ax)
    save_fig(fig, path)


def main():
    pkl_path = os.path.join(OUT_DIR, "results.pkl")
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    # Build canonical (n1, n2) → list of rows. Mirror (0, n) ↔ (n, 0).
    by_pair = {}
    for r in results:
        n1, n2 = r["n1"], r["n2"]
        by_pair.setdefault((n1, n2), []).append(r)
        # Mirror one-hidden-layer cases: (0, n) is the same arch as (n, 0).
        if n1 == 0 and n2 > 0:
            by_pair.setdefault((n2, 0), []).append(r)
        elif n2 == 0 and n1 > 0:
            by_pair.setdefault((0, n1), []).append(r)

    # Build matrices
    n = len(SIZES)
    val_m = np.zeros((n, n)); val_s = np.zeros((n, n))
    test_m = np.zeros((n, n)); test_s = np.zeros((n, n))
    train_m = np.zeros((n, n))
    gap_m = np.zeros((n, n))
    n_params_mat = np.zeros((n, n), dtype=int)
    epochs_m = np.zeros((n, n))
    time_m = np.zeros((n, n))

    for i, n1 in enumerate(SIZES):
        for j, n2 in enumerate(SIZES):
            rows = by_pair.get((n1, n2), [])
            if not rows:
                continue
            m, s = aggregate(rows, "best_val_acc"); val_m[i, j] = m; val_s[i, j] = s
            m, s = aggregate(rows, "test_acc");     test_m[i, j] = m; test_s[i, j] = s
            m, _ = aggregate(rows, "train_acc");    train_m[i, j] = m
            gaps = [r["train_acc"] - r["best_val_acc"] for r in rows]
            gap_m[i, j] = float(np.mean(gaps))
            n_params_mat[i, j] = rows[0]["n_params"]
            m, _ = aggregate(rows, "stopped_at"); epochs_m[i, j] = m
            m, _ = aggregate(rows, "time_per_epoch"); time_m[i, j] = m

    # Plots
    heatmap(val_m, "EJ2 — Val accuracy: heatmap (capa 1 × capa 2)",
            os.path.join(OUT_DIR, "val_acc_heatmap.png"),
            cmap="viridis", std_matrix=val_s)
    heatmap(test_m, "EJ2 — Test accuracy: heatmap (capa 1 × capa 2)",
            os.path.join(OUT_DIR, "test_acc_heatmap.png"),
            cmap="viridis", std_matrix=test_s)
    heatmap(train_m, "EJ2 — Train accuracy: heatmap (capa 1 × capa 2)",
            os.path.join(OUT_DIR, "train_acc_heatmap.png"),
            cmap="magma")
    heatmap(gap_m, "EJ2 — Generalization gap (train − val): heatmap",
            os.path.join(OUT_DIR, "gap_heatmap.png"),
            cmap="coolwarm", fmt="+.3f")
    heatmap(n_params_mat.astype(float), "EJ2 — Cantidad de parámetros: heatmap",
            os.path.join(OUT_DIR, "n_params_heatmap.png"),
            cmap="cividis", fmt=".0f")
    heatmap(time_m, "EJ2 — Tiempo por época (s): heatmap",
            os.path.join(OUT_DIR, "time_per_epoch_heatmap.png"),
            cmap="plasma", fmt=".2f")

    # Summary
    summary = "EJ2 — Two-layer heatmap (n1, n2 ∈ {0, 8, 16, 32, 64, 128}):\n\n"
    summary += "Configuración: Adam lr=1e-3, batch=32, max_epochs=500, "
    summary += "patience=50, tanh, init_scale=0.1.\n"
    summary += "Notación: n=0 ⇒ capa no existe.  (0,n) y (n,0) son el mismo MLP.\n\n"
    summary += "Tabla val_acc (mean ± std sobre 5 seeds):\n"
    summary += "          " + "  ".join(f" n2={s:>3}    " for s in SIZES) + "\n"
    for i, n1 in enumerate(SIZES):
        line = f"n1={n1:>3}    "
        for j, _ in enumerate(SIZES):
            line += f"{val_m[i,j]:.3f}±{val_s[i,j]:.3f} "
        summary += line + "\n"

    summary += "\nTabla test_acc (mean ± std):\n"
    summary += "          " + "  ".join(f" n2={s:>3}    " for s in SIZES) + "\n"
    for i, n1 in enumerate(SIZES):
        line = f"n1={n1:>3}    "
        for j, _ in enumerate(SIZES):
            line += f"{test_m[i,j]:.3f}±{test_s[i,j]:.3f} "
        summary += line + "\n"

    best_val = np.unravel_index(np.argmax(val_m), val_m.shape)
    best_test = np.unravel_index(np.argmax(test_m), test_m.shape)
    summary += (f"\nMejor val_acc: n1={SIZES[best_val[0]]}, n2={SIZES[best_val[1]]}, "
                f"val={val_m[best_val]:.4f}, test={test_m[best_val]:.4f}, "
                f"params={n_params_mat[best_val]:,}\n")
    summary += (f"Mejor test_acc: n1={SIZES[best_test[0]]}, n2={SIZES[best_test[1]]}, "
                f"val={val_m[best_test]:.4f}, test={test_m[best_test]:.4f}, "
                f"params={n_params_mat[best_test]:,}\n")

    summary_path = os.path.join(OUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(summary)
    print(f"\nResultados en {OUT_DIR}")


if __name__ == "__main__":
    main()
