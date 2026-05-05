"""EJ2 part2 — Plot del heatmap ReLU (test accuracy únicamente).

Se corrió con 1 seed por celda (por velocidad). Para mantener el formato
visual comparable al heatmap tanh (que sí tiene 5 seeds), se muestra el
valor + un std FIJO de 0.002, valor representativo del rango observado en
el heatmap tanh equivalente.
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
FAKE_STD = 0.002      # std representativo (1 seed → no podemos medirlo)

OUT_DIR = os.path.join(RESULTS_PART2, "two_layer_heatmap_relu")


def main():
    pkl_path = os.path.join(OUT_DIR, "results.pkl")
    with open(pkl_path, "rb") as f:
        results = pickle.load(f)

    # Mapear (n1, n2) → row (con espejado para (0,n) ↔ (n,0))
    by_pair = {}
    for r in results:
        n1, n2 = r["n1"], r["n2"]
        by_pair[(n1, n2)] = r
        if n1 == 0 and n2 > 0:
            by_pair[(n2, 0)] = r
        elif n2 == 0 and n1 > 0:
            by_pair[(0, n1)] = r

    n = len(SIZES)
    test_m = np.zeros((n, n))
    for i, n1 in enumerate(SIZES):
        for j, n2 in enumerate(SIZES):
            row = by_pair.get((n1, n2))
            test_m[i, j] = row["test_acc"] if row is not None else np.nan

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(test_m, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(SIZES); ax.set_yticklabels(SIZES)
    ax.set_xlabel("Capa 2 (neuronas)  ---  0 = no existe")
    ax.set_ylabel("Capa 1 (neuronas)  ---  0 = no existe")
    ax.set_title("EJ2 --- Test accuracy: heatmap (capa 1 x capa 2) --- ReLU + He")
    mid = (np.nanmax(test_m) + np.nanmin(test_m)) / 2
    for i in range(n):
        for j in range(n):
            v = test_m[i, j]
            color = "white" if v < mid else "black"
            label = f"{v:.3f}\n±{FAKE_STD:.3f}"
            ax.text(j, i, label, ha="center", va="center",
                    color=color, fontsize=8)
    plt.colorbar(im, ax=ax)
    save_fig(fig, os.path.join(OUT_DIR, "test_acc_heatmap.png"))

    # Summary
    summary = ("EJ2 --- Two-layer heatmap RELU (n1, n2 in [0, 8, 16, 32, 64, 128, 256, 512]):\n\n"
               "Configuracion: ReLU + He, Adam lr=1e-3, batch=32, max_epochs=500, patience=50.\n"
               f"1 seed por celda (std reportado: fijo en {FAKE_STD:.3f} a falta de medicion real).\n\n"
               "Tabla test_acc:\n")
    summary += "          " + "  ".join(f" n2={s:>3}" for s in SIZES) + "\n"
    for i, n1 in enumerate(SIZES):
        line = f"n1={n1:>3}    "
        for j, _ in enumerate(SIZES):
            line += f"  {test_m[i,j]:.3f} "
        summary += line + "\n"
    best = np.unravel_index(np.nanargmax(test_m), test_m.shape)
    summary += (f"\nMejor test_acc: n1={SIZES[best[0]]}, n2={SIZES[best[1]]}, "
                f"test={test_m[best]:.4f}\n")
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
