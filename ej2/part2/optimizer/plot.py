"""EJ2 part2 — Plots para variantes de optimizador."""
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
from shared.utils import (plot_multi_bar, plot_multi_learning_curves, save_fig)

OUT_DIR = os.path.join(RESULTS_PART2, "optimizer")


def main():
    with open(os.path.join(OUT_DIR, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    opt_order = []
    seen = set()
    for r in results:
        if r["opt_name"] not in seen:
            opt_order.append(r["opt_name"])
            seen.add(r["opt_name"])

    val_acc_curves = {}
    val_loss_curves = {}
    test_acc_data = {}
    val_acc_data = {}
    elapsed_data = {}
    for opt in opt_order:
        rows = [r for r in results if r["opt_name"] == opt]
        val_acc_curves[opt] = [r["val_accs"] for r in rows]
        val_loss_curves[opt] = [r["val_losses"] for r in rows]
        test_accs = [r["test_acc"] for r in rows]
        val_accs = [r["best_val_acc"] for r in rows]
        elapsed = [r["elapsed"] for r in rows]
        test_acc_data[opt] = (float(np.mean(test_accs)), float(np.std(test_accs)))
        val_acc_data[opt] = (float(np.mean(val_accs)), float(np.std(val_accs)))
        elapsed_data[opt] = (float(np.mean(elapsed)), float(np.std(elapsed)))

    plot_multi_learning_curves(
        val_acc_curves,
        title="EJ2 — Val accuracy vs epoch por optimizador",
        path=os.path.join(OUT_DIR, "val_acc_curves.png"),
        ylabel="Accuracy (val)",
    )
    plot_multi_learning_curves(
        val_loss_curves,
        title="EJ2 — Val loss vs epoch por optimizador",
        path=os.path.join(OUT_DIR, "val_loss_curves.png"),
        ylabel="Loss (val)",
    )
    plot_multi_bar(val_acc_data,
                   title="EJ2 — Best val accuracy por optimizador (3 seeds)",
                   path=os.path.join(OUT_DIR, "val_acc_bar.png"),
                   ylabel="Val accuracy")
    plot_multi_bar(test_acc_data,
                   title="EJ2 — Test accuracy por optimizador (3 seeds)",
                   path=os.path.join(OUT_DIR, "test_acc_bar.png"),
                   ylabel="Test accuracy")
    plot_multi_bar(elapsed_data,
                   title="EJ2 — Tiempo de entrenamiento por optimizador (s)",
                   path=os.path.join(OUT_DIR, "elapsed_bar.png"),
                   ylabel="Segundos")

    summary = "Comparación de optimizadores (3 seeds):\n\n"
    for opt in sorted(opt_order, key=lambda o: -val_acc_data[o][0]):
        summary += (f"  {opt:14s}: val={val_acc_data[opt][0]:.4f}±{val_acc_data[opt][1]:.4f}"
                    f"  test={test_acc_data[opt][0]:.4f}"
                    f"  elapsed={elapsed_data[opt][0]:.1f}s\n")
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
