"""EJ2 part2 — Plots para variantes de arquitectura."""
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

OUT_DIR = os.path.join(RESULTS_PART2, "architecture")


def main():
    with open(os.path.join(OUT_DIR, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    labels = []
    seen = set()
    for r in results:
        if r["arch_label"] not in seen:
            labels.append(r["arch_label"])
            seen.add(r["arch_label"])

    val_acc_curves = {}
    test_acc_data = {}
    val_acc_data = {}
    train_acc_data = {}
    n_params = {}
    for label in labels:
        rows = [r for r in results if r["arch_label"] == label]
        val_acc_curves[label] = [r["val_accs"] for r in rows]
        test_accs = [r["test_acc"] for r in rows]
        val_accs = [r["best_val_acc"] for r in rows]
        train_accs = [r["train_acc"] for r in rows]
        test_acc_data[label] = (float(np.mean(test_accs)), float(np.std(test_accs)))
        val_acc_data[label] = (float(np.mean(val_accs)), float(np.std(val_accs)))
        train_acc_data[label] = (float(np.mean(train_accs)), float(np.std(train_accs)))
        n_params[label] = rows[0]["n_params"]

    plot_multi_learning_curves(
        val_acc_curves,
        title="EJ2 — Val accuracy vs epoch por arquitectura",
        path=os.path.join(OUT_DIR, "val_acc_curves.png"),
        ylabel="Accuracy (val)",
    )
    plot_multi_bar(val_acc_data,
                   title="EJ2 — Best val accuracy por arquitectura (3 seeds)",
                   path=os.path.join(OUT_DIR, "val_acc_bar.png"),
                   ylabel="Val accuracy")
    plot_multi_bar(test_acc_data,
                   title="EJ2 — Test accuracy por arquitectura (3 seeds)",
                   path=os.path.join(OUT_DIR, "test_acc_bar.png"),
                   ylabel="Test accuracy")

    # Plot val_acc vs n_params (capacity vs accuracy) — Clase 13 Regularización
    fig, ax = plt.subplots(figsize=(8, 5))
    sorted_labels = sorted(labels, key=lambda l: n_params[l])
    xs = [n_params[l] for l in sorted_labels]
    ys = [val_acc_data[l][0] for l in sorted_labels]
    es = [val_acc_data[l][1] for l in sorted_labels]
    ts = [test_acc_data[l][0] for l in sorted_labels]
    trs = [train_acc_data[l][0] for l in sorted_labels]
    ax.errorbar(xs, ys, yerr=es, fmt="o-", color="tomato", label="Val",
                capsize=4, linewidth=2)
    ax.plot(xs, ts, "s--", color="forestgreen", label="Test", linewidth=2)
    ax.plot(xs, trs, "^:", color="steelblue", label="Train", linewidth=2)
    for x, y, l in zip(xs, ys, sorted_labels):
        ax.annotate(l, (x, y), textcoords="offset points", xytext=(5, -10),
                    fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Cantidad de parámetros")
    ax.set_ylabel("Accuracy")
    ax.set_title("EJ2 — Capacidad del modelo (parámetros) vs accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, os.path.join(OUT_DIR, "capacity_vs_acc.png"))

    best_label = max(val_acc_data, key=lambda k: val_acc_data[k][0])
    summary = f"Mejor arquitectura (por val_acc): {best_label}\n"
    summary += "Ranking por val_acc:\n"
    for label, (m, s) in sorted(val_acc_data.items(),
                                key=lambda kv: -kv[1][0]):
        tm, ts = test_acc_data[label]
        trm, _ = train_acc_data[label]
        gap = trm - m
        summary += (f"  {label:25s}: val={m:.4f}±{s:.4f}  "
                    f"test={tm:.4f}±{ts:.4f}  train={trm:.4f}  "
                    f"gap={gap:.4f}  params={n_params[label]:,}\n")
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
