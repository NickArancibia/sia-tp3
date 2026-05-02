"""EJ3 part2 — Plots para comparación digits vs more_digits."""
import os
import pickle
import sys

EJ3_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ3_DIR)
sys.path.insert(0, EJ3_DIR)
sys.path.insert(0, REPO_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS_PART2
from shared.utils import (plot_confusion_matrix, plot_multi_bar,
                          plot_multi_learning_curves, save_fig)

OUT_DIR = os.path.join(RESULTS_PART2, "data_comparison")


def main():
    with open(os.path.join(OUT_DIR, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    datasets = []
    for r in results:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])

    val_acc_curves = {}
    test_acc_data = {}
    val_acc_data = {}
    cms = {}
    per_class_f1s = {}
    for ds in datasets:
        rows = [r for r in results if r["dataset"] == ds]
        val_acc_curves[ds] = [r["val_accs"] for r in rows]
        test_accs = [r["test_acc"] for r in rows]
        val_accs = [r["best_val_acc"] for r in rows]
        test_acc_data[ds] = (float(np.mean(test_accs)), float(np.std(test_accs)))
        val_acc_data[ds] = (float(np.mean(val_accs)), float(np.std(val_accs)))
        cms[ds] = np.stack([r["test_cm"] for r in rows]).mean(axis=0).round().astype(int)
        per_class_f1s[ds] = np.stack([r["test_per_class"]["f1"] for r in rows])

    plot_multi_learning_curves(
        val_acc_curves,
        title="EJ3 — Val accuracy: digits vs more_digits (mismo test, 3 seeds)",
        path=os.path.join(OUT_DIR, "val_acc_curves.png"),
        ylabel="Val accuracy",
    )
    plot_multi_bar(test_acc_data,
                   title="EJ3 — Test accuracy: digits vs more_digits (mismo test)",
                   path=os.path.join(OUT_DIR, "test_acc_bar.png"),
                   ylabel="Test accuracy")

    for ds in datasets:
        plot_confusion_matrix(cms[ds], labels=[str(i) for i in range(10)],
                              title=f"EJ3 — Confusion Matrix test ({ds}, promedio)",
                              path=os.path.join(OUT_DIR, f"cm_{ds}.png"))

    # Per-class F1 comparison
    n_classes = per_class_f1s[datasets[0]].shape[1]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(n_classes)
    w = 0.4
    colors = ["steelblue", "tomato"]
    for i, ds in enumerate(datasets):
        means = per_class_f1s[ds].mean(axis=0)
        stds = per_class_f1s[ds].std(axis=0)
        ax.bar(x + (i - 0.5) * w, means, w, yerr=stds, label=ds,
               color=colors[i % len(colors)], capsize=3, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_classes)])
    ax.set_xlabel("Dígito")
    ax.set_ylabel("F1 (test)")
    ax.set_title("EJ3 — F1 por clase (test): digits vs more_digits")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, os.path.join(OUT_DIR, "per_class_f1.png"))

    summary = "EJ3 — Comparación digits.csv vs more_digits.csv (mismo test):\n\n"
    for ds in datasets:
        summary += f"{ds}:\n"
        summary += f"  test_acc = {test_acc_data[ds][0]:.4f} ± {test_acc_data[ds][1]:.4f}\n"
        summary += f"  val_acc  = {val_acc_data[ds][0]:.4f} ± {val_acc_data[ds][1]:.4f}\n"
        f1_mean = per_class_f1s[ds].mean(axis=0)
        f1_std = per_class_f1s[ds].std(axis=0)
        for c in range(n_classes):
            summary += f"  F1 dígito {c}: {f1_mean[c]:.4f}±{f1_std[c]:.4f}\n"
        summary += "\n"
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
