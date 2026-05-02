"""EJ2 part2 — Plots para variantes de learning rate.

Lee results.pkl y genera:
- val_acc_curves.png / val_loss_curves.png: curvas vs época por lr (mean ± std)
- test_acc_bar.png / val_acc_bar.png: barras de acc final por lr
- summary.txt: ranking + lr ganador
"""
import os
import pickle
import sys

PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART2_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(PART2_DIR)))

import numpy as np

from common import RESULTS_PART2
from shared.utils import (plot_multi_bar, plot_multi_learning_curves)

OUT_DIR = os.path.join(RESULTS_PART2, "learning_rate")


def main():
    with open(os.path.join(OUT_DIR, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    lrs = sorted({r["lr"] for r in results})

    val_acc_curves = {}
    val_loss_curves = {}
    test_acc_data = {}
    val_acc_data = {}
    train_acc_data = {}
    for lr in lrs:
        rows = [r for r in results if r["lr"] == lr]
        label = f"lr={lr:.0e}"
        val_acc_curves[label] = [r["val_accs"] for r in rows]
        val_loss_curves[label] = [r["val_losses"] for r in rows]
        test_accs = [r["test_acc"] for r in rows]
        val_accs = [r["best_val_acc"] for r in rows]
        train_accs = [r["train_acc"] for r in rows]
        test_acc_data[label] = (float(np.mean(test_accs)), float(np.std(test_accs)))
        val_acc_data[label] = (float(np.mean(val_accs)), float(np.std(val_accs)))
        train_acc_data[label] = (float(np.mean(train_accs)), float(np.std(train_accs)))

    plot_multi_learning_curves(
        val_acc_curves,
        title="EJ2 — Val accuracy vs epoch por learning rate",
        path=os.path.join(OUT_DIR, "val_acc_curves.png"),
        ylabel="Accuracy (val)",
    )
    plot_multi_learning_curves(
        val_loss_curves,
        title="EJ2 — Val loss (MSE) vs epoch por learning rate",
        path=os.path.join(OUT_DIR, "val_loss_curves.png"),
        ylabel="Loss (val)",
    )
    plot_multi_bar(test_acc_data,
                   title="EJ2 — Test accuracy final por lr (3 seeds)",
                   path=os.path.join(OUT_DIR, "test_acc_bar.png"),
                   ylabel="Test accuracy")
    plot_multi_bar(val_acc_data,
                   title="EJ2 — Best val accuracy por lr (3 seeds)",
                   path=os.path.join(OUT_DIR, "val_acc_bar.png"),
                   ylabel="Val accuracy")

    best_lr_label = max(val_acc_data, key=lambda k: val_acc_data[k][0])
    best_test = test_acc_data[best_lr_label]
    best_val = val_acc_data[best_lr_label]
    summary = (
        f"Mejor lr (por val_acc promedio): {best_lr_label}\n"
        f"  test_acc = {best_test[0]:.4f} ± {best_test[1]:.4f}\n"
        f"  val_acc  = {best_val[0]:.4f} ± {best_val[1]:.4f}\n\n"
        "Ranking por val_acc:\n"
    )
    for label, (m, s) in sorted(val_acc_data.items(),
                                key=lambda kv: -kv[1][0]):
        tm, ts = test_acc_data[label]
        trm, _ = train_acc_data[label]
        summary += (f"  {label:12s}: val={m:.4f}±{s:.4f}  "
                    f"test={tm:.4f}±{ts:.4f}  train={trm:.4f}\n")
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
