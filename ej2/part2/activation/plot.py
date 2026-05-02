"""EJ2 part2 — Plots para activación intermedia."""
import os
import pickle
import sys

PART2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PART2_DIR)
sys.path.insert(0, os.path.dirname(os.path.dirname(PART2_DIR)))

import numpy as np

from common import RESULTS_PART2
from shared.utils import plot_multi_bar, plot_multi_learning_curves

OUT_DIR = os.path.join(RESULTS_PART2, "activation")


def main():
    with open(os.path.join(OUT_DIR, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    acts = []
    for r in results:
        if r["activation"] not in acts:
            acts.append(r["activation"])

    val_acc_curves = {}
    val_loss_curves = {}
    test_acc_data = {}
    val_acc_data = {}
    train_acc_data = {}
    elapsed_data = {}
    for act in acts:
        rows = [r for r in results if r["activation"] == act]
        val_acc_curves[act] = [r["val_accs"] for r in rows]
        val_loss_curves[act] = [r["val_losses"] for r in rows]
        test_accs = [r["test_acc"] for r in rows]
        val_accs = [r["best_val_acc"] for r in rows]
        train_accs = [r["train_acc"] for r in rows]
        elapseds = [r["elapsed"] for r in rows]
        test_acc_data[act] = (float(np.mean(test_accs)), float(np.std(test_accs)))
        val_acc_data[act] = (float(np.mean(val_accs)), float(np.std(val_accs)))
        train_acc_data[act] = (float(np.mean(train_accs)), float(np.std(train_accs)))
        elapsed_data[act] = (float(np.mean(elapseds)), float(np.std(elapseds)))

    plot_multi_learning_curves(
        val_acc_curves,
        title="EJ2 — Val accuracy vs epoch por activación intermedia",
        path=os.path.join(OUT_DIR, "val_acc_curves.png"),
        ylabel="Val accuracy",
    )
    plot_multi_learning_curves(
        val_loss_curves,
        title="EJ2 — Val loss vs epoch por activación",
        path=os.path.join(OUT_DIR, "val_loss_curves.png"),
        ylabel="Loss (val)",
    )
    plot_multi_bar(val_acc_data,
                   title="EJ2 — Best val accuracy por activación intermedia (3 seeds)",
                   path=os.path.join(OUT_DIR, "val_acc_bar.png"),
                   ylabel="Val accuracy")
    plot_multi_bar(test_acc_data,
                   title="EJ2 — Test accuracy por activación intermedia (3 seeds)",
                   path=os.path.join(OUT_DIR, "test_acc_bar.png"),
                   ylabel="Test accuracy")

    summary = "EJ2 — Comparación de activación intermedia (3 seeds):\n\n"
    summary += "act        | val_acc          | test_acc         | train_acc | elapsed\n"
    summary += "-" * 78 + "\n"
    for act in sorted(acts, key=lambda a: -val_acc_data[a][0]):
        v, vs = val_acc_data[act]
        t, ts = test_acc_data[act]
        tr, _ = train_acc_data[act]
        e, _ = elapsed_data[act]
        summary += (f"{act:10s} | {v:.4f}±{vs:.4f}    | {t:.4f}±{ts:.4f}   | "
                    f"{tr:.4f}    | {e:.1f}s\n")
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
