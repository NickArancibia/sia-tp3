"""EJ2 part2 — Plots para batch_lr (heatmap 2D)."""
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
from shared.utils import plot_multi_bar, save_fig

OUT_DIR = os.path.join(RESULTS_PART2, "batch_lr")


def main():
    with open(os.path.join(OUT_DIR, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    lrs = sorted({r["lr"] for r in results})
    batch_order_canonical = [1, 8, 32, 128, 0]
    batches = [b for b in batch_order_canonical
               if any(r["batch_size"] == b for r in results)]
    batch_lbls = ["online" if b == 1 else ("full" if b == 0 else str(b))
                  for b in batches]
    lr_lbls = [f"{lr:.0e}" for lr in lrs]

    val = np.full((len(batches), len(lrs)), np.nan)
    test = np.full((len(batches), len(lrs)), np.nan)
    for i, b in enumerate(batches):
        for j, lr in enumerate(lrs):
            rows = [r for r in results
                    if r["batch_size"] == b and abs(r["lr"] - lr) < 1e-12]
            if rows:
                val[i, j] = float(np.mean([r["best_val_acc"] for r in rows]))
                test[i, j] = float(np.mean([r["test_acc"] for r in rows]))

    # Heatmap val_acc
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(val, cmap="viridis", aspect="auto", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels(lr_lbls)
    ax.set_yticks(range(len(batches)))
    ax.set_yticklabels(batch_lbls)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Batch size")
    ax.set_title("EJ2 — Val accuracy: batch × lr (Adam, arch [784,64,32,10], 3 seeds)")
    for i in range(len(batches)):
        for j in range(len(lrs)):
            v = val[i, j]
            txt = f"{v:.3f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if (np.isnan(v) or v < 0.7) else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, label="Val accuracy")
    save_fig(fig, os.path.join(OUT_DIR, "val_acc_heatmap.png"))

    # Heatmap test_acc
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(test, cmap="viridis", aspect="auto", vmin=0.5, vmax=0.9)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels(lr_lbls)
    ax.set_yticks(range(len(batches)))
    ax.set_yticklabels(batch_lbls)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Batch size")
    ax.set_title("EJ2 — Test accuracy: batch × lr (3 seeds)")
    for i in range(len(batches)):
        for j in range(len(lrs)):
            v = test[i, j]
            txt = f"{v:.3f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if (np.isnan(v) or v < 0.7) else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, label="Test accuracy")
    save_fig(fig, os.path.join(OUT_DIR, "test_acc_heatmap.png"))

    # Bar: best val acc por batch
    best_per_batch = {}
    for i, b in enumerate(batches):
        row = val[i, :]
        if np.all(np.isnan(row)):
            continue
        j = int(np.nanargmax(row))
        rows = [r for r in results
                if r["batch_size"] == b and abs(r["lr"] - lrs[j]) < 1e-12]
        m = float(np.mean([r["best_val_acc"] for r in rows]))
        s = float(np.std([r["best_val_acc"] for r in rows]))
        best_per_batch[f"{batch_lbls[i]}\nlr={lr_lbls[j]}"] = (m, s)

    plot_multi_bar(best_per_batch,
                   title="EJ2 — Mejor val accuracy por batch_size",
                   path=os.path.join(OUT_DIR, "best_per_batch.png"),
                   ylabel="Val accuracy")

    # Summary
    flat_idx = np.unravel_index(np.nanargmax(val), val.shape)
    best_b, best_lr = batches[flat_idx[0]], lrs[flat_idx[1]]
    best_val = val[flat_idx]
    best_test = test[flat_idx]
    summary = "EJ2 — Sweep batch × lr (Adam):\n\n"
    summary += f"Mejor combinación (por val_acc): batch={batch_label_str(best_b)} lr={best_lr:.0e}\n"
    summary += f"  val_acc={best_val:.4f}  test_acc={best_test:.4f}\n\n"
    summary += "Tabla val_acc:\n      "
    for lr in lrs:
        summary += f"{lr:>10.0e}"
    summary += "\n"
    for i, b in enumerate(batches):
        summary += f"{batch_lbls[i]:>6s}"
        for j in range(len(lrs)):
            v = val[i, j]
            summary += f"{v:>10.4f}" if not np.isnan(v) else f"{'—':>10s}"
        summary += "\n"
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


def batch_label_str(b):
    if b == 0:
        return "full"
    if b == 1:
        return "online"
    return str(b)


if __name__ == "__main__":
    main()
