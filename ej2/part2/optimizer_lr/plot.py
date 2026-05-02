"""EJ2 part2 — Plots para optimizer × lr (heatmap 2D + curva)."""
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

OUT_DIR = os.path.join(RESULTS_PART2, "optimizer_lr")


def main():
    with open(os.path.join(OUT_DIR, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    opts = []
    for r in results:
        if r["opt_name"] not in opts:
            opts.append(r["opt_name"])
    lrs = sorted({r["lr"] for r in results})
    lr_lbls = [f"{lr:.0e}" for lr in lrs]

    val = np.full((len(opts), len(lrs)), np.nan)
    val_std = np.full((len(opts), len(lrs)), np.nan)
    test = np.full((len(opts), len(lrs)), np.nan)
    for i, opt in enumerate(opts):
        for j, lr in enumerate(lrs):
            rows = [r for r in results
                    if r["opt_name"] == opt and abs(r["lr"] - lr) < 1e-12]
            if rows:
                val[i, j] = float(np.mean([r["best_val_acc"] for r in rows]))
                val_std[i, j] = float(np.std([r["best_val_acc"] for r in rows]))
                test[i, j] = float(np.mean([r["test_acc"] for r in rows]))

    # Heatmap val_acc
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(val, cmap="viridis", aspect="auto", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(lrs)))
    ax.set_xticklabels(lr_lbls)
    ax.set_yticks(range(len(opts)))
    ax.set_yticklabels(opts)
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Optimizador")
    ax.set_title("EJ2 — Val accuracy: optimizer × lr (arch [784,64,32,10], 3 seeds)")
    for i in range(len(opts)):
        for j in range(len(lrs)):
            v = val[i, j]
            txt = f"{v:.3f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if (np.isnan(v) or v < 0.7) else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, label="Val accuracy")
    save_fig(fig, os.path.join(OUT_DIR, "val_acc_heatmap.png"))

    # Curva val_acc vs lr (1 línea por opt)
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, opt in enumerate(opts):
        ax.errorbar(lrs, val[i, :], yerr=val_std[i, :],
                    fmt="o-", color=colors[i % len(colors)],
                    label=opt, capsize=4, linewidth=2, markersize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Val accuracy")
    ax.set_title("EJ2 — Val accuracy vs lr por optimizador (3 seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.0)
    save_fig(fig, os.path.join(OUT_DIR, "val_acc_vs_lr.png"))

    # Summary
    summary = "EJ2 — Sweep optimizer × lr:\n\n"
    flat_idx = np.unravel_index(np.nanargmax(val), val.shape)
    best_opt, best_lr = opts[flat_idx[0]], lrs[flat_idx[1]]
    best_val = val[flat_idx]
    best_test = test[flat_idx]
    summary += f"Mejor combinación global: opt={best_opt} lr={best_lr:.0e}\n"
    summary += f"  val_acc={best_val:.4f}  test_acc={best_test:.4f}\n\n"
    summary += "Mejor lr por optimizador:\n"
    for i, opt in enumerate(opts):
        if np.all(np.isnan(val[i, :])):
            continue
        j = int(np.nanargmax(val[i, :]))
        m, sd = val[i, j], val_std[i, j]
        t = test[i, j]
        summary += (f"  {opt:10s}: lr={lrs[j]:.0e}  "
                    f"val={m:.4f}±{sd:.4f}  test={t:.4f}\n")
    summary += "\nTabla val_acc:\n          "
    for lr in lrs:
        summary += f"{lr:>10.0e}"
    summary += "\n"
    for i, opt in enumerate(opts):
        summary += f"{opt:>10s}"
        for j in range(len(lrs)):
            v = val[i, j]
            summary += f"{v:>10.4f}" if not np.isnan(v) else f"{'—':>10s}"
        summary += "\n"
    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
