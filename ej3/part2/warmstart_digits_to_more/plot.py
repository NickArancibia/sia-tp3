"""EJ3 part2 - Plots para warm-start digits -> more_digits."""
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

OUT_DIR = os.path.join(RESULTS_PART2, "warmstart_digits_to_more")


def main():
    with open(os.path.join(OUT_DIR, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    pre_val_curves = [r["pretrain_hist"]["val_accs"] for r in results]
    ft_val_curves = [r["finetune_hist"]["val_accs"] for r in results]

    pre_test = np.array([r["pretrain_eval"]["test_acc"] for r in results])
    ft_test = np.array([r["finetune_eval"]["test_acc"] for r in results])

    pre_val_best = np.array([r["pretrain_hist"]["best_val_acc"] for r in results])
    ft_val_best = np.array([r["finetune_hist"]["best_val_acc"] for r in results])

    plot_multi_learning_curves(
        {
            "pretrain (digits)": pre_val_curves,
            "finetune (more_digits)": ft_val_curves,
        },
        title="EJ3 - Warm-start: val accuracy (pretrain vs finetune)",
        path=os.path.join(OUT_DIR, "val_acc_curves.png"),
        ylabel="Val accuracy",
    )

    plot_multi_bar(
        {
            "pretrain (digits)": (float(pre_test.mean()), float(pre_test.std())),
            "finetune (more_digits)": (float(ft_test.mean()), float(ft_test.std())),
        },
        title="EJ3 - Warm-start: test accuracy",
        path=os.path.join(OUT_DIR, "test_acc_bar.png"),
        ylabel="Test accuracy",
    )

    cms = np.stack([r["finetune_eval"]["test_cm"] for r in results])
    cm_mean = cms.mean(axis=0).round().astype(int)
    plot_confusion_matrix(
        cm_mean,
        labels=[str(i) for i in range(cm_mean.shape[0])],
        title="EJ3 - Warm-start: Confusion Matrix (finetune test, promedio)",
        path=os.path.join(OUT_DIR, "confusion_matrix.png"),
    )

    per_class_f1 = np.stack([r["finetune_eval"]["test_per_class"]["f1"] for r in results])
    n_classes = per_class_f1.shape[1]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_classes)
    means = per_class_f1.mean(axis=0)
    stds = per_class_f1.std(axis=0)
    ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_classes)])
    ax.set_xlabel("Digito")
    ax.set_ylabel("F1 (test)")
    ax.set_title("EJ3 - Warm-start: F1 por clase (finetune test)")
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, os.path.join(OUT_DIR, "per_class_f1.png"))

    summary = "EJ3 - Warm-start digits -> more_digits\n\n"
    summary += f"Seeds: {len(results)}\n"
    summary += f"Config: {results[0]['config']}\n\n"
    summary += "Resultados (mean +/- std):\n"
    summary += f"  Pretrain test acc: {pre_test.mean():.4f} +/- {pre_test.std():.4f}\n"
    summary += f"  Pretrain val acc:  {pre_val_best.mean():.4f} +/- {pre_val_best.std():.4f}\n"
    summary += f"  Finetune test acc: {ft_test.mean():.4f} +/- {ft_test.std():.4f}\n"
    summary += f"  Finetune val acc:  {ft_val_best.mean():.4f} +/- {ft_val_best.std():.4f}\n"

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
