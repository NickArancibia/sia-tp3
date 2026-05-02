"""EJ3 part2 — Plots para el modelo seleccionado."""
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
from shared.utils import (plot_accuracy_curves, plot_confusion_matrix,
                          plot_learning_curves, save_fig)

OUT_DIR = os.path.join(RESULTS_PART2, "selected_model")


def main():
    with open(os.path.join(OUT_DIR, "results.pkl"), "rb") as f:
        results = pickle.load(f)

    train_losses = [r["train_losses"] for r in results]
    val_losses = [r["val_losses"] for r in results]
    train_accs = [r["train_accs"] for r in results]
    val_accs = [r["val_accs"] for r in results]
    test_accs = np.array([r["test_acc"] for r in results])
    val_acc_finals = np.array([r["best_val_acc"] for r in results])
    cms = np.stack([r["test_cm"] for r in results])
    cm_mean = cms.mean(axis=0).round().astype(int)

    print(f"Test accuracy: {test_accs.mean():.4f} ± {test_accs.std():.4f}")
    print(f"Val accuracy:  {val_acc_finals.mean():.4f} ± {val_acc_finals.std():.4f}")
    target = 0.98
    n_pass = int((test_accs >= target).sum())
    print(f"Seeds que alcanzaron ≥98% test acc: {n_pass}/{len(test_accs)}")

    n_seeds = len(results)
    plot_learning_curves(train_losses, val_losses,
                         title=f"EJ3 modelo seleccionado — Loss ({n_seeds} seeds)",
                         path=os.path.join(OUT_DIR, "loss_curves.png"),
                         ylabel="MSE")
    plot_accuracy_curves(train_accs, val_accs,
                         title=f"EJ3 modelo seleccionado — Accuracy ({n_seeds} seeds)",
                         path=os.path.join(OUT_DIR, "accuracy_curves.png"))
    plot_confusion_matrix(cm_mean, labels=[str(i) for i in range(10)],
                          title="EJ3 modelo seleccionado — Confusion Matrix (test, promedio)",
                          path=os.path.join(OUT_DIR, "confusion_matrix.png"))

    per_class_pre = np.stack([r["test_per_class"]["precision"] for r in results])
    per_class_rec = np.stack([r["test_per_class"]["recall"] for r in results])
    per_class_f1 = np.stack([r["test_per_class"]["f1"] for r in results])

    n_classes = per_class_f1.shape[1]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(n_classes)
    w = 0.27
    ax.bar(x - w, per_class_pre.mean(axis=0), w, yerr=per_class_pre.std(axis=0),
           label="Precision", capsize=3, alpha=0.8)
    ax.bar(x,     per_class_rec.mean(axis=0), w, yerr=per_class_rec.std(axis=0),
           label="Recall",    capsize=3, alpha=0.8)
    ax.bar(x + w, per_class_f1.mean(axis=0),  w, yerr=per_class_f1.std(axis=0),
           label="F1",        capsize=3, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(i) for i in range(n_classes)])
    ax.set_xlabel("Dígito")
    ax.set_ylabel("Score")
    ax.set_title(f"EJ3 modelo seleccionado — Métricas por clase (test, {n_seeds} seeds)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_fig(fig, os.path.join(OUT_DIR, "per_class_metrics.png"))

    macro_f1 = np.array([r["test_per_class"]["macro_f1"] for r in results])
    weighted_f1 = np.array([r["test_per_class"]["weighted_f1"] for r in results])
    summary = f"""EJ3 — Modelo seleccionado
Configuración: {results[0]['config']}
Parámetros: {results[0]['n_params']:,}
Seeds: {len(results)}

Resultados (mean ± std):
  Test accuracy:  {test_accs.mean():.4f} ± {test_accs.std():.4f}
  Val accuracy:   {val_acc_finals.mean():.4f} ± {val_acc_finals.std():.4f}
  Macro F1:       {macro_f1.mean():.4f} ± {macro_f1.std():.4f}
  Weighted F1:    {weighted_f1.mean():.4f} ± {weighted_f1.std():.4f}

Seeds que alcanzaron ≥{target*100:.0f}% test acc: {n_pass}/{len(test_accs)}

Per-class F1 (test):
"""
    for c in range(n_classes):
        summary += f"  Dígito {c}: F1={per_class_f1.mean(axis=0)[c]:.4f}±{per_class_f1.std(axis=0)[c]:.4f}\n"

    # Top confusiones por dígito (matriz CM promedio sobre seeds).
    summary += "\nTop confusiones por dígito (matriz CM promedio):\n"
    summary += "Real  | Acc clase | Top-1 confusión       | Top-2 confusión\n"
    summary += "-" * 68 + "\n"
    for c in range(n_classes):
        total = cm_mean[c, :].sum()
        if total == 0:
            summary += f"  {c}   | (sin muestras reales)\n"
            continue
        acc_c = cm_mean[c, c] / total
        off_diag = [(j, cm_mean[c, j]) for j in range(n_classes) if j != c]
        off_diag.sort(key=lambda kv: -kv[1])
        top1 = off_diag[0]
        top2 = off_diag[1] if len(off_diag) > 1 else (None, 0)
        s1 = f"→{top1[0]}: {top1[1]} ({top1[1]/total*100:.1f}%)"
        s2 = (f"→{top2[0]}: {top2[1]} ({top2[1]/total*100:.1f}%)"
              if top2[0] is not None and top2[1] > 0 else "—")
        summary += f"  {c}   |  {acc_c:.3f}    | {s1:22s} | {s2}\n"

    with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    print(summary)


if __name__ == "__main__":
    main()
