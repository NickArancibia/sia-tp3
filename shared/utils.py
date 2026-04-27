import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_learning_curves(epoch_train, val_losses=None, step_train=None,
                         title="Learning Curve", path=None):
    """Two panels: left = per-step train loss (noisy), right = per-epoch avg + val."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: per-step train — shows real variation
    ax = axes[0]
    if step_train:
        ax.plot(step_train, color="steelblue", alpha=0.6, linewidth=0.6, label="Train MSE (por step)")
    ax.set_xlabel("Step (mini-batch)")
    ax.set_ylabel("MSE")
    ax.set_title(f"{title}\nTrain por step")
    ax.legend()

    # Right: per-epoch average train + val — shows convergence trend
    ax = axes[1]
    ax.plot(epoch_train, label="Train MSE (promedio época)")
    if val_losses is not None:
        ax.plot(val_losses, label="Val MSE")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title(f"{title}\nPromedio por época")
    ax.legend()

    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_multi_learning_curves(epoch_curves, step_curves=None, title="Learning Curves", path=None):
    """epoch_curves: {label: epoch_losses}. step_curves: {label: step_losses}."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # Left: per-step — shows real variation
    ax = axes[0]
    if step_curves:
        for label, losses in step_curves.items():
            ax.plot(losses, alpha=0.6, linewidth=0.6, label=label)
    ax.set_xlabel("Step (mini-batch)")
    ax.set_ylabel("MSE")
    ax.set_title(f"{title}\nPor step")
    ax.legend()

    # Right: per-epoch average — shows convergence trend
    ax = axes[1]
    for label, losses in epoch_curves.items():
        ax.plot(losses, label=label)
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title(f"{title}\nPromedio por época")
    ax.legend()

    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_roc(fpr, tpr, auc_val, path=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC-ROC = {auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR (Falsos Positivos)")
    ax.set_ylabel("TPR (Recall)")
    ax.set_title("Curva ROC")
    ax.legend()
    if path:
        save_fig(fig, path)
    return fig


def plot_pr(precisions, recalls, auc_val, path=None):
    order = np.argsort(recalls)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recalls[order], precisions[order], label=f"AUC-PR = {auc_val:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall")
    ax.legend()
    if path:
        save_fig(fig, path)
    return fig


def plot_confusion_matrix(cm, path=None):
    labels = ["No Fraude (0)", "Fraude (1)"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Pred {l}" for l in labels])
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f"Real {l}" for l in labels])
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=12)
    ax.set_title("Matriz de Confusión")
    fig.colorbar(im)
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_threshold_sweep(thresholds, precisions, recalls, f1s, best_t, path=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls, label="Recall")
    ax.plot(thresholds, f1s, label="F1", linewidth=2)
    ax.axvline(best_t, color="red", linestyle="--",
               label=f"Umbral óptimo = {best_t:.3f}")
    ax.set_xlabel("Umbral de detección")
    ax.set_ylabel("Score")
    ax.set_title("Barrido de umbral de detección de fraude")
    ax.legend()
    if path:
        save_fig(fig, path)
    return fig
