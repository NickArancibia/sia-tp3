import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _to_runs(data):
    """Normalize a single run (list of floats) or multiple runs (list of lists) to a 2D array.
    Shorter runs are padded with their last value."""
    if data is None:
        return None
    if isinstance(data[0], (int, float, np.floating, np.integer)):
        data = [data]
    max_len = max(len(r) for r in data)
    padded = [list(r) + [r[-1]] * (max_len - len(r)) for r in data]
    return np.array(padded, dtype=float)


def plot_learning_curves(epoch_train, val_losses=None, title="Learning Curve", path=None, ylabel="MSE"):
    """Plot per-epoch loss with optional mean ± std bands when multiple seeds are provided."""
    train_arr = _to_runs(epoch_train)
    val_arr = _to_runs(val_losses)
    epochs = np.arange(1, train_arr.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    train_mean = train_arr.mean(axis=0)
    train_std = train_arr.std(axis=0)
    ax.plot(epochs, train_mean, label="Train", color="steelblue")
    if train_arr.shape[0] > 1:
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color="steelblue")

    if val_arr is not None:
        val_mean = val_arr.mean(axis=0)
        val_std = val_arr.std(axis=0)
        ax.plot(epochs, val_mean, label="Validation", color="tomato")
        if val_arr.shape[0] > 1:
            ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                            alpha=0.2, color="tomato")

    ax.set_xlabel("Epoca")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_accuracy_curves(train_acc_runs, val_acc_runs=None, title="Accuracy", path=None):
    """Plot accuracy curves over epochs with mean ± std bands."""
    return plot_learning_curves(train_acc_runs, val_acc_runs, title=title, path=path, ylabel="Accuracy")


def plot_multi_learning_curves(epoch_curves_runs, title="Learning Curves", path=None, ylabel="MSE"):
    """Two panels: learning curves with bands (left) + final value bar chart with
    error bars (right).

    epoch_curves_runs: {label: runs} where runs is a list of floats (single seed)
                       or a list of lists (multiple seeds).
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    multi_seed = any(
        not isinstance(runs[0], (int, float, np.floating, np.integer))
        for runs in epoch_curves_runs.values()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    finals = {}
    for i, (label, runs) in enumerate(epoch_curves_runs.items()):
        arr = _to_runs(runs)
        epochs = np.arange(1, arr.shape[1] + 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        color = colors[i % len(colors)]
        ax.plot(epochs, mean, label=label, color=color)
        if arr.shape[0] > 1:
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)
        finals[label] = (arr[:, -1].mean(), arr[:, -1].std(), color)

    ax.set_xlabel("Epoca")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nCurvas de aprendizaje (media +/- std)")
    ax.legend()

    ax = axes[1]
    labels = list(finals.keys())
    means = [finals[l][0] for l in labels]
    stds = [finals[l][1] for l in labels]
    bar_colors = [finals[l][2] for l in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds if multi_seed else None,
                  color=bar_colors, alpha=0.7, capsize=8, error_kw={"linewidth": 2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(f"{ylabel} final")
    ax.set_title(f"{ylabel} final +/- std")
    for bar, mean, std in zip(bars, means, stds):
        label_txt = f"{mean:.5f}" + (f"\n+/-{std:.5f}" if multi_seed else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(stds) if multi_seed else 0),
                label_txt, ha="center", va="bottom", fontsize=8)
    margin = max(stds) * 5 if multi_seed and max(stds) > 0 else max(means) * 0.05
    ax.set_ylim(max(0, min(means) - margin * 3), max(means) + margin * 8)

    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_multi_bar(data, title="Comparison", path=None, ylabel="Score"):
    """Bar chart comparing multiple configurations with error bars.

    data: dict {label: (mean, std)} or {label: mean} for single runs.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(data.keys())
    multi = all(isinstance(v, (list, tuple)) and len(v) == 2 for v in data.values())
    if multi:
        means = [data[l][0] for l in labels]
        stds = [data[l][1] for l in labels]
    else:
        means = list(data.values())
        stds = None
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=8, alpha=0.7, error_kw={"linewidth": 2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{m:.4f}", ha="center", va="bottom", fontsize=8)
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


def plot_confusion_matrix(cm, path=None, labels=None, title="Matriz de Confusion"):
    """Plot NxN confusion matrix heatmap.

    cm: (C, C) confusion matrix
    labels: list of class names (optional)
    """
    n = cm.shape[0]
    if labels is None:
        labels = [str(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(5, n * 0.7), max(4, n * 0.6)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=8)
    ax.set_title(title)
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
               label=f"Umbral optimo = {best_t:.3f}")
    ax.set_xlabel("Umbral de deteccion")
    ax.set_ylabel("Score")
    ax.set_title("Barrido de umbral de deteccion de fraude")
    ax.legend()
    if path:
        save_fig(fig, path)
    return fig


def plot_misclassified_samples(images, y_true, y_pred, n=16, path=None):
    """Plot a grid of misclassified digit images with true/predicted labels.

    images: array of shape (N, 784) or (N, 28, 28)
    y_true, y_pred: arrays of shape (N,) with integer labels
    n: max number of samples to show
    """
    mis_idx = np.where(y_true != y_pred)[0]
    if len(mis_idx) == 0:
        return None
    mis_idx = mis_idx[:n]
    cols = min(4, len(mis_idx))
    rows = (len(mis_idx) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    for i, idx in enumerate(mis_idx):
        img = images[idx]
        if img.ndim == 1:
            size = int(np.sqrt(img.shape[0]))
            img = img.reshape(size, size)
        axes[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"Real: {y_true[idx]} / Pred: {y_pred[idx]}", fontsize=9)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    fig.suptitle("Muestras mal clasificadas", fontsize=12)
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig