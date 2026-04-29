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
    if data is None:
        return None
    if isinstance(data[0], (int, float, np.floating, np.integer)):
        data = [data]
    max_len = max(len(r) for r in data)
    padded = [list(r) + [r[-1]] * (max_len - len(r)) for r in data]
    return np.array(padded, dtype=float)


def plot_learning_curves(epoch_train, val_losses=None, title="Learning Curve", path=None):
    train_arr = _to_runs(epoch_train)
    val_arr = _to_runs(val_losses)
    epochs = np.arange(1, train_arr.shape[1] + 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    train_mean = train_arr.mean(axis=0)
    train_std = train_arr.std(axis=0)
    ax.plot(epochs, train_mean, label="Train MSE", color="steelblue")
    if train_arr.shape[0] > 1:
        ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color="steelblue")

    if val_arr is not None:
        val_mean = val_arr.mean(axis=0)
        val_std = val_arr.std(axis=0)
        ax.plot(epochs, val_mean, label="Val MSE", color="tomato")
        if val_arr.shape[0] > 1:
            ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                            alpha=0.2, color="tomato")

    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_multi_learning_curves(epoch_curves_runs, title="Learning Curves", path=None):
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

    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title(f"{title}\nCurvas de aprendizaje (media ± std)")
    ax.legend()

    ax = axes[1]
    labels = list(finals.keys())
    means  = [finals[l][0] for l in labels]
    stds   = [finals[l][1] for l in labels]
    bar_colors = [finals[l][2] for l in labels]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds if multi_seed else None,
                  color=bar_colors, alpha=0.7, capsize=8, error_kw={"linewidth": 2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("MSE final")
    ax.set_title("MSE final ± std")
    for bar, mean, std in zip(bars, means, stds):
        label_txt = f"{mean:.5f}" + (f"\n±{std:.5f}" if multi_seed else "")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max(stds) if multi_seed else 0),
                label_txt, ha="center", va="bottom", fontsize=8)
    margin = max(stds) * 5 if multi_seed and max(stds) > 0 else max(means) * 0.05
    ax.set_ylim(max(0, min(means) - margin * 3), max(means) + margin * 8)

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