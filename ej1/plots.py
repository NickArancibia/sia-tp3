import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from shared.activations import activate


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


def plot_learning_curves(epoch_train, val_losses=None, title="Learning Curve", path=None,
                         zoom_tail=False):
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

    if zoom_tail:
        tail_start = max(0, int(len(epochs) * 0.7))
        tail_lows = [np.min(train_mean[tail_start:] - train_std[tail_start:])]
        tail_highs = [np.max(train_mean[tail_start:] + train_std[tail_start:])]
        if val_arr is not None:
            tail_lows.append(np.min(val_mean[tail_start:] - val_std[tail_start:]))
            tail_highs.append(np.max(val_mean[tail_start:] + val_std[tail_start:]))
        ymin = float(min(tail_lows))
        ymax = float(max(tail_highs))
        margin = max((ymax - ymin) * 0.15, ymax * 0.02, 1e-4)
        ax.set_ylim(max(0.0, ymin - margin), ymax + margin)

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


def plot_learning_curve_comparison(epoch_curves_runs, title="Learning Curves", path=None):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(9, 5))

    for idx, (label, runs) in enumerate(epoch_curves_runs.items()):
        arr = _to_runs(runs)
        epochs = np.arange(1, arr.shape[1] + 1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        color = colors[idx % len(colors)]
        ax.plot(epochs, mean, label=label, color=color)
        if arr.shape[0] > 1:
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title(title)
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


def plot_threshold_sweep(thresholds, precisions, recalls, f1s, best_t, path=None,
                         precisions_std=None, recalls_std=None, f1s_std=None):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(thresholds, precisions, label="Precision")
    ax.plot(thresholds, recalls, label="Recall")
    ax.plot(thresholds, f1s, label="F1", linewidth=2)
    if precisions_std is not None:
        ax.fill_between(thresholds, precisions - precisions_std, precisions + precisions_std,
                        alpha=0.2)
    if recalls_std is not None:
        ax.fill_between(thresholds, recalls - recalls_std, recalls + recalls_std,
                        alpha=0.2)
    if f1s_std is not None:
        ax.fill_between(thresholds, f1s - f1s_std, f1s + f1s_std,
                        alpha=0.15)
    ax.axvline(best_t, color="red", linestyle="--",
               label=f"Umbral óptimo = {best_t:.3f}")
    ax.set_xlabel("Umbral de detección")
    ax.set_ylabel("Score")
    ax.set_title("Barrido de umbral de detección de fraude")
    ax.legend()
    if path:
        save_fig(fig, path)
    return fig


def plot_target_vs_prediction(targets, predictions, path=None):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(targets, predictions, alpha=0.35, s=18, color="slateblue")

    lo = float(min(targets.min(), predictions.min()))
    hi = float(max(targets.max(), predictions.max()))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="Ideal: y = x")

    ax.set_xlabel("Target (BigModel)")
    ax.set_ylabel("Predicción del perceptrón")
    ax.set_title("Target vs Predicción")
    ax.legend()
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_internal_function(pre_activations, targets, activation, beta=1.0, path=None):
    pre_activations = np.asarray(pre_activations)
    targets = np.asarray(targets)

    order = np.argsort(pre_activations)
    h_sorted = pre_activations[order]
    curve = activate(h_sorted, activation, beta)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pre_activations, targets, alpha=0.35, s=18, color="slateblue",
               label="Targets BigModel")
    ax.plot(h_sorted, curve, color="crimson", linewidth=2.0,
            label=f"Curva del perceptrón ({activation})")

    ax.set_xlabel("Score interno h = w·x + b")
    ax.set_ylabel("Target / predicción")
    ax.set_title("Función interna del perceptrón")
    ax.legend()
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_internal_function_comparison(model_runs, title="Función interna", path=None):
    fig, axes = plt.subplots(1, len(model_runs), figsize=(7 * len(model_runs), 5), sharey=True)
    if len(model_runs) == 1:
        axes = [axes]

    for ax, run in zip(axes, model_runs):
        pre_activations = np.asarray(run["pre_activations"])
        targets = np.asarray(run["targets"])
        order = np.argsort(pre_activations)
        h_sorted = pre_activations[order]
        curve = activate(h_sorted, run["activation"], run.get("beta", 1.0))

        ax.scatter(pre_activations, targets, alpha=0.2, s=14, color="slateblue",
                   label="Targets BigModel")
        ax.plot(h_sorted, curve, color="crimson", linewidth=2.0,
                label=f"Salida {run['label']}")
        ax.set_title(run["label"])
        ax.set_xlabel("u = w·x + b")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Target / salida del modelo")
    axes[0].legend()
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_target_vs_prediction_comparison(model_runs, title="Target vs predicción", path=None):
    fig, axes = plt.subplots(1, len(model_runs), figsize=(6.5 * len(model_runs), 5), sharex=True, sharey=True)
    if len(model_runs) == 1:
        axes = [axes]

    all_targets = np.concatenate([np.asarray(run["targets"]) for run in model_runs])
    all_predictions = np.concatenate([np.asarray(run["predictions"]) for run in model_runs])
    lo = float(min(all_targets.min(), all_predictions.min()))
    hi = float(max(all_targets.max(), all_predictions.max()))

    for ax, run in zip(axes, model_runs):
        targets = np.asarray(run["targets"])
        predictions = np.asarray(run["predictions"])
        ax.scatter(targets, predictions, alpha=0.2, s=14, color="slateblue")
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0, label="Ideal: y = x")
        ax.set_title(run["label"])
        ax.set_xlabel("Target de BigModel")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Predicción de TinyModel")
    axes[0].legend()
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_metric_bars(labels, means, stds, ylabel, title, path=None):
    labels = list(labels)
    means = np.asarray(means, dtype=float)
    stds = np.asarray(stds, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=8, alpha=0.8, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    mean_max = float(np.max(means)) if len(means) else 0.0
    mean_min = float(np.min(means)) if len(means) else 0.0
    std_max = float(np.max(stds)) if len(stds) else 0.0
    margin = max(std_max * 2, mean_max * 0.03, 0.01)
    ax.set_ylim(max(0.0, mean_min - margin), mean_max + margin * 2)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + margin * 0.25,
            f"{mean:.4f}\n±{std:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_grouped_metric_bars(labels, series, ylabel, title, path=None):
    labels = list(labels)
    series_names = list(series.keys())
    x = np.arange(len(labels))
    width = 0.8 / max(1, len(series_names))

    fig, ax = plt.subplots(figsize=(9, 5))
    all_means = []
    all_stds = []

    for idx, name in enumerate(series_names):
        means = np.asarray(series[name]["means"], dtype=float)
        stds = np.asarray(series[name]["stds"], dtype=float)
        offsets = x + (idx - (len(series_names) - 1) / 2) * width
        bars = ax.bar(offsets, means, width=width, yerr=stds, capsize=6, alpha=0.85, label=name)
        all_means.extend(means.tolist())
        all_stds.extend(stds.tolist())
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(std * 0.4, 0.005),
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    mean_max = max(all_means) if all_means else 1.0
    mean_min = min(all_means) if all_means else 0.0
    std_max = max(all_stds) if all_stds else 0.0
    margin = max(std_max * 2, (mean_max - mean_min) * 0.15, 0.02)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(max(0.0, mean_min - margin), min(1.05, mean_max + margin * 1.4))
    ax.legend()
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig


def plot_strategy_overfitting_curves(
    strategy_curves,
    path=None,
    title="Overfitting por estrategia de datos",
    zoom_tail=False,
    show_std=True,
):
    labels = list(strategy_curves.keys())
    fig, axes = plt.subplots(1, len(labels), figsize=(6 * len(labels), 4.5), sharey=True)
    if len(labels) == 1:
        axes = [axes]

    tail_lows = []
    tail_highs = []

    for ax, label in zip(axes, labels):
        train_arr = _to_runs(strategy_curves[label]["train"])
        val_arr = _to_runs(strategy_curves[label]["val"])
        epochs = np.arange(1, train_arr.shape[1] + 1)

        train_mean = train_arr.mean(axis=0)
        train_std = train_arr.std(axis=0)
        val_mean = val_arr.mean(axis=0)
        val_std = val_arr.std(axis=0)

        if zoom_tail:
            tail_start = max(0, int(len(epochs) * 0.7))
            train_low = train_mean[tail_start:] - (train_std[tail_start:] if show_std else 0.0)
            val_low = val_mean[tail_start:] - (val_std[tail_start:] if show_std else 0.0)
            train_high = train_mean[tail_start:] + (train_std[tail_start:] if show_std else 0.0)
            val_high = val_mean[tail_start:] + (val_std[tail_start:] if show_std else 0.0)
            tail_lows.extend([np.min(train_low), np.min(val_low)])
            tail_highs.extend([np.max(train_high), np.max(val_high)])

        ax.plot(epochs, train_mean, label="Train MSE", color="steelblue")
        if show_std and train_arr.shape[0] > 1:
            ax.fill_between(epochs, train_mean - train_std, train_mean + train_std,
                            alpha=0.2, color="steelblue")

        ax.plot(epochs, val_mean, label="Val MSE", color="tomato")
        if show_std and val_arr.shape[0] > 1:
            ax.fill_between(epochs, val_mean - val_std, val_mean + val_std,
                            alpha=0.2, color="tomato")

        ax.set_title(label)
        ax.set_xlabel("Época")
        ax.grid(alpha=0.2)

    if zoom_tail:
        ymin = float(min(tail_lows)) if tail_lows else 0.0
        ymax = float(max(tail_highs)) if tail_highs else 1.0
        margin = max((ymax - ymin) * 0.15, ymax * 0.02, 1e-4)
        for ax in axes:
            ax.set_ylim(max(0.0, ymin - margin), ymax + margin)

    axes[0].set_ylabel("MSE")
    axes[0].legend()
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    if path:
        save_fig(fig, path)
    return fig
