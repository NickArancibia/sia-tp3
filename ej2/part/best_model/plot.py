"""Gráficos del mejor modelo guardado en el experimento 5.

Carga model.npz + scaler.npz, evalúa sobre digits_test.csv y produce:
  results/part/best_model/confusion_matrix.png  — matriz de confusión 10×10
  results/part/best_model/f1_per_digit.png      — F1 por dígito (barras, destaca 5 y 8)
  results/part/best_model/misclassified.png     — grilla de muestras mal clasificadas
"""
import os
import sys

EJ2_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ2_DIR)
sys.path.insert(0, EJ2_DIR)
sys.path.insert(0, REPO_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shared.mlp import MLP
from shared.preprocessing import ZScoreScaler
from shared.utils import plot_confusion_matrix, plot_misclassified_samples, save_fig

RESULTS_DIR = os.path.join(EJ2_DIR, "results", "part", "best_model")
TEST_CSV = os.path.join(EJ2_DIR, "data", "digits_test.csv")


def _load_test(n_classes):
    df = pd.read_csv(TEST_CSV, header=None)
    X = df.iloc[:, 1:].to_numpy(dtype=float) / 255.0
    y = df.iloc[:, 0].to_numpy(dtype=int)
    return X, y


def _predict(model, scaler, X_raw):
    X = scaler.transform(X_raw)
    probs = model.predict(X)
    return np.argmax(probs, axis=1)


def _compute_metrics(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    f1_per_class = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_class.append(f1)

    return cm, np.array(f1_per_class)


def plot_f1_per_digit(f1_scores, path):
    n = len(f1_scores)
    digits = list(range(n))
    colors = []
    for d in digits:
        if d == 8:
            colors.append("tomato")
        elif d == 5:
            colors.append("orange")
        else:
            colors.append("steelblue")

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(digits, f1_scores, color=colors, alpha=0.85)
    for bar, f1 in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{f1:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(digits)
    ax.set_xticklabels([str(d) for d in digits])
    ax.set_xlabel("Dígito")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.08)
    ax.set_title("F1-score por dígito (mejor modelo)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.85, label="Dígito regular"),
        Patch(facecolor="orange", alpha=0.85, label="Dígito 5 (sub-representado)"),
        Patch(facecolor="tomato", alpha=0.85, label="Dígito 8 (ausente en train)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8)
    fig.tight_layout()
    save_fig(fig, path)
    print(f"Guardado: {path}")


def main():
    model_path = os.path.join(RESULTS_DIR, "model.npz")
    scaler_path = os.path.join(RESULTS_DIR, "scaler.npz")

    for p in (model_path, scaler_path):
        if not os.path.exists(p):
            print(f"ERROR: ejecutar run.py primero (falta {p})")
            return

    model = MLP.load(model_path)
    scaler = ZScoreScaler.load(scaler_path)
    n_classes = model.architecture[-1]

    X_raw, y_true = _load_test(n_classes)
    y_pred = _predict(model, scaler, X_raw)

    cm, f1_scores = _compute_metrics(y_true, y_pred, n_classes)
    acc = (y_true == y_pred).mean()
    macro_f1 = f1_scores.mean()

    print(f"\nMétricas sobre digits_test.csv:")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {macro_f1:.4f}")
    print(f"\nF1 por dígito:")
    for d, f1 in enumerate(f1_scores):
        flag = " ← sub-representado" if d == 5 else (" ← ausente en train" if d == 8 else "")
        print(f"  {d}: {f1:.4f}{flag}")

    plot_confusion_matrix(
        cm,
        path=os.path.join(RESULTS_DIR, "confusion_matrix.png"),
        labels=[str(i) for i in range(n_classes)],
        title=f"Matriz de confusión — mejor modelo (acc={acc:.4f})",
    )
    print(f"Guardado: {os.path.join(RESULTS_DIR, 'confusion_matrix.png')}")

    plot_f1_per_digit(
        f1_scores,
        path=os.path.join(RESULTS_DIR, "f1_per_digit.png"),
    )

    fig = plot_misclassified_samples(
        X_raw, y_true, y_pred, n=16,
        path=os.path.join(RESULTS_DIR, "misclassified.png"),
    )
    if fig is not None:
        print(f"Guardado: {os.path.join(RESULTS_DIR, 'misclassified.png')}")
    else:
        print("No hay muestras mal clasificadas.")

    print(f"\nGráficos guardados en {RESULTS_DIR}")


if __name__ == "__main__":
    main()
