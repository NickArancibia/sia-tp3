"""Infraestructura común para EJ3.

Diferencias clave con EJ2:
- Train: more_digits.csv (15741 samples, 10 clases incluido el 8)
- Test: digits_test.csv (mismo del EJ2 — el "mundo real" según el enunciado)
- Val: split estratificado del train (típicamente 20%)

Esto permite comparar EJ2 vs EJ3 sobre el MISMO test.
"""
import copy
import os
import sys
import time

EJ3_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EJ3_DIR)
sys.path.insert(0, REPO_DIR)

import numpy as np
from scipy.ndimage import rotate, shift

from shared.digit_dataset_loader import load_dataset
from shared.metrics import (accuracy, classify_from_output,
                            confusion_matrix_multiclass, per_class_metrics)
from shared.mlp import MLP
from shared.optimizers import build_optimizer
from shared.preprocessing import (build_scaler, one_hot_encode, stratified_split)
from shared.regularization import EarlyStopping

# El test del EJ3 ES el mismo test del EJ2 (digits_test.csv) por el
# enunciado: "Consideren digits_test.csv al equivalente a poner el modelo
# en producción".
TRAIN_CSV = os.path.join(EJ3_DIR, "data", "more_digits.csv")
TEST_CSV = os.path.join(os.path.dirname(EJ3_DIR), "ej2", "data", "digits_test.csv")
RESULTS_PART2 = os.path.join(EJ3_DIR, "results", "part2")


_IMG_SHAPE = (28, 28)


def _aug_batch(X_batch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-batch online augmentation: rotation ±5°, translation ±2px, noise std=0.1.

    Operates on z-score normalized data — spatial transforms are pixel-order ops
    so normalization doesn't affect them; noise std=0.1 ≈ 0.03 in [0,1] pixel space.
    """
    out = np.empty_like(X_batch)
    for i, flat in enumerate(X_batch):
        img = flat.reshape(_IMG_SHAPE)
        img = rotate(img, rng.uniform(-5.0, 5.0), reshape=False, mode="nearest")
        dy, dx = rng.uniform(-2.0, 2.0, size=2)
        img = shift(img, [dy, dx], mode="nearest")
        img = img + rng.normal(0.0, 0.1, img.shape)
        out[i] = img.flatten()
    return out


def prepare_data(train_csv=TRAIN_CSV, test_csv=TEST_CSV, val_frac=0.2,
                 scaler="z-score", stratify=True, seed=42):
    train_df = load_dataset(train_csv)
    test_df = load_dataset(test_csv)
    X_all = np.stack(train_df["image"].values)
    y_all = train_df["label"].values.astype(int)
    X_test_raw = np.stack(test_df["image"].values)
    y_test = test_df["label"].values.astype(int)

    if stratify:
        train_idx, val_idx, _ = stratified_split(y_all, val_frac, 0.0, seed)
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(y_all))
        rng.shuffle(idx)
        n_val = int(len(idx) * val_frac)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

    sc = build_scaler(scaler) if scaler else None
    if sc is not None:
        X_train = sc.fit_transform(X_all[train_idx])
        X_val = sc.transform(X_all[val_idx])
        X_test = sc.transform(X_test_raw)
    else:
        X_train = X_all[train_idx]
        X_val = X_all[val_idx]
        X_test = X_test_raw

    n_classes = max(int(y_all.max()), int(y_test.max())) + 1
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_all[train_idx],
        "y_val": y_all[val_idx],
        "y_test": y_test,
        "Y_train": one_hot_encode(y_all[train_idx], n_classes),
        "Y_val": one_hot_encode(y_all[val_idx], n_classes),
        "Y_test": one_hot_encode(y_test, n_classes),
        "scaler": sc,
        "n_classes": n_classes,
    }


def build_mlp(arch, hidden_act="tanh", out_act="logistic", beta=1.0,
              init_scale=0.1, seed=42, weight_decay=0.0, loss_name="mse"):
    return MLP(
        architecture=arch,
        hidden_activation=hidden_act,
        output_activation=out_act,
        beta=beta,
        init_scale=init_scale,
        seed=seed,
        weight_decay=weight_decay,
        loss_name=loss_name,
    )


def train_model(model, data, optimizer, max_epochs=100, batch_size=32,
                early_stopping_patience=15, verbose=False, seed=None, augment_fn=None):
    """seed: controla el orden de mini-batches (debería ser distinto por
    seed en multi-seed sweeps para no subestimar la varianza)."""
    rng = np.random.default_rng(seed)
    X_tr, Y_tr = data["X_train"], data["Y_train"]
    X_va, Y_va = data["X_val"], data["Y_val"]
    y_tr, y_va = data["y_train"], data["y_val"]

    es = EarlyStopping(patience=early_stopping_patience) if early_stopping_patience > 0 else None
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_acc, best_params = -1.0, None
    stopped_at = max_epochs

    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train_epoch(X_tr, Y_tr, optimizer, batch_size=batch_size,
                          shuffle=True, rng=rng, augment_fn=augment_fn)
        tr_pred = model.predict(X_tr)
        va_pred = model.predict(X_va)
        tr_loss = float(model._loss_fn(Y_tr, tr_pred))
        va_loss = float(model._loss_fn(Y_va, va_pred))
        tr_acc = accuracy(y_tr, classify_from_output(tr_pred))
        va_acc = accuracy(y_va, classify_from_output(va_pred))
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_params = copy.deepcopy(model.get_params())
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"    ep{epoch:3d} | trL {tr_loss:.4f} vaL {va_loss:.4f} "
                  f"| trA {tr_acc:.3f} vaA {va_acc:.3f}")
        if es is not None and es(va_loss, model.get_params()):
            stopped_at = epoch
            break
    elapsed = time.time() - t0
    if best_params is not None:
        model.set_params(best_params)
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "stopped_at": stopped_at,
        "elapsed": elapsed,
        "best_val_acc": best_val_acc,
    }


def evaluate_on_test(model, data):
    n_classes = data["n_classes"]
    pred_te = model.predict(data["X_test"])
    pred_va = model.predict(data["X_val"])
    pred_tr = model.predict(data["X_train"])
    y_pred_te = classify_from_output(pred_te)
    y_pred_va = classify_from_output(pred_va)
    y_pred_tr = classify_from_output(pred_tr)
    cm_te = confusion_matrix_multiclass(data["y_test"], y_pred_te, n_classes)
    return {
        "test_acc": accuracy(data["y_test"], y_pred_te),
        "val_acc": accuracy(data["y_val"], y_pred_va),
        "train_acc": accuracy(data["y_train"], y_pred_tr),
        "test_cm": cm_te,
        "test_per_class": per_class_metrics(cm_te),
        "y_pred_test": y_pred_te,
    }
