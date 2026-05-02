"""Infraestructura común para los experimentos de EJ2 part2.

Provee:
- prepare_data(cfg): carga digits.csv + digits_test.csv, splittea train/val,
  aplica scaler. n_classes se computa del UNION de y_train ∪ y_test, así que
  el modelo siempre tiene 10 outputs aunque train no tenga la clase 8.
- train_model(model_cfg, training_cfg, data, seed): entrena un MLP con early
  stopping, devuelve historial completo (loss, acc por época) y mejor val acc.
- evaluate_on_test(model, data): accuracy + confusion + per-class.
- run_grid(grid, base_cfg, seeds): corre N combinaciones x M seeds, retorna
  lista de resultados.
"""
import copy
import os
import sys
import time

EJ2_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(EJ2_DIR)
sys.path.insert(0, REPO_DIR)

# Paths centralizados — los run.py/plot.py los importan en lugar de
# recalcular relative dirs (frágil).
TRAIN_CSV = os.path.join(EJ2_DIR, "data", "digits.csv")
TEST_CSV = os.path.join(EJ2_DIR, "data", "digits_test.csv")
RESULTS_PART2 = os.path.join(EJ2_DIR, "results", "part2")

import numpy as np

from shared.digit_dataset_loader import load_dataset
from shared.metrics import (accuracy, classify_from_output,
                            confusion_matrix_multiclass, per_class_metrics)
from shared.mlp import MLP
from shared.optimizers import build_optimizer
from shared.preprocessing import (build_scaler, one_hot_encode, stratified_split)
from shared.regularization import EarlyStopping


def prepare_data(train_csv, test_csv, val_frac=0.2, scaler="z-score",
                 stratify=True, seed=42):
    """Load digits CSVs and produce train/val/test splits + scaling.

    Returns dict con X_train, X_val, X_test, Y_train (one-hot), Y_val, Y_test,
    y_train, y_val, y_test, scaler, n_classes.

    n_classes = max(train_max, test_max) + 1, así que si test tiene 10 clases
    pero train sólo tiene 9 (digits.csv carece del 8), igual el modelo va a
    tener 10 outputs.
    """
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
              init_scale=0.1, seed=42, weight_decay=0.0):
    """Build an MLP. Wrapper para construir desde un config flat."""
    return MLP(
        architecture=arch,
        hidden_activation=hidden_act,
        output_activation=out_act,
        beta=beta,
        init_scale=init_scale,
        seed=seed,
        weight_decay=weight_decay,
    )


def train_model(model, data, optimizer, max_epochs=100, batch_size=32,
                early_stopping_patience=15, verbose=False, seed=None):
    """Train one model. Returns dict con todo el historial.

    {train_losses, val_losses, train_accs, val_accs, stopped_at, elapsed,
     best_val_acc, best_params}

    seed: si se pasa, controla el orden de mini-batches por seed (importante
    para que la varianza entre seeds en multi-seed sweeps NO esté
    artificialmente subestimada por compartir el orden de batches).
    """
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
                          shuffle=True, rng=rng)
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
    """Evaluate on test set. Returns dict con test_acc, train_acc, val_acc,
    cm (test), per_class (test)."""
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


def aggregate(per_seed_results, key):
    """Helper para agregar (mean, std) a través de los seeds."""
    vals = [r[key] for r in per_seed_results]
    return float(np.mean(vals)), float(np.std(vals))
