"""Shared helpers for ej2 experiments. Import from each run.py."""
import copy
import csv
import os
import time

import numpy as np
import pandas as pd

from shared.digit_dataset_loader import load_dataset
from shared.losses import build_loss
from shared.metrics import (
    accuracy,
    classify_from_output,
    confusion_matrix_multiclass,
    per_class_metrics,
)
from shared.mlp import MLP
from shared.optimizers import build_optimizer
from shared.preprocessing import ZScoreScaler, one_hot_encode, stratified_split
from shared.regularization import EarlyStopping


def batch_label(bs):
    if bs == 1:
        return "online"
    if bs <= 0:
        return "full"
    return f"mini{bs}"


def load_raw(cfg, ej2_dir):
    train_df = load_dataset(os.path.join(ej2_dir, cfg["data"]["train_path"]))
    test_df = load_dataset(os.path.join(ej2_dir, cfg["data"]["test_path"]))
    X_all = np.stack(train_df["image"].values)
    y_all = train_df["label"].values.astype(int)
    X_test_raw = np.stack(test_df["image"].values)
    y_test = test_df["label"].values.astype(int)
    n_classes = max(y_all.max(), y_test.max()) + 1
    return X_all, y_all, X_test_raw, y_test, n_classes


def split_scale(X_all, y_all, X_test_raw, cfg, seed):
    val_frac = cfg["data"]["split"]["val_frac"]
    train_idx, val_idx, _ = stratified_split(y_all, val_frac, test_frac=0.0, seed=seed)
    scaler = ZScoreScaler()
    X_tr = scaler.fit_transform(X_all[train_idx])
    X_va = scaler.transform(X_all[val_idx])
    X_te = scaler.transform(X_test_raw)
    return X_tr, y_all[train_idx], X_va, y_all[val_idx], X_te


def train(X_tr, y_tr, X_va, y_va, X_te, y_te, n_classes,
          full_arch, hidden_act, opt_cfg, batch_size, seed, max_epochs, patience):
    """Train one MLP run. Returns metrics and per-epoch loss curves. Saves nothing."""
    Y_tr = one_hot_encode(y_tr, n_classes)
    Y_va = one_hot_encode(y_va, n_classes)
    loss_fn, _ = build_loss("mse")
    model = MLP(
        architecture=full_arch,
        hidden_activation=hidden_act,
        output_activation="logistic",
        beta=1.0,
        initializer="random_normal",
        init_scale=0.1,
        seed=seed,
    )
    opt = build_optimizer(opt_cfg)
    es = EarlyStopping(patience=patience)
    rng = np.random.default_rng(seed)
    eff_bs = X_tr.shape[0] if batch_size <= 0 else batch_size
    train_losses, val_losses = [], []

    t0 = time.perf_counter()
    for _ in range(max_epochs):
        model.train_epoch(X_tr, Y_tr, opt, batch_size=eff_bs, shuffle=True, rng=rng)
        tr_l = float(loss_fn(Y_tr, model.predict(X_tr)))
        va_l = float(loss_fn(Y_va, model.predict(X_va)))
        train_losses.append(tr_l)
        val_losses.append(va_l)
        if es(va_l, model.get_params()):
            break

    elapsed_s = time.perf_counter() - t0

    if es.best_params is not None:
        model.set_params(es.best_params)

    te_pred = classify_from_output(model.predict(X_te))
    va_pred = classify_from_output(model.predict(X_va))
    tr_pred = classify_from_output(model.predict(X_tr))
    cm = confusion_matrix_multiclass(y_te, te_pred, n_classes)

    return {
        "train_acc": accuracy(y_tr, tr_pred),
        "val_acc": accuracy(y_va, va_pred),
        "test_acc": accuracy(y_te, te_pred),
        "macro_f1": per_class_metrics(cm)["macro_f1"],
        "epochs": len(train_losses),
        "elapsed_s": elapsed_s,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }


def append_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def best_lr_for_source(summary_df, lr_source):
    """Find the best learning rate from batch_lr summary for a given lr_source."""
    source_to_bs = {"online": 1, "mini32": 32, "mini64": 64, "mini128": 128, "full": -1}
    if lr_source == "global":
        row = summary_df.loc[summary_df["mean_val_acc"].idxmax()]
    else:
        bs = source_to_bs[lr_source]
        subset = summary_df[summary_df["batch_size"] == bs]
        row = subset.loc[subset["mean_val_acc"].idxmax()]
    return float(row["learning_rate"])


def summarize_group(raw_df, config_name, extra_fields=None):
    """Aggregate per-seed rows into mean ± std summary."""
    r = raw_df[raw_df["config_name"] == config_name]
    s = {
        "config_name": config_name,
        "mean_val_acc": float(r["val_acc"].mean()),
        "std_val_acc": float(r["val_acc"].std(ddof=0)),
        "mean_test_acc": float(r["test_acc"].mean()),
        "std_test_acc": float(r["test_acc"].std(ddof=0)),
        "mean_macro_f1": float(r["macro_f1"].mean()),
        "std_macro_f1": float(r["macro_f1"].std(ddof=0)),
        "mean_epochs": float(r["epochs"].mean()),
    }
    if "elapsed_s" in r.columns:
        s["mean_elapsed_s"] = float(r["elapsed_s"].mean())
        s["std_elapsed_s"] = float(r["elapsed_s"].std(ddof=0))
    if extra_fields:
        for field in extra_fields:
            s[field] = r[field].iloc[0]
    return s
