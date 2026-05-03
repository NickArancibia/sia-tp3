"""Experimento 5: entrena el mejor modelo encontrado en los experimentos anteriores.

Lee los summary.csv de architecture, activation y batch_lr para determinar
la mejor configuración. Corre 5 seeds, guarda el modelo con mejor val_acc.

Produce:
  results/part/best_model/raw.csv      — métricas por seed
  results/part/best_model/summary.csv  — media ± std global
  results/part/best_model/model.npz    — pesos del mejor modelo
  results/part/best_model/scaler.npz   — scaler del mejor modelo
  results/part/best_model/config.yaml  — hiperparámetros exactos usados
"""
import ast
import copy
import os
import sys

EJ2_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPO_DIR = os.path.dirname(EJ2_DIR)
PART_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EJ2_DIR)
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, PART_DIR)

import numpy as np
import pandas as pd
import yaml

from _common import append_csv, best_lr_for_source, load_raw, split_scale, train
from shared.config_loader import load_config
from shared.losses import build_loss
from shared.mlp import MLP
from shared.optimizers import build_optimizer
from shared.preprocessing import ZScoreScaler, one_hot_encode, stratified_split
from shared.regularization import EarlyStopping

RESULTS_DIR = os.path.join(EJ2_DIR, "results", "part", "best_model")
BATCH_LR_SUMMARY = os.path.join(EJ2_DIR, "results", "part", "batch_lr", "summary.csv")
ARCH_SUMMARY = os.path.join(EJ2_DIR, "results", "part", "architecture", "summary.csv")
ACTIV_SUMMARY = os.path.join(EJ2_DIR, "results", "part", "activation", "summary.csv")
OPTIMIZER_SUMMARY = os.path.join(EJ2_DIR, "results", "part", "optimizer", "summary.csv")

RAW_FIELDS = [
    "seed", "architecture", "hidden_activation", "optimizer",
    "batch_size", "learning_rate",
    "train_acc", "val_acc", "test_acc", "macro_f1", "epochs",
]
SUMMARY_FIELDS = [
    "architecture", "hidden_activation", "optimizer", "batch_size", "learning_rate",
    "mean_val_acc", "std_val_acc", "mean_test_acc", "std_test_acc",
    "mean_macro_f1", "std_macro_f1", "mean_epochs",
]


def _resolve_best_config(cfg, n_features, n_classes):
    arch_summary = pd.read_csv(ARCH_SUMMARY) if os.path.exists(ARCH_SUMMARY) else None
    activ_summary = pd.read_csv(ACTIV_SUMMARY) if os.path.exists(ACTIV_SUMMARY) else None

    # Use Adam lr from optimizer experiment (batch_lr used GD, so its lr is incompatible)
    if os.path.exists(OPTIMIZER_SUMMARY):
        opt_df = pd.read_csv(OPTIMIZER_SUMMARY)
        adam_rows = opt_df[opt_df["optimizer"] == "adam"]
        lr = float(adam_rows.loc[adam_rows["mean_val_acc"].idxmax(), "learning_rate"])
        print(f"LR (experimento optimizer/Adam):  {lr:.0e}")
    else:
        lr = 0.001
        print(f"LR (default):                     {lr:.0e}")
    bs = 32

    if arch_summary is not None:
        best_row = arch_summary.loc[arch_summary["mean_test_acc"].idxmax()]
        full_arch = ast.literal_eval(best_row["architecture"])
        print(f"Arquitectura (experimento arch):  {full_arch}")
    else:
        hidden = cfg["model"]["architecture"][1:-1]
        full_arch = [n_features] + hidden + [n_classes]
        print(f"Arquitectura (default config):    {full_arch}")

    if activ_summary is not None:
        best_row = activ_summary.loc[activ_summary["mean_test_acc"].idxmax()]
        hidden_act = best_row["hidden_activation"]
        print(f"Activación oculta (experimento):  {hidden_act}")
    else:
        hidden_act = cfg["model"].get("hidden_activation", "tanh")
        print(f"Activación oculta (default):      {hidden_act}")

    return full_arch, hidden_act, lr, bs


def _retrain_and_save(X_all, y_all, cfg, seed, n_classes,
                      full_arch, hidden_act, opt_cfg, bs, max_epochs, patience):
    """Reentrena con la seed indicada y persiste model.npz + scaler.npz."""
    val_frac = cfg["data"]["split"]["val_frac"]
    train_idx, val_idx, _ = stratified_split(y_all, val_frac, test_frac=0.0, seed=seed)

    scaler = ZScoreScaler()
    X_tr = scaler.fit_transform(X_all[train_idx])
    X_va = scaler.transform(X_all[val_idx])
    y_tr = y_all[train_idx]
    y_va = y_all[val_idx]

    Y_tr = one_hot_encode(y_tr, n_classes)
    Y_va = one_hot_encode(y_va, n_classes)
    loss_fn, _ = build_loss("mse")

    model = MLP(architecture=full_arch, hidden_activation=hidden_act,
                output_activation="logistic", beta=1.0,
                initializer="random_normal", init_scale=0.1, seed=seed)
    opt = build_optimizer(opt_cfg)
    es = EarlyStopping(patience=patience)
    rng = np.random.default_rng(seed)
    eff_bs = X_tr.shape[0] if bs <= 0 else bs

    for _ in range(max_epochs):
        model.train_epoch(X_tr, Y_tr, opt, batch_size=eff_bs, shuffle=True, rng=rng)
        va_l = float(loss_fn(Y_va, model.predict(X_va)))
        if es(va_l, model.get_params()):
            break

    if es.best_params is not None:
        model.set_params(es.best_params)

    model.save(os.path.join(RESULTS_DIR, "model.npz"))
    scaler.save(os.path.join(RESULTS_DIR, "scaler.npz"))


def main():
    cfg = load_config(os.path.join(EJ2_DIR, "config.yaml"))
    seeds = [int(s) for s in cfg["search"]["seeds"]]
    max_epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping"].get("patience", 20)
    adam_betas = cfg["training"].get("adam_betas", [0.9, 0.999])
    adam_eps = cfg["training"].get("adam_eps", 1e-8)

    X_all, y_all, X_test_raw, y_test, n_classes = load_raw(cfg, EJ2_DIR)
    n_features = X_all.shape[1]

    full_arch, hidden_act, lr, bs = _resolve_best_config(cfg, n_features, n_classes)
    opt_cfg = {
        "optimizer": "adam",
        "learning_rate": lr,
        "adam_betas": adam_betas,
        "adam_eps": adam_eps,
    }

    print(f"\nConfiguración final:")
    print(f"  arch={full_arch}")
    print(f"  act={hidden_act}, optimizer=adam, lr={lr:.0e}, batch={bs}")
    print(f"  seeds={seeds}\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_path = os.path.join(RESULTS_DIR, "raw.csv")
    summary_path = os.path.join(RESULTS_DIR, "summary.csv")
    for p in (raw_path, summary_path):
        if os.path.exists(p):
            os.remove(p)

    best_val_acc = -1.0
    best_seed = None

    for seed in seeds:
        print(f"Seed {seed}...")
        X_tr, y_tr, X_va, y_va, X_te = split_scale(X_all, y_all, X_test_raw, cfg, seed)
        result = train(X_tr, y_tr, X_va, y_va, X_te, y_test, n_classes,
                       full_arch, hidden_act, opt_cfg, bs, seed, max_epochs, patience)

        append_csv(raw_path, [{
            "seed": seed,
            "architecture": str(full_arch),
            "hidden_activation": hidden_act,
            "optimizer": "adam",
            "batch_size": bs,
            "learning_rate": lr,
            "train_acc": result["train_acc"],
            "val_acc": result["val_acc"],
            "test_acc": result["test_acc"],
            "macro_f1": result["macro_f1"],
            "epochs": result["epochs"],
        }], RAW_FIELDS)

        print(f"  val_acc={result['val_acc']:.4f}  test_acc={result['test_acc']:.4f}  "
              f"macro_f1={result['macro_f1']:.4f}  epochs={result['epochs']}")

        if result["val_acc"] > best_val_acc:
            best_val_acc = result["val_acc"]
            best_seed = seed
            _retrain_and_save(X_all, y_all, cfg, seed, n_classes,
                              full_arch, hidden_act, opt_cfg, bs, max_epochs, patience)
            print(f"  → nuevo mejor modelo guardado (seed={seed})")

    raw_df = pd.read_csv(raw_path)
    summary = {
        "architecture": str(full_arch),
        "hidden_activation": hidden_act,
        "optimizer": "adam",
        "batch_size": bs,
        "learning_rate": lr,
        "mean_val_acc": float(raw_df["val_acc"].mean()),
        "std_val_acc": float(raw_df["val_acc"].std(ddof=0)),
        "mean_test_acc": float(raw_df["test_acc"].mean()),
        "std_test_acc": float(raw_df["test_acc"].std(ddof=0)),
        "mean_macro_f1": float(raw_df["macro_f1"].mean()),
        "std_macro_f1": float(raw_df["macro_f1"].std(ddof=0)),
        "mean_epochs": float(raw_df["epochs"].mean()),
    }
    pd.DataFrame([summary], columns=SUMMARY_FIELDS).to_csv(summary_path, index=False)

    with open(os.path.join(RESULTS_DIR, "config.yaml"), "w") as f:
        yaml.dump({
            "architecture": full_arch,
            "hidden_activation": hidden_act,
            "output_activation": "logistic",
            "optimizer": "adam",
            "learning_rate": lr,
            "batch_size": bs,
            "adam_betas": adam_betas,
            "adam_eps": adam_eps,
            "best_seed": best_seed,
            "seeds_run": seeds,
        }, f, default_flow_style=False)

    print(f"\n{'='*50}")
    print(f"RESULTADOS FINALES ({len(seeds)} seeds)")
    print(f"{'='*50}")
    print(f"Test accuracy : {summary['mean_test_acc']:.4f} ± {summary['std_test_acc']:.4f}")
    print(f"Macro F1      : {summary['mean_macro_f1']:.4f} ± {summary['std_macro_f1']:.4f}")
    print(f"Mejor seed    : {best_seed}  (val_acc={best_val_acc:.4f})")
    print(f"Modelo        : {RESULTS_DIR}/model.npz")
    print(f"Scaler        : {RESULTS_DIR}/scaler.npz")


if __name__ == "__main__":
    main()
