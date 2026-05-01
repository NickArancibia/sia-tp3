import ast
import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared.config_loader import load_config
from shared.digit_dataset_loader import load_dataset, get_image
from shared.losses import mse, build_loss
from shared.metrics import (
    accuracy,
    classify_from_output,
    confusion_matrix_multiclass,
    per_class_metrics,
)
from shared.mlp import MLP
from shared.optimizers import build_optimizer
from shared.preprocessing import (
    ZScoreScaler,
    build_scaler,
    one_hot_decode,
    one_hot_encode,
    stratified_split,
)
from shared.regularization import EarlyStopping
from shared.utils import (
    plot_accuracy_curves,
    plot_confusion_matrix,
    plot_learning_curves,
    plot_misclassified_samples,
    plot_multi_bar,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def extract_XY(df):
    """Extract feature matrix X and label vector y from a loaded digits DataFrame."""
    X = np.stack(df["image"].values)
    y = df["label"].values.astype(int)
    return X, y


def load_and_preprocess(cfg):
    """Load train/val data from digits.csv and test data from digits_test.csv.

    Returns: X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, scaler, n_classes
    """
    train_df = load_dataset(os.path.join(os.path.dirname(__file__), cfg["data"]["train_path"]))
    test_df = load_dataset(os.path.join(os.path.dirname(__file__), cfg["data"]["test_path"]))

    X_all, y_all = extract_XY(train_df)
    X_test_raw, y_test = extract_XY(test_df)

    split_cfg = cfg["data"]["split"]
    val_frac = split_cfg["val_frac"]
    stratify = split_cfg.get("stratify", True)
    seed = split_cfg.get("seed", 42)

    if stratify:
        train_idx, val_idx, _ = stratified_split(y_all, val_frac, test_frac=0.0, seed=seed)
    else:
        rng = np.random.default_rng(seed)
        idx = np.arange(len(y_all))
        rng.shuffle(idx)
        n_val = int(len(idx) * val_frac)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]

    X_train_raw, y_train = X_all[train_idx], y_all[train_idx]
    X_val_raw, y_val = X_all[val_idx], y_all[val_idx]

    scaler_cfg = cfg["data"]["preprocess"]["feature_scaler"]
    scaler = build_scaler(scaler_cfg)
    if scaler is not None:
        X_train = scaler.fit_transform(X_train_raw)
        X_val = scaler.transform(X_val_raw)
        X_test = scaler.transform(X_test_raw)
    else:
        X_train = X_train_raw
        X_val = X_val_raw
        X_test = X_test_raw

    n_classes = max(y_all.max(), y_test.max()) + 1

    Y_train = one_hot_encode(y_train, n_classes)
    Y_val = one_hot_encode(y_val, n_classes)
    Y_test = one_hot_encode(y_test, n_classes)

    return (
        X_train, Y_train, y_train,
        X_val, Y_val, y_val,
        X_test, Y_test, y_test,
        scaler, n_classes,
    )


def build_model(cfg, n_features, n_classes):
    arch = [n_features] + cfg["model"]["architecture"][1:-1] + [n_classes]
    return MLP(
        architecture=arch,
        hidden_activation=cfg["model"].get("hidden_activation", "tanh"),
        output_activation=cfg["model"].get("output_activation", "logistic"),
        beta=cfg["model"].get("beta", 1.0),
        initializer=cfg["model"].get("initializer", "random_normal"),
        init_scale=cfg["model"].get("init_scale", 0.1),
        seed=cfg["experiment"].get("seed", 42),
        weight_decay=cfg["training"].get("weight_decay", 0.0),
        loss_name=cfg["training"].get("loss", "mse"),
    )


def train_and_evaluate(cfg, verbose=True):
    """Full training pipeline: load data, build model, train, evaluate, save results."""
    exp_name = cfg["experiment"]["name"]
    results_dir = os.path.join(RESULTS_DIR, exp_name)
    os.makedirs(results_dir, exist_ok=True)

    (
        X_train, Y_train, y_train,
        X_val, Y_val, y_val,
        X_test, Y_test, y_test,
        scaler, n_classes,
    ) = load_and_preprocess(cfg)

    n_features = X_train.shape[1]
    model = build_model(cfg, n_features, n_classes)
    optimizer = build_optimizer(cfg["training"])

    max_epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"].get("batch_size", 32)
    if batch_size == 0:
        batch_size = X_train.shape[0]

    es_cfg = cfg["training"].get("early_stopping", {})
    early_stopping = None
    if es_cfg.get("enabled", False):
        patience = es_cfg.get("patience", 20)
        early_stopping = EarlyStopping(patience=patience)

    log_every = cfg["experiment"].get("log_every", 1)
    seed = cfg["experiment"].get("seed", 42)
    rng = np.random.default_rng(seed)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    loss_fn, _ = build_loss(cfg["training"].get("loss", "mse"))

    if verbose:
        print(f"Experiment: {exp_name}")
        print(f"Architecture: {model.architecture}")
        print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        print(f"Optimizer: {cfg['training']['optimizer']}, LR: {cfg['training']['learning_rate']}")
        print(f"Batch size: {batch_size}, Max epochs: {max_epochs}")
        print()

    start_time = time.time()
    best_val_loss = np.inf
    best_params = None

    for epoch in range(1, max_epochs + 1):
        model.train_epoch(X_train, Y_train, optimizer, batch_size=batch_size,
                         shuffle=True, rng=rng)

        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)

        tr_loss = float(loss_fn(Y_train, train_pred))
        va_loss = float(loss_fn(Y_val, val_pred))
        tr_acc = accuracy(y_train, classify_from_output(train_pred))
        va_acc = accuracy(y_val, classify_from_output(val_pred))

        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_params = copy.deepcopy(model.get_params())

        if verbose and (epoch % log_every == 0 or epoch == 1):
            print(f"Epoch {epoch:4d} | Train Loss: {tr_loss:.6f} | Val Loss: {va_loss:.6f} | "
                  f"Train Acc: {tr_acc:.4f} | Val Acc: {va_acc:.4f}")

        if early_stopping is not None:
            if early_stopping(va_loss, model.get_params()):
                if verbose:
                    print(f"Early stopping at epoch {epoch} (patience={early_stopping.patience})")
                break

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nTraining completed in {elapsed:.1f}s ({epoch} epochs)")

    if best_params is not None:
        model.set_params(best_params)

    # Final evaluation on all sets
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    train_y_pred = classify_from_output(train_pred)
    val_y_pred = classify_from_output(val_pred)
    test_y_pred = classify_from_output(test_pred)

    train_acc_final = accuracy(y_train, train_y_pred)
    val_acc_final = accuracy(y_val, val_y_pred)
    test_acc_final = accuracy(y_test, test_y_pred)

    train_cm = confusion_matrix_multiclass(y_train, train_y_pred, n_classes)
    test_cm = confusion_matrix_multiclass(y_test, test_y_pred, n_classes)
    test_metrics = per_class_metrics(test_cm)

    if verbose:
        print(f"\n=== Final Results ===")
        print(f"Train Accuracy: {train_acc_final:.4f}")
        print(f"Val Accuracy:   {val_acc_final:.4f}")
        print(f"Test Accuracy:  {test_acc_final:.4f}")
        print(f"\nPer-class metrics (test):")
        for c in range(n_classes):
            p = test_metrics["precision"][c]
            r = test_metrics["recall"][c]
            f = test_metrics["f1"][c]
            print(f"  Digit {c}: P={p:.4f} R={r:.4f} F1={f:.4f}")
        print(f"\nMacro F1:  {test_metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {test_metrics['weighted_f1']:.4f}")

    # Save results
    save_cfg = cfg["experiment"]

    if save_cfg.get("save_plots", True):
        plot_learning_curves(train_losses, val_losses,
                            title=f"{exp_name} - Loss",
                            path=os.path.join(results_dir, "loss_curve.png"))
        plot_accuracy_curves(train_accs, val_accs,
                            title=f"{exp_name} - Accuracy",
                            path=os.path.join(results_dir, "accuracy_curve.png"))
        digit_labels = [str(i) for i in range(n_classes)]
        plot_confusion_matrix(test_cm,
                             labels=digit_labels,
                             title=f"{exp_name} - Confusion Matrix (Test)",
                             path=os.path.join(results_dir, "confusion_matrix.png"))
        if X_test.shape[1] == 784:
            plot_misclassified_samples(X_test, y_test, test_y_pred, n=16,
                                     path=os.path.join(results_dir, "misclassified.png"))

    if save_cfg.get("save_model", True):
        model_path = os.path.join(results_dir, "model.npz")
        model.save(model_path)
        if scaler is not None:
            scaler.save(os.path.join(results_dir, "scaler.npz"))

    # Save training log
    log_df = pd.DataFrame({
        "epoch": range(1, len(train_losses) + 1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
    })
    log_df.to_csv(os.path.join(results_dir, "train_log.csv"), index=False)

    # Save summary
    summary = {
        "experiment": exp_name,
        "architecture": model.architecture,
        "optimizer": cfg["training"]["optimizer"],
        "learning_rate": cfg["training"]["learning_rate"],
        "batch_size": batch_size,
        "epochs_trained": len(train_losses),
        "train_acc": train_acc_final,
        "val_acc": val_acc_final,
        "test_acc": test_acc_final,
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
        "elapsed_s": elapsed,
    }

    results = {
        "model": model,
        "cfg": cfg,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "test_cm": test_cm,
        "test_metrics": test_metrics,
        "test_acc": test_acc_final,
        "summary": summary,
        "n_classes": n_classes,
    }

    return results


def run_multi_seed(cfg, seeds=None):
    """Run the same experiment with multiple seeds and return aggregated results."""
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    all_train_losses = []
    all_val_losses = []
    all_train_accs = []
    all_val_accs = []
    all_test_accs = []
    all_macro_f1s = []
    all_weighted_f1s = []

    exp_name = cfg["experiment"]["name"]
    results_dir = os.path.join(RESULTS_DIR, f"{exp_name}_multiseed")
    os.makedirs(results_dir, exist_ok=True)

    for s in seeds:
        print(f"\n{'='*50}")
        print(f"Running seed {s}")
        print(f"{'='*50}")

        cfg_copy = copy.deepcopy(cfg)
        cfg_copy["experiment"]["seed"] = s
        cfg_copy["data"]["split"]["seed"] = s
        cfg_copy["experiment"]["name"] = f"{exp_name}_seed{s}"

        results = train_and_evaluate(cfg_copy, verbose=True)

        all_train_losses.append(results["train_losses"])
        all_val_losses.append(results["val_losses"])
        all_train_accs.append(results["train_accs"])
        all_val_accs.append(results["val_accs"])
        all_test_accs.append(results["test_acc"])
        all_macro_f1s.append(results["test_metrics"]["macro_f1"])
        all_weighted_f1s.append(results["test_metrics"]["weighted_f1"])

    mean_test_acc = np.mean(all_test_accs)
    std_test_acc = np.std(all_test_accs)
    mean_macro_f1 = np.mean(all_macro_f1s)
    std_macro_f1 = np.std(all_macro_f1s)

    print(f"\n{'='*50}")
    print(f"AGGREGATED RESULTS ({len(seeds)} seeds)")
    print(f"{'='*50}")
    print(f"Test Accuracy: {mean_test_acc:.4f} +/- {std_test_acc:.4f}")
    print(f"Macro F1:      {mean_macro_f1:.4f} +/- {std_macro_f1:.4f}")

    plot_learning_curves(all_train_losses, all_val_losses,
                        title=f"{exp_name} - Loss ({len(seeds)} seeds)",
                        path=os.path.join(results_dir, "loss_curve.png"))
    plot_accuracy_curves(all_train_accs, all_val_accs,
                        title=f"{exp_name} - Accuracy ({len(seeds)} seeds)",
                        path=os.path.join(results_dir, "accuracy_curve.png"))

    bar_data = {}
    for i, s in enumerate(seeds):
        bar_data[f"seed{s}"] = (all_test_accs[i], 0.0)
    plot_multi_bar(bar_data, title=f"{exp_name} - Test Accuracy per Seed",
                  path=os.path.join(results_dir, "accuracy_per_seed.png"), ylabel="Accuracy")

    return {
        "seeds": seeds,
        "test_acc_mean": mean_test_acc,
        "test_acc_std": std_test_acc,
        "macro_f1_mean": mean_macro_f1,
        "macro_f1_std": std_macro_f1,
        "all_train_losses": all_train_losses,
        "all_val_losses": all_val_losses,
        "all_train_accs": all_train_accs,
        "all_val_accs": all_val_accs,
    }


def evaluate_saved_model(model_dir, test_csv_path=None, verbose=True):
    """Load a saved model + scaler and evaluate on test data.

    model_dir: path to results/<experiment_name>/ containing model.npz and scaler.npz
    test_csv_path: path to test CSV (defaults to data/digits_test.csv)
    """
    model_path = os.path.join(model_dir, "model.npz")
    scaler_path = os.path.join(model_dir, "scaler.npz")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}")

    model = MLP.load(model_path)

    if test_csv_path is None:
        test_csv_path = os.path.join(os.path.dirname(__file__), "data", "digits_test.csv")

    test_df = load_dataset(test_csv_path)
    X_test_raw, y_test = extract_XY(test_df)

    if os.path.exists(scaler_path):
        from shared.preprocessing import ZScoreScaler as _ZS, MinMaxScaler as _MM
        try:
            scaler = ZScoreScaler.load(scaler_path)
        except KeyError:
            scaler = MinMaxScaler.load(scaler_path)
        X_test = scaler.transform(X_test_raw)
    else:
        X_test = X_test_raw

    n_classes = max(model.architecture[-1], y_test.max() + 1)
    Y_test = one_hot_encode(y_test, n_classes)

    pred = model.predict(X_test)
    y_pred = classify_from_output(pred)

    acc = accuracy(y_test, y_pred)
    cm = confusion_matrix_multiclass(y_test, y_pred, n_classes)
    metrics = per_class_metrics(cm)
    loss_fn, _ = build_loss("mse")
    loss = loss_fn(Y_test, pred)

    if verbose:
        print(f"=== Evaluation of saved model ===")
        print(f"Model: {model_path}")
        print(f"Architecture: {model.architecture}")
        print(f"Test samples: {len(y_test)}")
        print(f"\nTest Loss: {loss:.6f}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"\nPer-class metrics:")
        for c in range(n_classes):
            p = metrics["precision"][c]
            r = metrics["recall"][c]
            f = metrics["f1"][c]
            print(f"  Digit {c}: P={p:.4f} R={r:.4f} F1={f:.4f}")
        print(f"\nMacro F1:  {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    return {
        "accuracy": acc,
        "loss": loss,
        "confusion_matrix": cm,
        "per_class": metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ej2 - Digit Classification with MLP")
    parser.add_argument("--eval", type=str, default=None,
                       help="Path to saved model directory to evaluate (e.g. results/ej2_baseline_tanh_40_20)")
    parser.add_argument("--test", type=str, default=None,
                       help="Path to test CSV for evaluation")
    args = parser.parse_args()

    if args.eval:
        evaluate_saved_model(args.eval, test_csv_path=args.test, verbose=True)
    else:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        cfg = load_config(config_path)

        seeds = cfg["experiment"].get("seeds", None)
        if seeds is not None:
            results = run_multi_seed(cfg, seeds=seeds)
        else:
            results = train_and_evaluate(cfg, verbose=True)
