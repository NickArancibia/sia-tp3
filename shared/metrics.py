import numpy as np


def confusion_matrix(y_true, y_pred):
    """Returns [[TN, FP], [FN, TP]]."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(precision), float(recall), float(f1)


def roc_curve(y_true, y_scores):
    """Returns (fpr, tpr) arrays sorted by fpr."""
    thresholds = np.sort(np.unique(y_scores))[::-1]
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    fprs, tprs = [0.0], [0.0]
    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fprs.append(float(fp / neg) if neg > 0 else 0.0)
        tprs.append(float(tp / pos) if pos > 0 else 0.0)
    fprs.append(1.0)
    tprs.append(1.0)
    return np.array(fprs), np.array(tprs)


def pr_curve(y_true, y_scores):
    """Returns (precisions, recalls) sorted by descending threshold.

    Includes the τ=+∞ extreme (R=0, P=1) at the start — by convention
    precision is 1 when no positives are predicted (0/0). The τ=−∞ extreme
    (R=1, P=base_rate) is already captured naturally when t = min(scores),
    since (scores >= t) is all-True there.
    """
    thresholds = np.sort(np.unique(y_scores))[::-1]
    precisions, recalls = [1.0], [0.0]  # τ=+∞ extreme
    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        p, r, _ = precision_recall_f1(y_true, pred)
        precisions.append(p)
        recalls.append(r)
    return np.array(precisions), np.array(recalls)


def auc(x, y):
    # TODO(A1): AUC-PR via trapezoid is not the standard definition. The PR
    # curve is non-monotonic in precision so trapezoid + argsort can over- or
    # under-estimate the area. The defensible alternative is the
    # average-precision step formula:  AP = Σ (R_i − R_{i−1}) · P_i  ordered
    # by ascending recall. Replace this `auc(recs, precs)` call site once we
    # ship a dedicated `average_precision(...)` helper.
    order = np.argsort(x)
    return float(np.trapezoid(y[order], x[order]))


def threshold_sweep(y_true, y_scores, n_points=300):
    """Compute precision/recall/F1 over a grid of thresholds.

    Returns (thresholds, precisions, recalls, f1s, best_threshold).
    best_threshold maximises F1.
    """
    # Combine unique predicted values with a linspace for smooth coverage
    unique = np.sort(np.unique(y_scores))
    linspace = np.linspace(float(y_scores.min()), float(y_scores.max()), n_points)
    thresholds = np.sort(np.unique(np.concatenate([unique, linspace])))

    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        p, r, f1 = precision_recall_f1(y_true, pred)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    best_idx = int(np.argmax(f1s))
    return thresholds, precisions, recalls, f1s, float(thresholds[best_idx])
