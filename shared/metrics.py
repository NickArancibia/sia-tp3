import numpy as np


def confusion_matrix(y_true, y_pred):
    """Returns [[TN, FP], [FN, TP]]."""
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return np.array([[tn, fp], [fn, tp]])


def confusion_matrix_multiclass(y_true, y_pred, n_classes=None):
    """Return n_classes x n_classes confusion matrix.

    cm[i, j] = number of samples with true class i predicted as class j.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if n_classes is None:
        n_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return float(precision), float(recall), float(f1)


def per_class_metrics(cm):
    """Compute per-class precision, recall, F1 from confusion matrix.

    cm: (C, C) confusion matrix
    Returns: dict with keys 'precision', 'recall', 'f1', each an array of length C
    """
    n_classes = cm.shape[0]
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1[c] = (2 * precision[c] * recall[c] / (precision[c] + recall[c])
                 if (precision[c] + recall[c]) > 0 else 0.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "weighted_precision": float(
            np.average(precision, weights=cm.sum(axis=1))
            if cm.sum() > 0 else 0.0
        ),
        "weighted_recall": float(
            np.average(recall, weights=cm.sum(axis=1))
            if cm.sum() > 0 else 0.0
        ),
        "weighted_f1": float(
            np.average(f1, weights=cm.sum(axis=1))
            if cm.sum() > 0 else 0.0
        ),
    }


def classify_from_output(output):
    """Convert MLP output (probabilities or scores) to class labels via argmax."""
    return np.argmax(output, axis=1)


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
    """Returns (precisions, recalls) sorted by descending threshold."""
    thresholds = np.sort(np.unique(y_scores))[::-1]
    precisions, recalls = [1.0], [0.0]
    for t in thresholds:
        pred = (y_scores >= t).astype(int)
        p, r, _ = precision_recall_f1(y_true, pred)
        precisions.append(p)
        recalls.append(r)
    return np.array(precisions), np.array(recalls)


def auc(x, y):
    order = np.argsort(x)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(y[order], x[order]))


def threshold_sweep(y_true, y_scores, n_points=300):
    """Compute precision/recall/F1 over a grid of thresholds."""
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
