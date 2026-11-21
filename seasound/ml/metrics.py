# -*- coding: utf-8 -*-
"""ml/metrics.py — 세그먼트/녹음 단위 지표 (torch 없이 numpy·sklearn만 사용)."""
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def recording_level(probs, rec_ids, labels):
    """세그먼트 확률을 녹음별로 평균 (명세: 녹음 단위 판정 = 세그먼트 확률 평균)."""
    rec_ids = np.asarray(rec_ids)
    uniq, inv = np.unique(rec_ids, return_inverse=True)
    rec_probs = np.zeros((len(uniq), probs.shape[1]), dtype=np.float64)
    np.add.at(rec_probs, inv, probs)
    counts = np.bincount(inv, minlength=len(uniq))[:, None]
    rec_probs /= counts
    rec_labels = np.zeros(len(uniq), dtype=np.int64)
    rec_labels[inv] = np.asarray(labels)
    return uniq, rec_probs, rec_labels


def report(y_true, y_pred, class_names):
    idx = list(range(len(class_names)))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=idx, zero_division=0)
    present = sup > 0
    per_class = [{"class": c, "precision": float(p[i]), "recall": float(r[i]),
                  "f1": float(f1[i]), "support": int(sup[i])}
                 for i, c in enumerate(class_names)]
    return {
        "accuracy": float(np.mean(np.asarray(y_true) == np.asarray(y_pred))) if len(y_true) else 0.0,
        # 평가셋에 있는 클래스만으로 낸 macro F1 (주 지표). 없는 클래스는 0으로 깎이지 않게.
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "macro_f1_all_classes": float(f1.mean()),
        "classes_missing": [c for i, c in enumerate(class_names) if not present[i]],
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=idx).tolist(),
    }
