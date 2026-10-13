"""Train the text sensitivity classifier and evaluate it honestly.

Model:  [TF-IDF word 1-2 grams  +  TF-IDF char 3-5 grams  +  PII/keyword counts]
        -> Logistic Regression (class_weight='balanced')

Evaluation design (the important part):
  * Rows are split by TEMPLATE with GroupShuffleSplit, so no sentence wording in
    the test set was ever seen during training. A random row split would let the
    model memorise wordings and report fake ~100% accuracy.
  * Three disjoint parts: train (fit weights) / validation (pick the decision
    threshold) / test (report final numbers, touched once).
  * The threshold is chosen for recall: missing a sensitive image is worse than a
    false alarm, so we take the highest threshold that keeps recall >= TARGET_RECALL.
  * GroupKFold cross-validation gives mean +/- std, so results aren't one lucky split.
  * A separate list of realistic harmless sentences full of numbers measures the
    false-alarm rate that the old regex-only approach suffered from.
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline

from config import Config
from pattern_detector import PatternDetector

# Harmless, real-world-looking text that the OLD system flagged as sensitive
# (mostly because any 4-digit number counted as an "ATM PIN").
HARD_NEGATIVES = [
    "Happy New Year 2025", "Total due: 1250 dollars", "Meeting moved to room 1205",
    "Flight AI 302 departs 0730", "Order 48213 will arrive on Friday",
    "Invoice 2025-03-14 total 1250.00", "Call 1800 on Monday for the event",
    "Paid by card XXXX XXXX XXXX 4821", "Population 2011 census 12.5 lakh",
    "Room 2104 Brickyard building ASU", "Bus 72 every 15 minutes",
    "Version 3.10.4 release notes", "Score 2 - 1 final", "Steps today 10432",
]


def build_pipeline():
    return Pipeline([
        ("features", FeatureUnion([
            ("words", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                      max_features=5000, sublinear_tf=True)),
            # character n-grams are robust to OCR typos ("Passw0rd", "AADHAR")
            ("chars", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                      max_features=20000, sublinear_tf=True)),
            ("patterns", PatternDetector()),
        ])),
        ("clf", LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000)),
    ])


def pick_threshold(y_true, probs, target_recall):
    """Highest threshold whose recall is still >= target (fewer false alarms)."""
    best = 0.5
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        if recall_score(y_true, probs >= t, zero_division=0) >= target_recall:
            best = t
    return float(best)


def scores(y_true, probs, threshold):
    pred = (probs >= threshold).astype(int)
    return {
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, probs), 4),
    }


def main():
    data = pd.read_csv(Config.TEXT_DATASET_PATH).drop_duplicates("text")
    X = data["text"].values
    y = (data["label"] == "Sensitive").astype(int).values
    groups = data["template"].values
    print(f"{len(data)} unique rows, {len(set(groups))} templates, {y.mean():.0%} sensitive")

    # ---- train+val / test split by template ----
    gss = GroupShuffleSplit(n_splits=1, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
    trval_idx, test_idx = next(gss.split(X, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=Config.VAL_SIZE / (1 - Config.TEST_SIZE),
                             random_state=Config.RANDOM_STATE)
    tr_rel, val_rel = next(gss2.split(X[trval_idx], y[trval_idx], groups[trval_idx]))
    train_idx, val_idx = trval_idx[tr_rel], trval_idx[val_rel]
    assert not set(groups[train_idx]) & set(groups[test_idx]), "template leak!"

    model = build_pipeline().fit(X[train_idx], y[train_idx])
    threshold = pick_threshold(y[val_idx], model.predict_proba(X[val_idx])[:, 1], Config.TARGET_RECALL)
    print(f"Decision threshold (chosen on validation, target recall {Config.TARGET_RECALL}): {threshold}")

    test_probs = model.predict_proba(X[test_idx])[:, 1]
    test_pred = (test_probs >= threshold).astype(int)
    print("\nHeld-out TEST set (unseen templates):")
    print(classification_report(y[test_idx], test_pred, target_names=["Non-Sensitive", "Sensitive"], digits=3))
    cm = confusion_matrix(y[test_idx], test_pred)
    print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)

    # ---- grouped 5-fold cross-validation (threshold 0.5, model only) ----
    cv = {k: [] for k in ["precision", "recall", "f1", "roc_auc"]}
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        p = build_pipeline().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        for k, v in scores(y[te], p, 0.5).items():
            cv[k].append(v)
    cv_summary = {k: f"{np.mean(v):.3f} +/- {np.std(v):.3f}" for k, v in cv.items()}
    print("\nGroupKFold (5 folds) at threshold 0.5:", cv_summary)

    hn_probs = model.predict_proba(HARD_NEGATIVES)[:, 1]
    hn_fp = int((hn_probs >= threshold).sum())
    print(f"\nHard negatives flagged by the ML model: {hn_fp}/{len(HARD_NEGATIVES)}")
    for s, p in zip(HARD_NEGATIVES, hn_probs):
        if p >= threshold:
            print(f"   {p:.2f}  {s}")

    # ---- final model: refit on train+val, keep validation-chosen threshold ----
    final = build_pipeline().fit(X[trval_idx], y[trval_idx])
    os.makedirs(os.path.dirname(Config.TEXT_MODEL_PATH), exist_ok=True)
    joblib.dump({"model": final, "threshold": threshold}, Config.TEXT_MODEL_PATH)

    metrics = {
        "rows": int(len(data)), "templates": int(len(set(groups))),
        "split_rows": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
        "threshold": threshold,
        "test": scores(y[test_idx], test_probs, threshold),
        "test_confusion_matrix": cm.tolist(),
        "group_kfold_cv": cv_summary,
        "hard_negative_false_positives": f"{hn_fp}/{len(HARD_NEGATIVES)}",
    }
    with open(Config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved model -> {Config.TEXT_MODEL_PATH}\nSaved metrics -> {Config.METRICS_PATH}")


if __name__ == "__main__":
    main()
