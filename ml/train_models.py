"""
Train three distinct XGBoost model versions with different objectives:
  v1-champion       — balanced, threshold 0.90
  v2-high-recall    — max recall, threshold 0.65
  v3-high-precision — max precision, threshold 0.95

Each uses the SAME architecture but different class weights, thresholds,
and subsampling — producing genuinely different precision/recall curves.

Run from project root:
    python ml/train_models.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ── Config ───────────────────────────────────────────────────────────────────

DATA_PATH = Path("data/raw/creditcard.csv")
SAVED = Path("saved_models")

VERSIONS = [
    {
        "tag": "v1-champion",
        "dir": SAVED / "v1",
        "description": "Standard threshold (0.90). Balanced precision and recall.",
        "params": {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 577,   # ~normal class ratio
            "random_state": 42,
            "eval_metric": "logloss",
            "verbosity": 0,
        },
        "threshold": 0.90,
    },
    {
        "tag": "v2-high-recall",
        "dir": SAVED / "v2",
        "description": "Low threshold (0.65). Catches more fraud, more reviews.",
        "params": {
            "n_estimators": 120,
            "max_depth": 5,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "scale_pos_weight": 900,   # higher weight → more sensitive
            "random_state": 7,
            "eval_metric": "logloss",
            "verbosity": 0,
        },
        "threshold": 0.65,
    },
    {
        "tag": "v3-high-precision",
        "dir": SAVED / "v3",
        "description": "High threshold (0.95). Minimises false positives.",
        "params": {
            "n_estimators": 80,
            "max_depth": 3,
            "learning_rate": 0.12,
            "subsample": 0.7,
            "colsample_bytree": 0.9,
            "scale_pos_weight": 300,   # lower weight → more conservative
            "random_state": 99,
            "eval_metric": "logloss",
            "verbosity": 0,
        },
        "threshold": 0.95,
    },
]

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_data(path: Path):
    df = pd.read_csv(path)
    X = df[FEATURES].copy()
    y = df["Class"]
    # log1p + scale Amount
    X["Amount"] = np.log1p(X["Amount"])
    return X, y


def make_scaler(X_train: pd.DataFrame) -> StandardScaler:
    sc = StandardScaler()
    sc.fit(X_train[["Amount"]])
    return sc


def apply_scaler(X: pd.DataFrame, sc: StandardScaler) -> pd.DataFrame:
    X = X.copy()
    X[["Amount"]] = sc.transform(X[["Amount"]])
    return X


def evaluate(model, X_test, y_test, threshold) -> dict:
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_test, probs), 4),
        "pr_auc": round(average_precision_score(y_test, probs), 4),
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, zero_division=0), 4),
    }


# ── Train ─────────────────────────────────────────────────────────────────────

def train():
    if not DATA_PATH.exists():
        print(f"[ERROR] Dataset not found at {DATA_PATH}")
        print("  Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return

    print(f"Loading data from {DATA_PATH}...")
    X, y = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15 / 0.85, stratify=y_train, random_state=42
    )

    for ver in VERSIONS:
        print(f"\n{'─'*60}")
        print(f"Training {ver['tag']}...")
        out = Path(ver["dir"])
        out.mkdir(parents=True, exist_ok=True)

        sc = make_scaler(X_train)
        Xtr = apply_scaler(X_train, sc)
        Xva = apply_scaler(X_val, sc)
        Xte = apply_scaler(X_test, sc)

        clf = XGBClassifier(**ver["params"])
        clf.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)

        metrics = evaluate(clf, Xte, y_test, ver["threshold"])
        print(f"  Threshold : {ver['threshold']}")
        for k, v in metrics.items():
            print(f"  {k:12s}: {v}")

        joblib.dump(clf, out / "xgb_model.joblib")
        joblib.dump(sc, out / "amount_scaler.joblib")

        # Also write to root saved_models/ for v1 (legacy default)
        if ver["tag"] == "v1-champion":
            joblib.dump(clf, SAVED / "xgb_model.joblib")
            joblib.dump(sc, SAVED / "amount_scaler.joblib")

        metadata = {
            "model_name": "xgboost",
            "version_tag": ver["tag"],
            "description": ver["description"],
            "model_params": ver["params"],
            "threshold": ver["threshold"],
            "features": FEATURES,
            "preprocessing": {
                "amount_transformation": "log1p + StandardScaler",
                "dropped_columns": ["Time"],
            },
            "evaluation": metrics,
            "data_split": {"train": 0.70, "validation": 0.15, "test": 0.15, "stratified": True},
        }
        with open(out / "model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        if ver["tag"] == "v1-champion":
            with open(SAVED / "model_metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)

        print(f"  Saved to {out}")

    print(f"\n{'─'*60}")
    print("All models trained successfully.")


if __name__ == "__main__":
    train()
