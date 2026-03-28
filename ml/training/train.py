"""
Model training script for fraud detection.

Workflow:
1. Load raw data
2. Prepare dataframe
3. Split into train / validation / test
4. Fit feature transformations on train only
5. Transform validation and test using same scaler
6. Train XGBoost with imbalance handling
7. Tune threshold on validation set
8. Evaluate final model on test set
9. Save model, scaler, and metadata

This script should be run offline during model development/training.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from ml.preprocessing.prepare_data import load_data, prepare_training_dataframe
from ml.preprocessing.transform_features import fit_transform_train, transform_new_data
from ml.training.evaluate import compare_thresholds, evaluate_at_threshold, find_best_threshold
from ml.utils import save_artifact, save_json, set_seed, split_features_target


DATA_PATH = "data/raw/creditcard.csv"
MODEL_DIR = "saved_models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "amount_scaler.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.json")


def get_scale_pos_weight(y: pd.Series) -> float:
    """
    Compute XGBoost's scale_pos_weight for imbalanced binary classification.

    Formula:
        number_of_negative_samples / number_of_positive_samples

    Args:
        y: Target labels.

    Returns:
        scale_pos_weight value.
    """
    negatives = (y == 0).sum()
    positives = (y == 1).sum()

    if positives == 0:
        raise ValueError("No positive class samples found in target.")

    return negatives / positives


def train_model() -> None:
    """
    Train the fraud detection model and save all artifacts.
    """
    set_seed(42)

    # 1. Load and prepare data
    df = load_data(DATA_PATH)
    df = prepare_training_dataframe(df)

    # 2. Split features and target
    X, y = split_features_target(df, target_col="Class")

    # 3. Train/validation/test split with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=42,
    )

    # 4. Transform features
    X_train_transformed, scaler = fit_transform_train(X_train)
    X_val_transformed = transform_new_data(X_val, scaler)
    X_test_transformed = transform_new_data(X_test, scaler)

    # 5. Train XGBoost
    model = XGBClassifier(
        n_estimators=60,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.21*get_scale_pos_weight(y_train),
        random_state=42,
        eval_metric="logloss",
    )

    model.fit(X_train_transformed, y_train)

    # 6. Predict probabilities
    val_probs = model.predict_proba(X_val_transformed)[:, 1]
    test_probs = model.predict_proba(X_test_transformed)[:, 1]

    # 7. Tune threshold on validation set
    best_threshold, best_metrics = find_best_threshold(
        y_val,
        val_probs,
        min_precision=0.80,
        recall_tolerance=0.02,
    )

    # 8. Final evaluation on test set
    final_results = evaluate_at_threshold(y_test, test_probs, best_threshold)

    threshold_df = compare_thresholds(
        y_val,
        val_probs,
        thresholds=[0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95],
    )

    # 9. Save artifacts
    save_artifact(model, MODEL_PATH)
    save_artifact(scaler, SCALER_PATH)

    metadata = {
        "model_name": "xgboost",
        "model_params": {
            "n_estimators": 60,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 1.21 * get_scale_pos_weight(y_train),
            "random_state": 42,
            "eval_metric": "logloss",
        },
        "threshold": float(best_threshold),
        "features": list(X_train_transformed.columns),
        "preprocessing": {
            "amount_transformation": "log1p + StandardScaler",
            "dropped_columns": ["Time"],
        },
        "threshold_selection": {
            "strategy": "validation-based tuning to balance precision and recall",
            "objective": "maximize recall while maintaining acceptable precision",
            "minimum_precision_constraint": 0.80,
            "recall_tolerance": 0.02,
            "selected_metrics_on_validation": best_metrics,
        },
        "evaluation": {
            "roc_auc": float(final_results["roc_auc"]),
            "pr_auc": float(final_results["pr_auc"]),
            "precision": float(final_results["precision"]),
            "recall": float(final_results["recall"]),
            "f1": float(final_results["f1"]),
        },
        "data_split": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
            "stratified": True,
        },
    }
    save_json(metadata, METADATA_PATH)

    # 10. Print summary
    print("\nTraining complete.")
    print(f"Best threshold (selected on validation): {best_threshold}")
    print("Best threshold metrics on validation:", best_metrics)

    print("\nFinal test evaluation:")
    print("ROC-AUC:", final_results["roc_auc"])
    print("PR-AUC:", final_results["pr_auc"])
    print("Confusion Matrix:\n", final_results["confusion_matrix"])
    print("\nClassification Report:\n", final_results["classification_report"])

    print("\nValidation threshold comparison:\n", threshold_df)


if __name__ == "__main__":
    train_model()