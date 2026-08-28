from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from razorguard.config import ARTIFACTS, RANDOM_SEED, TRANSACTIONS_PATH
from razorguard.ml.features import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    build_model_frame,
)

NUMERIC = NUMERIC_FEATURES
CATEGORICAL = CATEGORICAL_FEATURES


def chronological_split(df: pd.DataFrame):
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def make_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                NUMERIC,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocess", pre),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    raw = pd.read_parquet(TRANSACTIONS_PATH)
    df = build_model_frame(raw)

    train, val, test = chronological_split(df)

    X_train, y_train = train[NUMERIC + CATEGORICAL], train["is_chargeback"]
    X_val, y_val = val[NUMERIC + CATEGORICAL], val["is_chargeback"]

    model = make_pipeline()
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]

    # Threshold selection is explicitly done on validation data only.
    thresholds = [i / 100 for i in range(10, 96)]
    rows = []
    for threshold in thresholds:
        pred = (val_prob >= threshold).astype(int)
        tp = ((pred == 1) & (y_val.to_numpy() == 1)).sum()
        fp = ((pred == 1) & (y_val.to_numpy() == 0)).sum()
        fn = ((pred == 0) & (y_val.to_numpy() == 1)).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": int(fp),
                "false_negatives": int(fn),
            }
        )

    threshold_df = pd.DataFrame(rows)
    best = threshold_df.sort_values(
        ["f1", "precision"], ascending=[False, False]
    ).iloc[0]

    joblib.dump(model, ARTIFACTS / "risk_model.joblib")
    threshold_df.to_csv(ARTIFACTS / "validation_thresholds.csv", index=False)

    metadata = {
        "model": "logistic_regression",
        "threshold": float(best["threshold"]),
        "selection_metric": "validation_f1",
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
    }
    (ARTIFACTS / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
