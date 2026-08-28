from __future__ import annotations

import json

import joblib
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from razorguard.config import ARTIFACTS, TRANSACTIONS_PATH
from razorguard.ml.features import build_model_frame
from razorguard.ml.train import CATEGORICAL, NUMERIC, chronological_split


def main() -> None:
    raw = pd.read_parquet(TRANSACTIONS_PATH)
    df = build_model_frame(raw)
    _, _, test = chronological_split(df)

    metadata = json.loads((ARTIFACTS / "model_metadata.json").read_text())
    threshold = float(metadata["threshold"])
    model = joblib.load(ARTIFACTS / "risk_model.joblib")

    X_test = test[NUMERIC + CATEGORICAL]
    y_test = test["is_chargeback"]

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    result = {
        "test_rows": int(len(test)),
        "positive_rate": float(y_test.mean()),
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fpr),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    (ARTIFACTS / "test_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print(json.dumps(result, indent=2))
    print("\\nClassification report:")
    print(classification_report(y_test, pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
