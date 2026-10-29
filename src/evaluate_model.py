from pathlib import Path
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "phishshift_prepared.csv"
OUTPUT_PATH = ROOT / "reports" / "model_metrics.json"

LABEL_MAP = {
    "phishing": "phishing",
    "phish": "phishing",
    "spam": "phishing",
    "malicious": "phishing",
    "1": "phishing",
    "legitimate": "legitimate",
    "legit": "legitimate",
    "ham": "legitimate",
    "benign": "legitimate",
    "safe": "legitimate",
    "0": "legitimate",
}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {"body_text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df[["body_text", "label"]].dropna().copy()
    df["body_text"] = df["body_text"].astype(str).str.strip()
    df["label"] = (
        df["label"].astype(str).str.strip().str.lower().map(LABEL_MAP)
    )
    df = df[(df["body_text"] != "") & df["label"].notna()]

    if df["label"].nunique() != 2:
        raise ValueError(
            "Expected both legitimate and phishing labels. "
            f"Found: {df['label'].value_counts().to_dict()}"
        )

    X = df["body_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Fixed order: legitimate first, phishing second.
    class_names = ["legitimate", "phishing"]
    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=class_names,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=class_names,
        output_dict=True,
        zero_division=0,
    )

    dataset_counts = df["label"].value_counts().to_dict()

    metrics = {
        "dataset": {
            "total": int(len(df)),
            "legitimate": int(dataset_counts.get("legitimate", 0)),
            "phishing": int(dataset_counts.get("phishing", 0)),
        },
        "split": {
            "training_messages": int(len(X_train)),
            "testing_messages": int(len(X_test)),
        },
        "scores": {
            "accuracy": round(float(accuracy_score(y_test, predictions)) * 100, 2),
            "precision": round(
                float(precision_score(
                    y_test, predictions, pos_label="phishing", zero_division=0
                )) * 100,
                2,
            ),
            "recall": round(
                float(recall_score(
                    y_test, predictions, pos_label="phishing", zero_division=0
                )) * 100,
                2,
            ),
            "f1_score": round(
                float(f1_score(
                    y_test, predictions, pos_label="phishing", zero_division=0
                )) * 100,
                2,
            ),
        },
        "confusion_matrix": {
            "labels": class_names,
            "values": matrix.tolist(),
        },
        "class_report": {
            "legitimate": {
                "precision": round(report["legitimate"]["precision"] * 100, 2),
                "recall": round(report["legitimate"]["recall"] * 100, 2),
                "f1_score": round(report["legitimate"]["f1-score"] * 100, 2),
                "support": int(report["legitimate"]["support"]),
            },
            "phishing": {
                "precision": round(report["phishing"]["precision"] * 100, 2),
                "recall": round(report["phishing"]["recall"] * 100, 2),
                "f1_score": round(report["phishing"]["f1-score"] * 100, 2),
                "support": int(report["phishing"]["support"]),
            },
        },
        "note": (
            "Metrics are from a held-out test split using this evaluation "
            "pipeline. They do not automatically describe a different saved model."
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Evaluation completed.")
    print(f"Training messages: {len(X_train)}")
    print(f"Testing messages: {len(X_test)}")
    print(f"Accuracy: {metrics['scores']['accuracy']}%")
    print(f"Precision: {metrics['scores']['precision']}%")
    print(f"Recall: {metrics['scores']['recall']}%")
    print(f"F1-score: {metrics['scores']['f1_score']}%")
    print(f"Metrics saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()