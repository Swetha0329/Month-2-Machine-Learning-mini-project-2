from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline


# --------------------------------------------------
# PATH CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data" / "phishshift_prepared.csv"
GROUP_PATH = ROOT / "data" / "template_groups.csv"

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():
    print("=" * 65)
    print("PHISHSHIFT MODEL TRAINING AND EVALUATION")
    print("=" * 65)

    # Check that required files exist.
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found: {DATA_PATH}\n"
            "Run the dataset preprocessing script first."
        )

    if not GROUP_PATH.exists():
        raise FileNotFoundError(
            f"Template groups file not found: {GROUP_PATH}\n"
            "Run the template analysis script first."
        )

    # Load the prepared dataset and template groups.
    data = pd.read_csv(DATA_PATH)
    groups = pd.read_csv(GROUP_PATH)

    required_data_columns = {"message_id", "message_text", "label"}
    required_group_columns = {"message_id", "derived_template_group"}

    if not required_data_columns.issubset(data.columns):
        raise ValueError(
            f"Dataset is missing columns: "
            f"{required_data_columns - set(data.columns)}"
        )

    if not required_group_columns.issubset(groups.columns):
        raise ValueError(
            f"Template groups file is missing columns: "
            f"{required_group_columns - set(groups.columns)}"
        )

    # Keep only the grouping information needed for splitting.
    groups = groups[
        ["message_id", "derived_template_group"]
    ].drop_duplicates(subset=["message_id"])

    groups = groups.rename(
        columns={"derived_template_group": "similarity_group"}
    )

    # Avoid accidentally keeping an older grouping column.
    data = data.drop(
        columns=["derived_template_group", "similarity_group"],
        errors="ignore",
    )

    # Join messages with their derived similarity groups.
    data = data.merge(
        groups,
        on="message_id",
        how="inner",
        validate="one_to_one",
    )

    # Clean and validate the data.
    data["message_text"] = data["message_text"].fillna("").astype(str)
    data["label"] = (
        data["label"].fillna("").astype(str).str.strip().str.lower()
    )

    data = data[
        data["message_text"].str.strip().ne("")
        & data["similarity_group"].notna()
        & data["label"].isin(["legitimate", "phishing"])
    ].copy()

    if data.empty:
        raise ValueError(
            "No usable messages remain after cleaning and grouping."
        )

    print(f"\nMessages loaded: {len(data)}")
    print("\nLabel counts:")
    print(data["label"].value_counts().to_string())
    print(f"\nSimilarity groups: {data['similarity_group'].nunique()}")

    if data["label"].nunique() != 2:
        raise ValueError(
            "Both legitimate and phishing examples are required."
        )

    # --------------------------------------------------
    # GROUP-BASED TRAIN/TEST SPLIT
    # --------------------------------------------------
    # Similarity groups are kept entirely within either train or test.
    # These are derived template groups, not verified campaign IDs.

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            data["message_text"],
            data["label"],
            groups=data["similarity_group"],
        )
    )

    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    shared_groups = (
        set(train["similarity_group"])
        & set(test["similarity_group"])
    )

    print(f"\nTraining messages: {len(train)}")
    print(f"Testing messages: {len(test)}")
    print(f"Training groups: {train['similarity_group'].nunique()}")
    print(f"Testing groups: {test['similarity_group'].nunique()}")
    print(f"Groups shared between train and test: {len(shared_groups)}")

    if shared_groups:
        raise ValueError(
            "Similarity-group leakage detected between train and test."
        )

    # --------------------------------------------------
    # DEFINE THREE MODELS
    # --------------------------------------------------

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = {}

    # --------------------------------------------------
    # TRAIN AND EVALUATE EACH MODEL
    # --------------------------------------------------

    for name, classifier in models.items():
        print("\n" + "-" * 65)
        print(f"Training model: {name}")
        print("-" * 65)

        pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100000,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", classifier),
        ])

        # Fit only on training data.
        pipeline.fit(train["message_text"], train["label"])

        # Evaluate on the same held-out test set.
        predictions = pipeline.predict(test["message_text"])

        metrics = {
            "accuracy": float(
                accuracy_score(test["label"], predictions)
            ),
            "precision_phishing": float(
                precision_score(
                    test["label"],
                    predictions,
                    pos_label="phishing",
                    zero_division=0,
                )
            ),
            "recall_phishing": float(
                recall_score(
                    test["label"],
                    predictions,
                    pos_label="phishing",
                    zero_division=0,
                )
            ),
            "f1_phishing": float(
                f1_score(
                    test["label"],
                    predictions,
                    pos_label="phishing",
                    zero_division=0,
                )
            ),
            "confusion_matrix_labels": [
                "legitimate",
                "phishing",
            ],
            "confusion_matrix": confusion_matrix(
                test["label"],
                predictions,
                labels=["legitimate", "phishing"],
            ).tolist(),
            "training_messages": int(len(train)),
            "testing_messages": int(len(test)),
            "training_groups": int(
                train["similarity_group"].nunique()
            ),
            "testing_groups": int(
                test["similarity_group"].nunique()
            ),
            "evaluation_type": "derived_similarity_group_split",
        }

        results[name] = metrics

        # Save the trained model.
        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(pipeline, model_path)

        print(f"\nAccuracy:           {metrics['accuracy']:.4f}")
        print(
            f"Phishing precision: {metrics['precision_phishing']:.4f}"
        )
        print(
            f"Phishing recall:    {metrics['recall_phishing']:.4f}"
        )
        print(
            f"Phishing F1-score:  {metrics['f1_phishing']:.4f}"
        )

        print("\nClassification report:")
        print(
            classification_report(
                test["label"],
                predictions,
                labels=["legitimate", "phishing"],
                zero_division=0,
            )
        )

        print(f"Saved model: {model_path}")

    # --------------------------------------------------
    # SAVE COMPARISON REPORT
    # --------------------------------------------------

    results_path = REPORT_DIR / "model_comparison.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\n" + "=" * 65)
    print("ALL THREE MODELS TRAINED AND EVALUATED")
    print("=" * 65)
    print(f"Comparison report: {results_path}")
    print(
        "\nImportant: evaluation uses derived similarity groups. "
        "It does not prove generalization to verified unseen campaigns."
    )


if __name__ == "__main__":
    main()