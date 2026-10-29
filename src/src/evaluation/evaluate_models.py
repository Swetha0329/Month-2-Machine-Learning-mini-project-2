from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# Project root: G:\phishshift
ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = ROOT / "data" / "phishshift_prepared.csv"
GROUP_PATH = ROOT / "data" / "template_groups.csv"
OUTPUT_PATH = ROOT / "reports" / "metrics" / "fair_model_comparison.json"


def main():
    print("=" * 60)
    print("PHISHSHIFT MODEL EVALUATION")
    print("=" * 60)

    # Check required files
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    if not GROUP_PATH.exists():
        raise FileNotFoundError(
            f"Template groups file not found: {GROUP_PATH}"
        )

    # Load dataset files
    print("\nLoading dataset...")

    df = pd.read_csv(DATA_PATH)
    groups_df = pd.read_csv(GROUP_PATH)

    print(f"Dataset rows: {len(df)}")
    print(f"Dataset columns: {list(df.columns)}")
    print(f"Group columns: {list(groups_df.columns)}")

    # Identify the text column
    text_candidates = [
        "body_text",
        "Email Text",
        "text",
        "message",
    ]

    text_col = next(
        (column for column in text_candidates if column in df.columns),
        None,
    )

    if text_col is None:
        raise ValueError(
            f"Text column not found. Available columns: {list(df.columns)}"
        )

    # Identify the label column
    label_candidates = [
        "label",
        "Email Type",
        "target",
    ]

    label_col = next(
        (column for column in label_candidates if column in df.columns),
        None,
    )

    if label_col is None:
        raise ValueError(
            f"Label column not found. Available columns: {list(df.columns)}"
        )

    # Check message IDs
    if "message_id" not in df.columns:
        raise ValueError(
            "Dataset is missing the message_id column."
        )

    if "message_id" not in groups_df.columns:
        raise ValueError(
            "Template groups file is missing message_id."
        )

    # Identify the group column
    group_candidates = [
        "template_group",
        "group_id",
        "similarity_group",
        "derived_template_group",
    ]

    group_col = next(
        (column for column in group_candidates if column in groups_df.columns),
        None,
    )

    if group_col is None:
        raise ValueError(
            "Group column not found in template_groups.csv. "
            f"Available columns: {list(groups_df.columns)}"
        )

    # Prepare the text and labels
    df = df.copy()

    df[text_col] = df[text_col].fillna("").astype(str)

    df[label_col] = (
        df[label_col]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    label_map = {
        "safe email": "legitimate",
        "legitimate": "legitimate",
        "ham": "legitimate",
        "phishing email": "phishing",
        "phishing": "phishing",
        "spam": "phishing",
    }

    df["target"] = df[label_col].map(label_map)

    # Remove rows without usable text or recognized labels
    df = df[
        df["target"].notna()
        & (df[text_col].str.strip() != "")
    ].copy()

    # Prepare group mapping
    group_mapping = groups_df[
        ["message_id", group_col]
    ].drop_duplicates(subset=["message_id"])

    # IMPORTANT:
    # Remove the existing group column before merging.
    # This prevents pandas from creating suffixed column names.
    if group_col in df.columns:
        df = df.drop(columns=[group_col])

    # Attach the correct group to each message
    df = df.merge(
        group_mapping,
        on="message_id",
        how="left",
        validate="many_to_one",
    )

    # Give messages without a group a unique fallback group
    missing_groups = df[group_col].isna()

    df.loc[missing_groups, group_col] = (
        "missing_group_"
        + df.loc[missing_groups, "message_id"].astype(str)
    )

    df = df.reset_index(drop=True)

    print(f"\nUsable messages: {len(df)}")

    print("\nClass distribution:")
    print(df["target"].value_counts().to_string())

    # Create a group-based train/test split.
    # Messages assigned to the same derived group stay together.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(
            df,
            df["target"],
            groups=df[group_col],
        )
    )

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    train_groups = set(train_df[group_col])
    test_groups = set(test_df[group_col])

    overlapping_groups = train_groups.intersection(test_groups)

    if overlapping_groups:
        raise RuntimeError(
            f"Group leakage detected: {len(overlapping_groups)} "
            "groups appear in both training and testing data."
        )

    train_text = train_df[text_col]
    test_text = test_df[text_col]

    y_train = train_df["target"]
    y_test = test_df["target"]

    print(f"\nTraining messages: {len(train_df)}")
    print(f"Testing messages: {len(test_df)}")
    print(f"Overlapping groups: {len(overlapping_groups)}")

    # Convert message text into TF-IDF features.
    # Fit the vectorizer only on the training data.
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
    )

    X_train = vectorizer.fit_transform(train_text)
    X_test = vectorizer.transform(test_text)

    print(f"\nTraining features: {X_train.shape[1]}")

    # Models to compare
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
        "Linear SVM": LinearSVC(
            class_weight="balanced",
            random_state=42,
        ),
    }

    # Save evaluation information
    results = {
        "evaluation_type": "derived_similarity_group_holdout",
        "note": (
            "Groups are derived from text similarity and are not verified "
            "real-world campaign IDs. Results must not be described as "
            "verified campaign-separated performance."
        ),
        "dataset": str(DATA_PATH),
        "text_column": text_col,
        "label_column": label_col,
        "group_column": group_col,
        "random_state": 42,
        "test_size": 0.20,
        "training_messages": int(len(train_df)),
        "test_messages": int(len(test_df)),
        "overlapping_groups": int(len(overlapping_groups)),
        "models": {},
    }

    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    # Train and evaluate each model on the same split
    for name, model in models.items():
        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            predictions,
            labels=["legitimate", "phishing"],
        ).ravel()

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        model_results = {
            "accuracy": round(
                float(accuracy_score(y_test, predictions)), 4
            ),
            "phishing_precision": round(
                float(
                    precision_score(
                        y_test,
                        predictions,
                        pos_label="phishing",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "phishing_recall": round(
                float(
                    recall_score(
                        y_test,
                        predictions,
                        pos_label="phishing",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "phishing_f1": round(
                float(
                    f1_score(
                        y_test,
                        predictions,
                        pos_label="phishing",
                        zero_division=0,
                    )
                ),
                4,
            ),
            "false_positive_rate": round(
                float(false_positive_rate), 4
            ),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }

        results["models"][name] = model_results

        print(f"\n{name}")
        print(f"  Accuracy:          {model_results['accuracy']}")
        print(
            "  Phishing precision: "
            f"{model_results['phishing_precision']}"
        )
        print(
            "  Phishing recall:    "
            f"{model_results['phishing_recall']}"
        )
        print(
            "  Phishing F1-score:  "
            f"{model_results['phishing_f1']}"
        )
        print(
            "  False positive rate: "
            f"{model_results['false_positive_rate']}"
        )

    # Save the results report
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("MODEL EVALUATION COMPLETED")
    print("=" * 60)
    print(f"\nReport saved to:\n{OUTPUT_PATH}")


if __name__ == "__main__":
    main()