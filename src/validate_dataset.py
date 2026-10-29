"""Validate the PhishShift dataset before model training."""

import sys
from pathlib import Path

import pandas as pd

# Make the project root importable when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402


def validate_dataset():
    """Check dataset existence, structure, size, labels, and essential values."""
    print("=" * 60)
    print("PHISHSHIFT DATASET VALIDATION")
    print("=" * 60)

    dataset_path = config.DATASET_PATH

    # 1. Check whether the dataset exists.
    print("\n1. DATASET FILE")
    if not dataset_path.exists():
        print(f"FAIL: Dataset not found at: {dataset_path}")
        print("Create or place the dataset in the data folder before continuing.")
        return False

    print(f"PASS: Dataset found at: {dataset_path}")

    # 2. Load the dataset.
    try:
        df = pd.read_csv(dataset_path)
    except Exception as exc:
        print(f"FAIL: Could not read the dataset: {exc}")
        return False

    # 3. Check required columns.
    print("\n2. REQUIRED COLUMNS")
    missing_columns = [
        column
        for column in config.REQUIRED_DATASET_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print(f"FAIL: Missing columns: {missing_columns}")
    else:
        print("PASS: All required columns are present.")

    # 4. Check dataset size.
    print("\n3. DATASET SIZE")
    print(f"Total messages: {len(df)}")

    if len(df) < config.MINIMUM_DATASET_ROWS:
        print(
            "FAIL: Dataset must contain at least "
            f"{config.MINIMUM_DATASET_ROWS} messages."
        )
    else:
        print("PASS: Minimum dataset size requirement met.")

    # 5. Check classification labels.
    print("\n4. CLASSIFICATION LABELS")
    if "label" not in df.columns:
        print("FAIL: Cannot validate labels because the label column is missing.")
    else:
        labels = set(
            df["label"].dropna().astype(str).str.strip().str.lower().unique()
        )
        invalid_labels = labels - config.ALLOWED_LABELS
        missing_label_count = int(df["label"].isna().sum())

        if invalid_labels:
            print(f"FAIL: Unexpected labels: {sorted(invalid_labels)}")
        elif missing_label_count:
            print(f"FAIL: {missing_label_count} messages have missing labels.")
        elif labels != config.ALLOWED_LABELS:
            print(
                "FAIL: Dataset must contain both "
                f"{sorted(config.ALLOWED_LABELS)} labels."
            )
        else:
            print("PASS: Both allowed labels are present.")
            print("Label counts:")
            print(
                df["label"]
                .astype(str)
                .str.strip()
                .str.lower()
                .value_counts()
                .to_string()
            )

    # 6. Check essential message text.
    print("\n5. MESSAGE CONTENT")
    essential_columns = ["subject_or_title", "body_text"]
    missing_content_columns = [
        column for column in essential_columns if column not in df.columns
    ]

    if missing_content_columns:
        print(f"FAIL: Missing content columns: {missing_content_columns}")
    else:
        empty_content = (
            df[essential_columns]
            .fillna("")
            .astype(str)
            .apply(lambda column: column.str.strip().eq(""))
            .all(axis=1)
        )
        empty_count = int(empty_content.sum())

        if empty_count:
            print(
                f"WARNING: {empty_count} messages have both title and body empty."
            )
        else:
            print("PASS: Every message has a title or body.")

    # 7. Final result.
    print("\n" + "=" * 60)

    checks_passed = (
        not missing_columns
        and len(df) >= config.MINIMUM_DATASET_ROWS
        and "label" in df.columns
        and not (
            set(
                df["label"]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
                .unique()
            )
            - config.ALLOWED_LABELS
        )
        and df["label"].notna().all()
        and set(
            df["label"].dropna().astype(str).str.strip().str.lower().unique()
        )
        == config.ALLOWED_LABELS
        and not missing_content_columns
        and not empty_content.any()
    )

    if checks_passed:
        print("FINAL RESULT: PASS")
        print("Dataset is ready for the next validation and training steps.")
    else:
        print("FINAL RESULT: FAIL")
        print("Fix the reported issues before training the model.")

    print("=" * 60)
    return checks_passed


if __name__ == "__main__":
    success = validate_dataset()
    sys.exit(0 if success else 1)
