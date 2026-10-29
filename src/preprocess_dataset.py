"""Prepare the PhishShift dataset for machine-learning experiments."""

import re
import sys
from pathlib import Path

import pandas as pd

# Make the project root accessible when this script runs directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DATASET_PATH, REQUIRED_DATASET_COLUMNS


OUTPUT_PATH = PROJECT_ROOT / "data" / "phishshift_prepared.csv"


def normalize_message(text):
    """Normalize message text for duplicate and template detection."""
    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # Replace email addresses and long numbers with placeholders.
    text = re.sub(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
        "<EMAIL>",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\b\d{5,}\b", "<NUMBER>", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text


def main():
    print("=" * 60)
    print("PHISHSHIFT DATASET PREPROCESSING")
    print("=" * 60)

    if not Path(DATASET_PATH).exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    missing_columns = [
        column
        for column in REQUIRED_DATASET_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    original_count = len(df)

    # Keep the original dataset unchanged.
    df["subject_or_title"] = (
        df["subject_or_title"].fillna("").astype(str)
    )
    df["body_text"] = df["body_text"].fillna("").astype(str)

    # Combine the available title and body for analysis.
    df["message_text"] = (
        df["subject_or_title"].str.strip()
        + " "
        + df["body_text"].str.strip()
    ).str.strip()

    # Remove records with no usable message text.
    df = df[df["message_text"].str.len() > 0].copy()

    # Normalize text to identify exact duplicates and template patterns.
    df["normalized_text"] = df["message_text"].apply(
        normalize_message
    )

    before_deduplication = len(df)

    # Remove exact duplicates after normalization.
    df = df.drop_duplicates(
        subset=["normalized_text"],
        keep="first",
    ).copy()

    duplicates_removed = before_deduplication - len(df)

    # Assign a derived group to each distinct normalized message.
    # These are NOT genuine source campaign IDs.
    df["derived_template_group"] = (
        "template_" +
        df["normalized_text"].map(
            lambda value: str(abs(hash(value)))
        )
    )

    # Preserve the source campaign field exactly as supplied.
    # Do not replace unknown campaign metadata with invented values.
    df["source_campaign_group"] = df["campaign_group"].fillna(
        "unknown_source_campaign"
    ).astype(str)

    # Save the prepared dataset separately from the original.
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nOriginal rows: {original_count}")
    print(f"Rows with usable text: {before_deduplication}")
    print(f"Normalized duplicates removed: {duplicates_removed}")
    print(f"Prepared rows: {len(df)}")

    print("\nLabel distribution:")
    print(df["label"].value_counts().to_string())

    print("\nSource campaign groups:")
    print(df["source_campaign_group"].value_counts().head(10).to_string())

    print("\nImportant limitation:")
    print(
        "Derived template groups are based on normalized text. "
        "They are not verified real-world campaign groups."
    )

    print(f"\nPrepared dataset saved to:\n{OUTPUT_PATH}")
    print("\nPreprocessing completed.")


if __name__ == "__main__":
    main()
