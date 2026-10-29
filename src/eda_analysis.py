
from pathlib import Path
import json
import re
from collections import Counter

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "phishshift_prepared.csv"
CHART_DIR = ROOT / "reports" / "eda"
SUMMARY_PATH = CHART_DIR / "eda_summary.json"

TEXT_COLUMN = "body_text"
LABEL_COLUMN = "label"


def main():
    print("=" * 60)
    print("PHISHSHIFT EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # Check dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    required_columns = [TEXT_COLUMN, LABEL_COLUMN]
    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns are missing: {missing}"
        )

    df = df.copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)
    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN].astype(str).str.lower().str.strip()
    )

    # Normalize label names for analysis
    label_map = {
        "safe email": "legitimate",
        "legitimate": "legitimate",
        "ham": "legitimate",
        "phishing email": "phishing",
        "phishing": "phishing",
        "spam": "phishing",
    }

    df["analysis_label"] = df[LABEL_COLUMN].map(label_map)

    # Exclude rows without usable text or recognized labels
    df = df[
        (df["analysis_label"].notna())
        & (df[TEXT_COLUMN].str.strip() != "")
    ].copy()

    if df.empty:
        raise ValueError(
            "No usable messages were found in the dataset."
        )

    # Create output directory
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate message lengths
    df["message_length_words"] = df[TEXT_COLUMN].apply(
        lambda text: len(text.split())
    )

    df["message_length_characters"] = df[TEXT_COLUMN].str.len()

    # --------------------------------------------------
    # CHART 1: CLASS DISTRIBUTION
    # --------------------------------------------------
    label_counts = (
        df["analysis_label"]
        .value_counts()
        .reindex(["legitimate", "phishing"], fill_value=0)
    )

    plt.figure(figsize=(8, 5))
    label_counts.plot(kind="bar")
    plt.title("PhishShift Dataset: Class Distribution")
    plt.xlabel("Message class")
    plt.ylabel("Number of messages")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "01_class_distribution.png",
        dpi=150,
    )
    plt.close()

    # --------------------------------------------------
    # CHART 2: MESSAGE LENGTH DISTRIBUTION
    # --------------------------------------------------
    plt.figure(figsize=(9, 5))

    for label in ["legitimate", "phishing"]:
        lengths = df.loc[
            df["analysis_label"] == label,
            "message_length_words",
        ]

        if not lengths.empty:
            plt.hist(
                lengths.clip(upper=500),
                bins=40,
                alpha=0.55,
                label=label,
            )

    plt.title("Message Length Distribution")
    plt.xlabel("Words per message (values capped at 500)")
    plt.ylabel("Number of messages")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        CHART_DIR / "02_message_length_distribution.png",
        dpi=150,
    )
    plt.close()

    # --------------------------------------------------
    # CHART 3: TOP WORDS BY MESSAGE CLASS
    # --------------------------------------------------
    stop_words = {
        "the", "and", "for", "you", "your", "are", "this",
        "that", "with", "from", "have", "has", "was", "were",
        "will", "would", "could", "should", "can", "our",
        "their", "they", "them", "his", "her", "she", "him",
        "its", "not", "but", "all", "any", "one", "out",
        "about", "into", "more", "some", "when", "what",
        "where", "who", "how", "why", "been", "being",
        "than", "then", "there", "here", "also", "just",
        "get", "got", "use", "used", "using", "please",
        "http", "https", "www", "com", "email",
    }

    def extract_words(text):
        words = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return [
            word for word in words
            if word not in stop_words
        ]

    word_counts = {}

    for label in ["legitimate", "phishing"]:
        texts = df.loc[
            df["analysis_label"] == label,
            TEXT_COLUMN,
        ]

        counts = Counter()

        for text in texts:
            counts.update(extract_words(text))

        word_counts[label] = counts.most_common(15)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    for axis, label in zip(
        axes,
        ["legitimate", "phishing"],
    ):
        common_words = word_counts[label]

        if common_words:
            words = [item[0] for item in common_words]
            counts = [item[1] for item in common_words]

            axis.barh(words[::-1], counts[::-1])
            axis.set_title(f"Top Words: {label.title()}")
            axis.set_xlabel("Word frequency")
            axis.set_ylabel("Word")

        else:
            axis.set_title(f"No words available: {label.title()}")

    fig.suptitle("Most Frequent Words by Message Class")
    fig.tight_layout()
    fig.savefig(
        CHART_DIR / "03_top_words_by_class.png",
        dpi=150,
    )
    plt.close(fig)

    # --------------------------------------------------
    # SUMMARY STATISTICS
    # --------------------------------------------------
    summary = {
        "dataset_path": str(DATA_PATH),
        "analysis_type": "exploratory_data_analysis",
        "messages_analyzed": int(len(df)),
        "class_distribution": {
            str(label): int(count)
            for label, count in label_counts.items()
        },
        "class_percentages": {
            str(label): round(
                float(count / len(df) * 100), 2
            )
            for label, count in label_counts.items()
        },
        "message_length_words": {
            "overall_mean": round(
                float(df["message_length_words"].mean()), 2
            ),
            "overall_median": round(
                float(df["message_length_words"].median()), 2
            ),
            "legitimate_mean": round(
                float(
                    df.loc[
                        df["analysis_label"] == "legitimate",
                        "message_length_words",
                    ].mean()
                ), 2
            ),
            "phishing_mean": round(
                float(
                    df.loc[
                        df["analysis_label"] == "phishing",
                        "message_length_words",
                    ].mean()
                ), 2
            ),
        },
        "top_words": {
            label: [
                {"word": word, "count": count}
                for word, count in words
            ]
            for label, words in word_counts.items()
        },
        "charts": [
            "01_class_distribution.png",
            "02_message_length_distribution.png",
            "03_top_words_by_class.png",
        ],
        "limitations": [
            "Word frequency does not establish that a word is a phishing indicator.",
            "Exploratory patterns do not prove causation.",
            "The analysis reflects this dataset and may not represent all real-world messages.",
        ],
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------
    print(f"\nMessages analyzed: {len(df)}")

    print("\nClass distribution:")
    print(label_counts.to_string())

    print("\nAverage message length:")
    print(
        summary["message_length_words"]["overall_mean"],
        "words per message",
    )

    print("\nEDA charts saved to:")
    print(CHART_DIR)

    print("\nSummary report saved to:")
    print(SUMMARY_PATH)

    print("\n" + "=" * 60)
    print("EDA ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()