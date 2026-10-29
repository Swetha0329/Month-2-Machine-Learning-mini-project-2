
import json
import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "phishshift_prepared.csv"
GROUP_PATH = ROOT / "data" / "template_groups.csv"
OUTPUT_PATH = ROOT / "reports" / "metrics" / "rules_baseline.json"


def predict_rule(text):
    text = str(text).lower()

    rules = [
        r"\b(urgent|immediately|act now|final warning|suspend|suspended|locked)\b",
        r"\b(verify|verification|confirm your account|login|sign in|credentials)\b",
        r"\b(password|bank account|credit card|personal information|security alert)\b",
        r"\b(prize|winner|claim now|free gift|claim your reward)\b",
        r"https?://|www\.",
        r"\bhxxps?://|\[\.\]|\(\.\)",
    ]

    score = sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in rules)
    return "phishing" if score >= 2 else "legitimate"


def main():
    data = pd.read_csv(DATA_PATH)
    groups = pd.read_csv(GROUP_PATH)

    groups = groups[
        ["message_id", "derived_template_group"]
    ].drop_duplicates(subset=["message_id"])

    groups = groups.rename(
        columns={"derived_template_group": "similarity_group"}
    )

    data = data.drop(
        columns=["derived_template_group", "similarity_group"],
        errors="ignore",
    )

    data = data.merge(
        groups,
        on="message_id",
        how="inner",
        validate="one_to_one",
    )

    data["message_text"] = data["message_text"].fillna("").astype(str)
    data["label"] = data["label"].fillna("").astype(str).str.strip().str.lower()

    data = data[
        data["message_text"].str.strip().ne("")
        & data["similarity_group"].notna()
        & data["label"].isin(["legitimate", "phishing"])
    ].copy()

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

    train = data.iloc[train_idx]
    test = data.iloc[test_idx].copy()

    overlap = set(train["similarity_group"]) & set(test["similarity_group"])
    if overlap:
        raise ValueError("Similarity-group leakage detected.")

    test["prediction"] = test["message_text"].apply(predict_rule)

    tn, fp, fn, tp = confusion_matrix(
        test["label"],
        test["prediction"],
        labels=["legitimate", "phishing"],
    ).ravel()

    results = {
        "evaluation_type": "derived_similarity_group_holdout",
        "note": "Uses the same grouping source and split settings as ML training. Similarity groups are not verified real-world campaign IDs.",
        "messages_evaluated": int(len(test)),
        "training_messages": int(len(train)),
        "test_messages": int(len(test)),
        "overlapping_groups": int(len(overlap)),
        "accuracy": float(accuracy_score(test["label"], test["prediction"])),
        "precision_phishing": float(precision_score(
            test["label"], test["prediction"],
            pos_label="phishing", zero_division=0
        )),
        "recall_phishing": float(recall_score(
            test["label"], test["prediction"],
            pos_label="phishing", zero_division=0
        )),
        "f1_phishing": float(f1_score(
            test["label"], test["prediction"],
            pos_label="phishing", zero_division=0
        )),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("RULE-BASED BASELINE — GROUP HOLDOUT")
    print(json.dumps(results, indent=4))
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()