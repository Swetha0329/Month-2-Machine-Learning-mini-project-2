import json
import re
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

OUTPUT_FILE = DATA_DIR / "phishshift_dataset.csv"
PROGRESS_FILE = DATA_DIR / "dataset_progress.json"

DATASET_NAME = "zefang-liu/phishing-email-dataset"
API_URL = "https://datasets-server.huggingface.co/rows"

BATCH_SIZE = 100
MINIMUM_ROWS = 400
MAX_RETRIES = 6

REQUIRED_COLUMNS = [
    "message_id",
    "channel",
    "subject_or_title",
    "body_text",
    "sanitized_url_features",
    "sender_domain_type",
    "label",
    "campaign_group",
    "collected_date",
]

DATA_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# URL FEATURE EXTRACTION
# Never visits or opens URLs.
# --------------------------------------------------

def get_url_features(message):
    urls = re.findall(
        r"https?://\S+|www\.\S+",
        message,
        flags=re.IGNORECASE,
    )

    if not urls:
        return "no_url_detected"

    features = []

    for url in urls:
        clean_url = url.rstrip(".,!?;:)")

        if clean_url.lower().startswith("https://"):
            scheme = "https"
        elif clean_url.lower().startswith("http://"):
            scheme = "http"
        else:
            scheme = "www_or_other"

        features.append(
            f"scheme={scheme};"
            f"contains_dot={'.' in clean_url};"
            f"length={len(clean_url)}"
        )

    return " | ".join(features)


# --------------------------------------------------
# DOWNLOAD ONE BATCH WITH RETRIES
# --------------------------------------------------

def fetch_batch(session, offset):
    params = {
        "dataset": DATASET_NAME,
        "config": "default",
        "split": "train",
        "offset": offset,
        "length": BATCH_SIZE,
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(
                API_URL,
                params=params,
                timeout=60,
            )

            if response.status_code == 429:
                wait_seconds = min(30 * attempt, 180)

                print(
                    f"Server rate limit reached. "
                    f"Waiting {wait_seconds} seconds "
                    f"(attempt {attempt}/{MAX_RETRIES})."
                )

                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(
                    f"Unable to retrieve batch at offset {offset}: {error}"
                ) from error

            wait_seconds = min(10 * attempt, 60)

            print(
                f"Request failed: {error}\n"
                f"Retrying in {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Could not retrieve records beginning at offset {offset}."
    )


# --------------------------------------------------
# CONVERT SOURCE RECORDS
# --------------------------------------------------

def convert_records(api_rows):
    records = []

    for item in api_rows:
        source_row = item.get("row", {})

        message = str(
            source_row.get("Email Text", "")
        ).strip()

        original_label = str(
            source_row.get("Email Type", "")
        ).strip().lower()

        if not message:
            continue

        if "safe" in original_label or "legitimate" in original_label:
            label = "legitimate"
        elif "phishing" in original_label:
            label = "phishing"
        else:
            continue

        source_id = source_row.get("Unnamed: 0")

        if source_id is None:
            source_id = item.get("row_idx", "unknown")

        records.append({
            "message_id": f"source_email_{source_id}",
            "channel": "email",
            "subject_or_title": "",
            "body_text": message,
            "sanitized_url_features": get_url_features(message),
            "sender_domain_type": "unknown",
            "label": label,
            "campaign_group": "unknown_source_campaign",
            "collected_date": date.today().isoformat(),
        })

    return records


# --------------------------------------------------
# SAVE PROGRESS
# --------------------------------------------------

def save_progress(records, next_offset, total_rows):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "next_offset": next_offset,
                "total_rows": total_rows,
                "records": records,
            },
            file,
            ensure_ascii=False,
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("=" * 60)
    print("PHISHSHIFT ONLINE DATASET PREPARATION")
    print("=" * 60)

    records = []
    offset = 0
    total_rows = None

    # Resume from a previous interrupted run if progress exists.
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as file:
                progress = json.load(file)

            records = progress.get("records", [])
            offset = progress.get("next_offset", 0)
            total_rows = progress.get("total_rows")

            print("Resuming previous download.")
            print("Previously retrieved records:", len(records))
            print("Next source offset:", offset)

        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            print("Previous progress file is invalid. Starting again.")
            records = []
            offset = 0
            total_rows = None

    session = requests.Session()
    session.headers.update({
        "User-Agent": "PhishShift-Student-Project/1.0"
    })

    try:
        while True:
            result = fetch_batch(session, offset)

            if total_rows is None:
                total_rows = result.get("num_rows_total")

                print("Dataset:", DATASET_NAME)
                print("Reported source rows:", total_rows)

            api_rows = result.get("rows", [])

            if not api_rows:
                break

            records.extend(convert_records(api_rows))

            offset += len(api_rows)

            # Save after every successful batch.
            save_progress(records, offset, total_rows)

            print(
                f"Retrieved {offset} source rows; "
                f"usable records collected: {len(records)}"
            )

            if total_rows is not None and offset >= total_rows:
                break

            # Pause between batches to reduce rate limiting.
            time.sleep(2)

    except (RuntimeError, requests.RequestException) as error:
        print("\nDataset retrieval was interrupted.")
        print("Reason:", error)
        print("Progress has been saved when possible.")
        print("Run this script again later to resume.")
        return

    # --------------------------------------------------
    # PREPARE FINAL CSV
    # --------------------------------------------------

    if len(records) < MINIMUM_ROWS:
        print(
            f"\nSTOP: Only {len(records)} usable records were retrieved."
        )
        print(f"At least {MINIMUM_ROWS} are required.")
        print("The final dataset was not created.")
        return

    df = pd.DataFrame(records, columns=REQUIRED_COLUMNS)

    # Remove duplicate message text.
    df = df.drop_duplicates(
        subset=["body_text"]
    ).reset_index(drop=True)

    if len(df) < MINIMUM_ROWS:
        print(
            f"\nSTOP: Only {len(df)} unique messages remain "
            "after duplicate removal."
        )
        print(f"At least {MINIMUM_ROWS} are required.")
        print("The final dataset was not created.")
        return

    df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("DATASET CREATED SUCCESSFULLY")
    print("=" * 60)

    print("Saved to:", OUTPUT_FILE)
    print("Unique messages:", len(df))

    print("\nLabel counts:")
    print(df["label"].value_counts().to_string())

    print("\nColumns:")
    for column in df.columns:
        print("-", column)

    print("\nIMPORTANT DATA LIMITATIONS:")
    print("1. The original source supplies email text and labels.")
    print("2. Subject/title is left blank because it is not supplied.")
    print("3. Sender domain type is marked unknown.")
    print("4. Genuine campaign groups are not supplied.")
    print("5. URL features are derived without visiting URLs.")
    print("6. Campaign-separated evaluation needs suitable")
    print("   documented campaign data or clearly labelled simulations.")

    print("\nProgress is saved in:", PROGRESS_FILE)
    print("Do not treat unknown campaign groups as genuine campaigns.")


if __name__ == "__main__":
    main()
