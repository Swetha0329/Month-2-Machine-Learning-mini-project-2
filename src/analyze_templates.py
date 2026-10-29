"""Find connected groups of similar messages for leakage analysis."""

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

INPUT_PATH = PROJECT_ROOT / "data" / "phishshift_prepared.csv"
PAIRS_PATH = PROJECT_ROOT / "data" / "template_similarity_candidates.csv"
GROUPS_PATH = PROJECT_ROOT / "data" / "template_groups.csv"

SIMILARITY_THRESHOLD = 0.85
NEIGHBORS_TO_CHECK = 10


class UnionFind:
    """Combine connected messages into groups."""

    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item):
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first, second):
        root_first = self.find(first)
        root_second = self.find(second)

        if root_first == root_second:
            return

        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first

        self.parent[root_second] = root_first

        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def main():
    print("=" * 60)
    print("PHISHSHIFT CONNECTED TEMPLATE GROUP ANALYSIS")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    # Convert missing values safely before combining text.
    titles = df["subject_or_title"].astype("string").fillna("")
    bodies = df["body_text"].astype("string").fillna("")
    texts = (titles + " " + bodies).str.strip().tolist()

    count = len(texts)
    print(f"\nMessages: {count}")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(texts)
    print(f"TF-IDF feature matrix: {matrix.shape}")

    # Query nearby messages in small batches to limit memory use.
    k = min(NEIGHBORS_TO_CHECK + 1, count)

    neighbors = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=k,
        n_jobs=-1,
    )
    neighbors.fit(matrix)

    union_find = UnionFind(count)
    pairs = []
    candidate_count = 0

    batch_size = 250

    print("\nSearching for similar messages in batches...")

    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)

        distances, indices = neighbors.kneighbors(
            matrix[start:end]
        )

        for local_row, (row_distances, row_indices) in enumerate(
            zip(distances, indices)
        ):
            row_number = start + local_row

            for distance, neighbor_index in zip(
                row_distances, row_indices
            ):
                neighbor_index = int(neighbor_index)

                if neighbor_index == row_number:
                    continue

                similarity = 1.0 - float(distance)

                if similarity < SIMILARITY_THRESHOLD:
                    continue

                # Each pair is recorded once.
                if neighbor_index <= row_number:
                    continue

                candidate_count += 1
                union_find.union(row_number, neighbor_index)

                pairs.append(
                    {
                        "message_id_1": str(
                            df.iloc[row_number]["message_id"]
                        ),
                        "message_id_2": str(
                            df.iloc[neighbor_index]["message_id"]
                        ),
                        "label_1": str(df.iloc[row_number]["label"]),
                        "label_2": str(
                            df.iloc[neighbor_index]["label"]
                        ),
                        "cosine_similarity": round(similarity, 4),
                        "possible_label_conflict": (
                            df.iloc[row_number]["label"]
                            != df.iloc[neighbor_index]["label"]
                        ),
                    }
                )

        print(f"Processed {end} of {count} messages")

    # Assign stable group names using the earliest row in each group.
    root_to_group = {}
    group_ids = []

    for row_number in range(count):
        root = union_find.find(row_number)

        if root not in root_to_group:
            root_to_group[root] = f"similarity_group_{row_number + 1:05d}"

        group_ids.append(root_to_group[root])

    # Save a separate grouping report; the source dataset remains unchanged.
    groups = pd.DataFrame(
        {
            "message_id": df["message_id"].astype(str),
            "label": df["label"].astype(str),
            "derived_template_group": group_ids,
        }
    )

    groups.to_csv(GROUPS_PATH, index=False, encoding="utf-8")

    pairs_df = pd.DataFrame(
        pairs,
        columns=[
            "message_id_1",
            "message_id_2",
            "label_1",
            "label_2",
            "cosine_similarity",
            "possible_label_conflict",
        ],
    )
    pairs_df.to_csv(PAIRS_PATH, index=False, encoding="utf-8")

    group_sizes = groups["derived_template_group"].value_counts()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Candidate similarity pairs: {candidate_count}")
    print(f"Connected groups: {len(group_sizes)}")
    print(f"Messages in groups larger than one: "
          f"{int((groups['derived_template_group'].map(group_sizes) > 1).sum())}")
    print(f"Largest group size: {int(group_sizes.max())}")
    print(f"Groups with more than 20 messages: "
          f"{int((group_sizes > 20).sum())}")

    if not pairs_df.empty:
        conflicts = int(pairs_df["possible_label_conflict"].sum())
        print(f"Candidate pairs with different labels: {conflicts}")

    print("\nLargest groups:")
    print(group_sizes.head(10).to_string())

    print(f"\nPair report: {PAIRS_PATH}")
    print(f"Group report: {GROUPS_PATH}")

    print(
        "\nLIMITATION: Groups are derived from nearest-neighbor text "
        "similarity, not verified campaign identities. Only the nearest "
        f"{NEIGHBORS_TO_CHECK} neighbors per message were checked, so "
        "this is a candidate grouping, not a complete similarity graph."
    )


if __name__ == "__main__":
    main()
