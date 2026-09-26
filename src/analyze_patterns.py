import pandas as pd
import re
import unicodedata
from pathlib import Path


# ============================================================
# ROBUST FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Your current project has:
# AmazonMLChallenge/
#   datasets/student_resource/dataset/
# If your dataset is directly under AmazonMLChallenge/dataset/,
# the fallback below handles that too.
DATASET_ROOT = PROJECT_ROOT / "datasets" / "student_resource" / "dataset"

if not DATASET_ROOT.exists():
    DATASET_ROOT = PROJECT_ROOT / "dataset"

TRAIN_DIR = DATASET_ROOT / "train"

GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"
SOURCE1 = TRAIN_DIR / "train_source1.tsv"
SOURCE2 = TRAIN_DIR / "train_source2.tsv"
SOURCE3 = TRAIN_DIR / "train_source3.tsv"


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def token_set(text):
    return set(normalize_text(text).split())


def jaccard_from_normalized(a, b):
    """Jaccard using already-normalized strings."""
    if not a or not b:
        return 0.0

    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


# ============================================================
# LOAD DATA
# ============================================================

def load_source(path):
    return pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )


def build_record_lookup(df):
    """
    Convert the dataframe into dictionaries once.

    This avoids repeatedly doing:
        source.loc[entity_id]
    for every ground-truth match.
    """
    lookup = {}

    for row in df.itertuples(index=False):
        lookup[row.entity_id] = {
            "business_name": row.business_name,
            "business_address": row.business_address,
            "country": row.country,
        }

    return lookup


# ============================================================
# NAME PATTERNS
# ============================================================

def analyze_name_patterns(s1, s2, s3, ground_truth):
    print("\n" + "=" * 70)
    print("NAME MATCHING PATTERNS")
    print("=" * 70)

    # Precompute normalized names ONCE.
    print("Preparing name lookup tables...")

    s1_names = {
        entity_id: normalize_text(record["business_name"])
        for entity_id, record in s1.items()
    }

    s2_names = {
        entity_id: normalize_text(record["business_name"])
        for entity_id, record in s2.items()
    }

    s3_names = {
        entity_id: normalize_text(record["business_name"])
        for entity_id, record in s3.items()
    }

    total_matches = 0
    exact = 0
    high_similarity = 0
    low_similarity = 0
    empty_name = 0

    examples = []
    total_rows = len(ground_truth)

    print(f"Analyzing {total_rows:,} ground-truth rows...")

    for count, row in enumerate(
        ground_truth.itertuples(index=False),
        start=1
    ):
        s1_id = row.source1_entity_id
        matched = row.matched_entity_ids

        if pd.isna(matched) or not str(matched).strip():
            continue

        normalized_s1 = s1_names.get(s1_id, "")

        for match_id in str(matched).split(","):
            match_id = match_id.strip()

            if not match_id:
                continue

            if match_id.startswith("S2-"):
                normalized_target = s2_names.get(match_id, "")
            elif match_id.startswith("S3-"):
                normalized_target = s3_names.get(match_id, "")
            else:
                continue

            total_matches += 1

            if not normalized_target:
                empty_name += 1
                continue

            if normalized_s1 == normalized_target:
                exact += 1

            sim = jaccard_from_normalized(
                normalized_s1,
                normalized_target
            )

            if sim >= 0.5:
                high_similarity += 1

            if sim < 0.3:
                low_similarity += 1

                if len(examples) < 15:
                    # Retrieve original values only for the few examples.
                    if s1_id in s1:
                        original_s1 = s1[s1_id]["business_name"]
                    else:
                        original_s1 = ""

                    if match_id.startswith("S2-"):
                        original_target = s2.get(
                            match_id, {}
                        ).get("business_name", "")
                    else:
                        original_target = s3.get(
                            match_id, {}
                        ).get("business_name", "")

                    examples.append(
                        (
                            original_s1,
                            original_target,
                            sim,
                        )
                    )

        # Progress indicator so the terminal never appears frozen.
        if count % 100_000 == 0:
            print(
                f"Processed ground-truth rows: "
                f"{count:,} / {total_rows:,}"
            )

    print(
        f"\nTotal ground-truth matches analyzed : "
        f"{total_matches:,}"
    )

    print(
        f"Exact normalized name matches       : "
        f"{exact:,}"
    )

    print(
        f"Name Jaccard >= 0.50                : "
        f"{high_similarity:,}"
    )

    print(
        f"Name Jaccard < 0.30                 : "
        f"{low_similarity:,}"
    )

    print(
        f"Missing target names                : "
        f"{empty_name:,}"
    )

    if total_matches:
        print("\nPercentages:")

        print(
            f"Exact name match       : "
            f"{exact / total_matches * 100:.2f}%"
        )

        print(
            f"Jaccard >= 0.50        : "
            f"{high_similarity / total_matches * 100:.2f}%"
        )

        print(
            f"Jaccard < 0.30         : "
            f"{low_similarity / total_matches * 100:.2f}%"
        )

    print("\nExamples where name similarity is LOW:")
    print("-" * 70)

    for name1, name2, sim in examples:
        print(f"S1 : {name1}")
        print(f"S2 : {name2}")
        print(f"Jaccard: {sim:.3f}")
        print()


# ============================================================
# ADDRESS PATTERNS
# ============================================================

def analyze_address_patterns(s1, s2, s3, ground_truth):
    print("\n" + "=" * 70)
    print("ADDRESS MATCHING PATTERNS")
    print("=" * 70)

    print("Preparing address lookup tables...")

    s1_addresses = {
        entity_id: normalize_text(record["business_address"])
        for entity_id, record in s1.items()
    }

    s2_addresses = {
        entity_id: normalize_text(record["business_address"])
        for entity_id, record in s2.items()
    }

    s3_addresses = {
        entity_id: normalize_text(record["business_address"])
        for entity_id, record in s3.items()
    }

    total = 0
    both_present = 0
    high_similarity = 0
    low_similarity = 0

    examples = []
    total_rows = len(ground_truth)

    print(f"Analyzing {total_rows:,} ground-truth rows...")

    for count, row in enumerate(
        ground_truth.itertuples(index=False),
        start=1
    ):
        s1_id = row.source1_entity_id
        matched = row.matched_entity_ids

        if pd.isna(matched) or not str(matched).strip():
            continue

        address1 = s1_addresses.get(s1_id, "")
        total_for_row = 0

        for match_id in str(matched).split(","):
            match_id = match_id.strip()

            if not match_id:
                continue

            if match_id.startswith("S2-"):
                address2 = s2_addresses.get(match_id, "")
            elif match_id.startswith("S3-"):
                address2 = s3_addresses.get(match_id, "")
            else:
                continue

            total += 1

            if not address1 or not address2:
                continue

            both_present += 1

            sim = jaccard_from_normalized(
                address1,
                address2
            )

            if sim >= 0.5:
                high_similarity += 1

            if sim < 0.3:
                low_similarity += 1

                if len(examples) < 15:
                    original_s1 = s1.get(
                        s1_id, {}
                    ).get("business_address", "")

                    if match_id.startswith("S2-"):
                        original_target = s2.get(
                            match_id, {}
                        ).get("business_address", "")
                    else:
                        original_target = s3.get(
                            match_id, {}
                        ).get("business_address", "")

                    examples.append(
                        (
                            original_s1,
                            original_target,
                            sim,
                        )
                    )

        if count % 100_000 == 0:
            print(
                f"Processed ground-truth rows: "
                f"{count:,} / {total_rows:,}"
            )

    print(
        f"\nTotal matches                  : "
        f"{total:,}"
    )

    print(
        f"Both addresses present        : "
        f"{both_present:,}"
    )

    print(
        f"Address Jaccard >= 0.50       : "
        f"{high_similarity:,}"
    )

    print(
        f"Address Jaccard < 0.30        : "
        f"{low_similarity:,}"
    )

    if both_present:
        print(
            "\nPercentages among pairs "
            "with both addresses:"
        )

        print(
            f"Jaccard >= 0.50 : "
            f"{high_similarity / both_present * 100:.2f}%"
        )

        print(
            f"Jaccard < 0.30  : "
            f"{low_similarity / both_present * 100:.2f}%"
        )

    print("\nExamples where address similarity is LOW:")
    print("-" * 70)

    for address1, address2, sim in examples:
        print(f"S1 : {address1}")
        print(f"S2 : {address2}")
        print(f"Jaccard: {sim:.3f}")
        print()


# ============================================================
# COUNTRY
# ============================================================

def analyze_country(s1, s2, s3, ground_truth):
    print("\n" + "=" * 70)
    print("COUNTRY CONSISTENCY")
    print("=" * 70)

    s1_countries = {
        entity_id: record["country"]
        for entity_id, record in s1.items()
    }

    s2_countries = {
        entity_id: record["country"]
        for entity_id, record in s2.items()
    }

    s3_countries = {
        entity_id: record["country"]
        for entity_id, record in s3.items()
    }

    total = 0
    same_country = 0
    different_country = 0

    for row in ground_truth.itertuples(index=False):
        s1_id = row.source1_entity_id
        matched = row.matched_entity_ids

        if pd.isna(matched) or not str(matched).strip():
            continue

        country1 = s1_countries.get(s1_id)

        for match_id in str(matched).split(","):
            match_id = match_id.strip()

            if match_id.startswith("S2-"):
                country2 = s2_countries.get(match_id)
            elif match_id.startswith("S3-"):
                country2 = s3_countries.get(match_id)
            else:
                continue

            if pd.isna(country1) or pd.isna(country2):
                continue

            total += 1

            if str(country1).upper() == str(country2).upper():
                same_country += 1
            else:
                different_country += 1

    print(
        f"Total matched pairs       : "
        f"{total:,}"
    )

    print(
        f"Same country              : "
        f"{same_country:,}"
    )

    print(
        f"Different country         : "
        f"{different_country:,}"
    )

    if total:
        print(
            f"\nCountry agreement: "
            f"{same_country / total * 100:.2f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("OPTIMIZED MATCHING PATTERN ANALYSIS")
    print("=" * 70)

    print(f"\nDataset folder: {DATASET_ROOT}")

    required_files = [
        SOURCE1,
        SOURCE2,
        SOURCE3,
        GROUND_TRUTH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}\n"
                f"\nExpected dataset structure:\n"
                f"{PROJECT_ROOT}\\datasets\\student_resource\\dataset\\train\\"
            )

    print("\nLoading training data...")

    s1_df = load_source(SOURCE1)
    print(f"Source 1 loaded: {len(s1_df):,}")

    s2_df = load_source(SOURCE2)
    print(f"Source 2 loaded: {len(s2_df):,}")

    s3_df = load_source(SOURCE3)
    print(f"Source 3 loaded: {len(s3_df):,}")

    ground_truth = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )

    print(
        f"Ground truth loaded: "
        f"{len(ground_truth):,}"
    )

    print("\nBuilding fast record lookups...")

    s1 = build_record_lookup(s1_df)
    s2 = build_record_lookup(s2_df)
    s3 = build_record_lookup(s3_df)

    del s1_df, s2_df, s3_df

    print("Lookups ready.")

    # Run analyses one at a time.
    analyze_name_patterns(
        s1,
        s2,
        s3,
        ground_truth,
    )

    analyze_address_patterns(
        s1,
        s2,
        s3,
        ground_truth,
    )

    analyze_country(
        s1,
        s2,
        s3,
        ground_truth,
    )

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
