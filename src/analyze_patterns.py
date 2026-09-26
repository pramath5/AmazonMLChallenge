import pandas as pd
from collections import defaultdict
import random

from preprocessing import (
    normalize_name,
    normalize_address,
    extract_numbers,
)


# ============================================================
# FILE PATHS
# ============================================================

SOURCE1 = "../dataset/train/train_source1.tsv"
SOURCE2 = "../dataset/train/train_source2.tsv"
SOURCE3 = "../dataset/train/train_source3.tsv"
GROUND_TRUTH = "../dataset/train/train_ground_truth.tsv"


# ============================================================
# CONFIGURATION
# ============================================================

# Number of S1 records used for evaluation.
# 20,000 gives us a useful estimate without processing
# all 2.2 million records.
SAMPLE_SIZE = 20_000

RANDOM_SEED = 42


# ============================================================
# LOAD SOURCE DATA
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


# ============================================================
# BUILD BLOCKING INDEX
# ============================================================

def build_index(df):

    """
    Build country-specific inverted indexes.

    Example:

        name_index["us"]["payne"]
            -> {S2-123, S3-456}

        address_index["us"]["peoria"]
            -> {S2-123, S3-900}

        number_index["us"]["3315"]
            -> {S2-123}

    Country is used as a filter.

    Country itself NEVER creates candidates.
    """

    name_index = defaultdict(
        lambda: defaultdict(set)
    )

    address_index = defaultdict(
        lambda: defaultdict(set)
    )

    number_index = defaultdict(
        lambda: defaultdict(set)
    )

    total = len(df)

    print("\nBuilding indexes...")

    for count, row in enumerate(
        df.itertuples(index=False),
        start=1
    ):

        entity_id = row.entity_id

        if pd.isna(row.country):
            continue

        country = str(
            row.country
        ).strip().lower()

        if not country:
            continue

        # ----------------------------------------------------
        # NAME
        # ----------------------------------------------------

        name = normalize_name(
            row.business_name
        )

        if name:

            for token in set(name.split()):

                if len(token) >= 2:

                    name_index[country][token].add(
                        entity_id
                    )

        # ----------------------------------------------------
        # ADDRESS
        # ----------------------------------------------------

        address = normalize_address(
            row.business_address
        )

        if address:

            for token in set(address.split()):

                if len(token) >= 2:

                    address_index[country][token].add(
                        entity_id
                    )

        # ----------------------------------------------------
        # NUMBERS
        # ----------------------------------------------------

        numbers = extract_numbers(
            row.business_address
        )

        for number in numbers:

            number_index[country][number].add(
                entity_id
            )

        # Progress
        if count % 500_000 == 0:

            print(
                f"Indexed: {count:,} / {total:,}"
            )

    return {
        "name": name_index,
        "address": address_index,
        "number": number_index,
    }


# ============================================================
# GENERATE CANDIDATES
# ============================================================

def generate_candidates(
    row,
    indexes
):

    """
    Generate candidates for one S1 record.

    Candidate requirements:

        SAME COUNTRY
        AND
        shared name/address/number evidence
    """

    if pd.isna(row.country):
        return set()

    country = str(
        row.country
    ).strip().lower()

    candidates = set()

    # --------------------------------------------------------
    # NAME BLOCK
    # --------------------------------------------------------

    name = normalize_name(
        row.business_name
    )

    if name:

        for token in set(name.split()):

            if len(token) < 2:
                continue

            candidates.update(
                indexes["name"][country].get(
                    token,
                    set()
                )
            )

    # --------------------------------------------------------
    # ADDRESS BLOCK
    # --------------------------------------------------------

    address = normalize_address(
        row.business_address
    )

    if address:

        for token in set(address.split()):

            if len(token) < 2:
                continue

            candidates.update(
                indexes["address"][country].get(
                    token,
                    set()
                )
            )

    # --------------------------------------------------------
    # NUMBER BLOCK
    # --------------------------------------------------------

    numbers = extract_numbers(
        row.business_address
    )

    for number in numbers:

        candidates.update(
            indexes["number"][country].get(
                number,
                set()
            )
        )

    return candidates


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():

    return pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        usecols=[
            "source1_entity_id",
            "matched_entity_ids",
        ],
    )


# ============================================================
# CREATE SAMPLE
# ============================================================

def create_sample(
    s1,
    ground_truth
):

    """
    Select a random sample of S1 records.

    We use a fixed random seed so that every run
    evaluates the same records.
    """

    random.seed(
        RANDOM_SEED
    )

    available_ids = set(
        ground_truth[
            "source1_entity_id"
        ]
    )

    sample_size = min(
        SAMPLE_SIZE,
        len(available_ids)
    )

    sampled_ids = random.sample(
        list(available_ids),
        sample_size
    )

    sampled_ids = set(
        sampled_ids
    )

    sample_s1 = s1[
        s1["entity_id"].isin(
            sampled_ids
        )
    ].copy()

    sample_ground_truth = ground_truth[
        ground_truth[
            "source1_entity_id"
        ].isin(
            sampled_ids
        )
    ].copy()

    return (
        sample_s1,
        sample_ground_truth
    )


# ============================================================
# CONVERT GROUND TRUTH TO DICTIONARY
# ============================================================

def build_truth_dictionary(
    ground_truth
):

    truth = {}

    for row in ground_truth.itertuples(
        index=False
    ):

        s1_id = row.source1_entity_id
        matched = row.matched_entity_ids

        if pd.isna(matched):

            truth[s1_id] = set()

            continue

        matched = str(
            matched
        ).strip()

        if not matched:

            truth[s1_id] = set()

            continue

        truth[s1_id] = {
            x.strip()
            for x in matched.split(",")
            if x.strip()
        }

    return truth


# ============================================================
# EVALUATE BLOCKING
# ============================================================

def evaluate_blocking(
    sample_s1,
    truth,
    indexes
):

    print("\n")
    print("=" * 70)
    print("BLOCKING EVALUATION")
    print("=" * 70)

    total_true_matches = 0
    found_true_matches = 0
    missed_true_matches = 0

    candidate_counts = []

    missed_examples = []

    total_records = len(
        sample_s1
    )

    for count, row in enumerate(
        sample_s1.itertuples(index=False),
        start=1
    ):

        candidates = generate_candidates(
            row,
            indexes
        )

        candidate_count = len(
            candidates
        )

        candidate_counts.append(
            candidate_count
        )

        true_matches = truth.get(
            row.entity_id,
            set()
        )

        total_true_matches += len(
            true_matches
        )

        found = (
            true_matches &
            candidates
        )

        missed = (
            true_matches -
            candidates
        )

        found_true_matches += len(
            found
        )

        missed_true_matches += len(
            missed
        )

        # Store a few examples for debugging.
        if missed and len(
            missed_examples
        ) < 15:

            missed_examples.append(
                {
                    "s1_id": row.entity_id,
                    "name": row.business_name,
                    "address": row.business_address,
                    "missed": list(missed)[:5],
                }
            )

        if count % 2_000 == 0:

            print(
                f"Evaluated: "
                f"{count:,} / "
                f"{total_records:,}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    if total_true_matches > 0:

        recall = (
            found_true_matches /
            total_true_matches
        )

    else:

        recall = 0.0

    candidate_series = pd.Series(
        candidate_counts
    )

    print("\n")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Sample S1 records       : "
        f"{total_records:,}"
    )

    print(
        f"True matches            : "
        f"{total_true_matches:,}"
    )

    print(
        f"Found by blocking       : "
        f"{found_true_matches:,}"
    )

    print(
        f"Missed by blocking      : "
        f"{missed_true_matches:,}"
    )

    print(
        f"Blocking recall         : "
        f"{recall * 100:.4f}%"
    )

    print("\n")
    print("=" * 70)
    print("CANDIDATE STATISTICS")
    print("=" * 70)

    print(
        f"Average candidates/S1   : "
        f"{candidate_series.mean():.2f}"
    )

    print(
        f"Median candidates/S1    : "
        f"{candidate_series.median():.2f}"
    )

    print(
        f"90th percentile         : "
        f"{candidate_series.quantile(0.90):.2f}"
    )

    print(
        f"95th percentile         : "
        f"{candidate_series.quantile(0.95):.2f}"
    )

    print(
        f"99th percentile         : "
        f"{candidate_series.quantile(0.99):.2f}"
    )

    print(
        f"Maximum candidates      : "
        f"{candidate_series.max():,}"
    )

    # ========================================================
    # MISSED MATCH EXAMPLES
    # ========================================================

    if missed_examples:

        print("\n")
        print("=" * 70)
        print("MISSED TRUE MATCH EXAMPLES")
        print("=" * 70)

        for example in missed_examples:

            print(
                f"\nS1 ID: "
                f"{example['s1_id']}"
            )

            print(
                f"Name: "
                f"{example['name']}"
            )

            print(
                f"Address: "
                f"{example['address']}"
            )

            print(
                f"Missed target IDs: "
                f"{example['missed']}"
            )

    else:

        print("\n")
        print(
            "No missed true matches "
            "in the sample."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OPTIMIZED BLOCKING TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load S1
    # --------------------------------------------------------

    print("\nLoading Source 1...")

    s1 = load_source(
        SOURCE1
    )

    print(
        f"Source 1 records: "
        f"{len(s1):,}"
    )

    # --------------------------------------------------------
    # Load ground truth FIRST
    # --------------------------------------------------------

    print("\nLoading ground truth...")

    ground_truth = load_ground_truth()

    print(
        f"Ground truth records: "
        f"{len(ground_truth):,}"
    )

    # --------------------------------------------------------
    # Select sample
    # --------------------------------------------------------

    print(
        f"\nSelecting "
        f"{SAMPLE_SIZE:,} S1 records..."
    )

    (
        sample_s1,
        sample_ground_truth
    ) = create_sample(
        s1,
        ground_truth
    )

    print(
        f"Sample selected: "
        f"{len(sample_s1):,}"
    )

    # We no longer need the full S1.
    del s1

    # --------------------------------------------------------
    # Build truth dictionary
    # --------------------------------------------------------

    truth = build_truth_dictionary(
        sample_ground_truth
    )

    del sample_ground_truth
    del ground_truth

    print(
        "Sample ground truth prepared."
    )

    # --------------------------------------------------------
    # Load S2
    # --------------------------------------------------------

    print("\nLoading Source 2...")

    s2 = load_source(
        SOURCE2
    )

    print(
        f"Source 2 records: "
        f"{len(s2):,}"
    )

    # --------------------------------------------------------
    # Load S3
    # --------------------------------------------------------

    print("\nLoading Source 3...")

    s3 = load_source(
        SOURCE3
    )

    print(
        f"Source 3 records: "
        f"{len(s3):,}"
    )

    # --------------------------------------------------------
    # Combine targets
    # --------------------------------------------------------

    print("\nCombining S2 + S3...")

    targets = pd.concat(
        [s2, s3],
        ignore_index=True
    )

    print(
        f"Target records: "
        f"{len(targets):,}"
    )

    del s2
    del s3

    # --------------------------------------------------------
    # Build indexes
    # --------------------------------------------------------

    indexes = build_index(
        targets
    )

    del targets

    print("\nIndexes built successfully.")

    print(
        f"Name countries: "
        f"{len(indexes['name']):,}"
    )

    print(
        f"Address countries: "
        f"{len(indexes['address']):,}"
    )

    print(
        f"Number countries: "
        f"{len(indexes['number']):,}"
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate_blocking(
        sample_s1,
        truth,
        indexes
    )


if __name__ == "__main__":
    main()