import os
import csv
import random
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

CANDIDATE_FILE = os.path.join(
    BASE_DIR,
    "candidate_pairs.tsv"
)

GROUND_TRUTH_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "train",
    "train_ground_truth.tsv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

POSITIVE_FILE = os.path.join(
    OUTPUT_DIR,
    "positive_pairs.tsv"
)

NEGATIVE_FILE = os.path.join(
    OUTPUT_DIR,
    "negative_pairs.tsv"
)

TRAINING_FILE = os.path.join(
    OUTPUT_DIR,
    "training_pairs.tsv"
)


# ============================================================
# SETTINGS
# ============================================================

MAX_NEGATIVES = 3_000_000

RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth():

    print("=" * 60)
    print("LOADING GROUND TRUTH")
    print("=" * 60)

    df = pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    ground_truth = {}

    for row in df.itertuples(index=False):

        s1_id = row.source1_entity_id
        matched_ids = row.matched_entity_ids

        if matched_ids:

            ground_truth[s1_id] = set(
                x.strip()
                for x in matched_ids.split(",")
                if x.strip()
            )

        else:

            ground_truth[s1_id] = set()

    total_matches = sum(
        len(matches)
        for matches in ground_truth.values()
    )

    print(
        f"S1 entities: {len(ground_truth):,}"
    )

    print(
        f"True match links: {total_matches:,}"
    )

    return ground_truth


# ============================================================
# PROCESS CANDIDATES
# ============================================================

def process_candidates(
    ground_truth,
    positive_pairs,
    negative_pairs
):

    print("\n" + "=" * 60)
    print("SCANNING CANDIDATE FILE")
    print("=" * 60)

    print(
        "Candidate file:",
        CANDIDATE_FILE
    )

    print(
        f"Maximum negatives: "
        f"{MAX_NEGATIVES:,}"
    )

    scanned = 0
    positives = 0
    negatives_seen = 0

    with open(
        CANDIDATE_FILE,
        "r",
        encoding="utf-8",
        errors="replace"
    ) as infile:

        reader = csv.reader(
            infile,
            delimiter="\t"
        )

        # Skip header
        next(reader)

        for row in reader:

            scanned += 1

            s1_id = row[0]
            candidate_id = row[1]

            true_matches = ground_truth.get(
                s1_id,
                set()
            )

            # =================================================
            # TRUE MATCH
            # =================================================

            if candidate_id in true_matches:

                positive_pairs.append(
                    (
                        s1_id,
                        candidate_id,
                        1
                    )
                )

                positives += 1

            # =================================================
            # NEGATIVE
            # =================================================

            else:

                negatives_seen += 1

                # ------------------------------------------------
                # Instead of random sampling every row,
                # keep the first 3 million negatives.
                #
                # This is MUCH faster.
                # ------------------------------------------------

                if len(negative_pairs) < MAX_NEGATIVES:

                    negative_pairs.append(
                        (
                            s1_id,
                            candidate_id,
                            0
                        )
                    )

            # =================================================
            # PROGRESS
            # =================================================

            if scanned % 10_000_000 == 0:

                print(
                    f"Scanned: {scanned:,} | "
                    f"Positives: {positives:,} | "
                    f"Negatives seen: {negatives_seen:,} | "
                    f"Negatives kept: "
                    f"{len(negative_pairs):,}"
                )

    print("\n" + "=" * 60)
    print("SCAN COMPLETE")
    print("=" * 60)

    print(
        f"Total candidates scanned: "
        f"{scanned:,}"
    )

    print(
        f"Positive pairs: "
        f"{positives:,}"
    )

    print(
        f"Negative candidates seen: "
        f"{negatives_seen:,}"
    )

    print(
        f"Negative pairs kept: "
        f"{len(negative_pairs):,}"
    )


# ============================================================
# WRITE FILE
# ============================================================

def write_file(
    filename,
    pairs
):

    with open(
        filename,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(
            f,
            delimiter="\t"
        )

        writer.writerow([
            "source1_entity_id",
            "candidate_entity_id",
            "label"
        ])

        writer.writerows(pairs)


# ============================================================
# COMBINE
# ============================================================

def create_training_file(
    positive_pairs,
    negative_pairs
):

    print("\n" + "=" * 60)
    print("WRITING TRAINING FILES")
    print("=" * 60)

    # Shuffle only the small training data,
    # not the 14 GB candidate file.

    random.shuffle(
        positive_pairs
    )

    random.shuffle(
        negative_pairs
    )

    # Positive file
    write_file(
        POSITIVE_FILE,
        positive_pairs
    )

    print(
        "Created:",
        POSITIVE_FILE
    )

    # Negative file
    write_file(
        NEGATIVE_FILE,
        negative_pairs
    )

    print(
        "Created:",
        NEGATIVE_FILE
    )

    # Combined file
    with open(
        TRAINING_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as outfile:

        writer = csv.writer(
            outfile,
            delimiter="\t"
        )

        writer.writerow([
            "source1_entity_id",
            "candidate_entity_id",
            "label"
        ])

        for pair in positive_pairs:
            writer.writerow(pair)

        for pair in negative_pairs:
            writer.writerow(pair)

    print(
        "Created:",
        TRAINING_FILE
    )

    print("\nTraining statistics:")

    print(
        f"Positive pairs: "
        f"{len(positive_pairs):,}"
    )

    print(
        f"Negative pairs: "
        f"{len(negative_pairs):,}"
    )

    print(
        f"Total training pairs: "
        f"{len(positive_pairs) + len(negative_pairs):,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("AMAZON ML CHALLENGE")
    print("TRAINING PAIR GENERATION")
    print("=" * 60)

    ground_truth = load_ground_truth()

    positive_pairs = []
    negative_pairs = []

    process_candidates(
        ground_truth,
        positive_pairs,
        negative_pairs
    )

    create_training_file(
        positive_pairs,
        negative_pairs
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()