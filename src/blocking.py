import pandas as pd
import numpy as np
from collections import defaultdict, Counter

from preprocessing import (
    normalize_name,
    normalize_address,
    get_tokens,
    extract_numbers
)


S1_FILE = "../dataset/train/train_source1.tsv"
S2_FILE = "../dataset/train/train_source2.tsv"
S3_FILE = "../dataset/train/train_source3.tsv"


# Maximum number of target records allowed in a block.
# Tokens occurring more often than this are ignored.
MAX_BLOCK_SIZE = 5000


def build_index(df, field, max_block_size=MAX_BLOCK_SIZE):

    index = defaultdict(list)

    frequencies = Counter()

    print(f"Building frequency index for {field}...")

    # First calculate token frequencies
    for value in df[field]:

        if not isinstance(value, str):
            continue

        tokens = get_tokens(value)

        # Count each token only once per record
        for token in set(tokens):
            frequencies[token] += 1

    print(f"Unique tokens: {len(frequencies):,}")

    # Keep only selective tokens
    useful_tokens = {
        token
        for token, count in frequencies.items()
        if count <= max_block_size
    }

    print(
        f"Useful tokens: {len(useful_tokens):,}"
    )

    # Build inverted index
    for idx, value in enumerate(df[field]):

        if not isinstance(value, str):
            continue

        tokens = set(get_tokens(value))

        for token in tokens:

            if token in useful_tokens:
                index[token].append(idx)

    return index, frequencies


def build_number_index(df):

    index = defaultdict(list)

    print("Building number index...")

    for idx, value in enumerate(df["business_address"]):

        if not isinstance(value, str):
            continue

        numbers = set(extract_numbers(value))

        for number in numbers:

            # Ignore extremely common numbers
            index[number].append(idx)

    return index


def get_candidates(
    s1_row,
    s2,
    s3,
    name_index,
    address_index,
    number_index
):

    candidates = set()

    country = s1_row["country"]

    name = normalize_name(s1_row["business_name"])
    address = normalize_address(s1_row["business_address"])

    # -----------------------------
    # NAME BLOCKING
    # -----------------------------

    for token in set(get_tokens(name)):

        if token in name_index:

            for source, idx in name_index[token]:

                if source == "S2":
                    candidates.add(("S2", idx))
                else:
                    candidates.add(("S3", idx))

    # -----------------------------
    # ADDRESS BLOCKING
    # -----------------------------

    for token in set(get_tokens(address)):

        if token in address_index:

            for source, idx in address_index[token]:

                if source == "S2":
                    candidates.add(("S2", idx))
                else:
                    candidates.add(("S3", idx))

    # -----------------------------
    # NUMBER BLOCKING
    # -----------------------------

    for number in set(extract_numbers(address)):

        if number in number_index:

            for source, idx in number_index[number]:

                if source == "S2":
                    candidates.add(("S2", idx))
                else:
                    candidates.add(("S3", idx))

    # -----------------------------
    # COUNTRY FILTER
    # -----------------------------

    filtered = set()

    for source, idx in candidates:

        if source == "S2":
            if s2.iloc[idx]["country"] == country:
                filtered.add((source, idx))

        else:
            if s3.iloc[idx]["country"] == country:
                filtered.add((source, idx))

    return filtered


def main():

    print("Loading data...")

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str
    )

    s2 = pd.read_csv(
        S2_FILE,
        sep="\t",
        dtype=str
    )

    s3 = pd.read_csv(
        S3_FILE,
        sep="\t",
        dtype=str
    )

    print()
    print("S1:", len(s1))
    print("S2:", len(s2))
    print("S3:", len(s3))

    # ----------------------------------------
    # ADD SOURCE LABEL
    # ----------------------------------------

    s2_indexed = []
    s3_indexed = []

    # Name index
    print("\nBuilding name index...")

    name_index = defaultdict(list)

    name_frequency = Counter()

    for source, df in [("S2", s2), ("S3", s3)]:

        for idx, value in enumerate(df["business_name"]):

            if not isinstance(value, str):
                continue

            tokens = set(get_tokens(normalize_name(value)))

            for token in tokens:
                name_frequency[token] += 1

    useful_name_tokens = {
        token
        for token, count in name_frequency.items()
        if count <= MAX_BLOCK_SIZE
    }

    print(
        "Total name tokens:",
        len(name_frequency)
    )

    print(
        "Useful name tokens:",
        len(useful_name_tokens)
    )

    for source, df in [("S2", s2), ("S3", s3)]:

        for idx, value in enumerate(df["business_name"]):

            if not isinstance(value, str):
                continue

            tokens = set(get_tokens(normalize_name(value)))

            for token in tokens:

                if token in useful_name_tokens:
                    name_index[token].append(
                        (source, idx)
                    )

    # ----------------------------------------
    # ADDRESS INDEX
    # ----------------------------------------

    print("\nBuilding address index...")

    address_frequency = Counter()

    for source, df in [("S2", s2), ("S3", s3)]:

        for value in df["business_address"]:

            if not isinstance(value, str):
                continue

            tokens = set(
                get_tokens(
                    normalize_address(value)
                )
            )

            for token in tokens:
                address_frequency[token] += 1

    useful_address_tokens = {
        token
        for token, count in address_frequency.items()
        if count <= MAX_BLOCK_SIZE
    }

    print(
        "Total address tokens:",
        len(address_frequency)
    )

    print(
        "Useful address tokens:",
        len(useful_address_tokens)
    )

    address_index = defaultdict(list)

    for source, df in [("S2", s2), ("S3", s3)]:

        for idx, value in enumerate(df["business_address"]):

            if not isinstance(value, str):
                continue

            tokens = set(
                get_tokens(
                    normalize_address(value)
                )
            )

            for token in tokens:

                if token in useful_address_tokens:
                    address_index[token].append(
                        (source, idx)
                    )

    # ----------------------------------------
    # NUMBER INDEX
    # ----------------------------------------

    print("\nBuilding number index...")

    number_frequency = Counter()

    for source, df in [("S2", s2), ("S3", s3)]:

        for value in df["business_address"]:

            if not isinstance(value, str):
                continue

            for number in set(extract_numbers(value)):
                number_frequency[number] += 1

    useful_numbers = {
        number
        for number, count in number_frequency.items()
        if count <= MAX_BLOCK_SIZE
    }

    number_index = defaultdict(list)

    for source, df in [("S2", s2), ("S3", s3)]:

        for idx, value in enumerate(df["business_address"]):

            if not isinstance(value, str):
                continue

            for number in set(extract_numbers(value)):

                if number in useful_numbers:
                    number_index[number].append(
                        (source, idx)
                    )

    print(
        "Useful numbers:",
        len(useful_numbers)
    )

    # ----------------------------------------
    # TEST BLOCKING
    # ----------------------------------------

    print("\n" + "=" * 60)
    print("TESTING BLOCKING V2")
    print("=" * 60)

    # Use same 20k sample as V1
    sample = s1.sample(
        n=20000,
        random_state=42
    )

    candidate_counts = []

    for count, (_, row) in enumerate(
        sample.iterrows(),
        start=1
    ):

        candidates = get_candidates(
            row,
            s2,
            s3,
            name_index,
            address_index,
            number_index
        )

        candidate_counts.append(len(candidates))

        if count % 1000 == 0:
            print(
                f"Processed {count:,}/20,000"
            )

    candidate_counts = np.array(candidate_counts)

    print("\nRESULTS")
    print("-" * 60)

    print(
        "Average candidates:",
        f"{candidate_counts.mean():,.2f}"
    )

    print(
        "Median candidates:",
        f"{np.median(candidate_counts):,.2f}"
    )

    print(
        "90th percentile:",
        f"{np.percentile(candidate_counts, 90):,.2f}"
    )

    print(
        "95th percentile:",
        f"{np.percentile(candidate_counts, 95):,.2f}"
    )

    print(
        "99th percentile:",
        f"{np.percentile(candidate_counts, 99):,.2f}"
    )

    print(
        "Maximum:",
        f"{candidate_counts.max():,}"
    )


if __name__ == "__main__":
    main()