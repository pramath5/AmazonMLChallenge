import os
import re
import pandas as pd
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "test"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs.tsv"
)


SOURCE1_FILE = os.path.join(
    TEST_DIR,
    "test_source1.tsv"
)

SOURCE2_FILE = os.path.join(
    TEST_DIR,
    "test_source2.tsv"
)

SOURCE3_FILE = os.path.join(
    TEST_DIR,
    "test_source3.tsv"
)


# ============================================================
# BLOCKING PARAMETERS
# ============================================================

MAX_TOKEN_FREQUENCY = 3000


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def tokenize(text):

    text = normalize_text(text)

    if not text:
        return []

    return text.split()


def extract_numbers(text):

    if pd.isna(text):
        return []

    return re.findall(
        r"\d+",
        str(text)
    )


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_data():

    print("=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)

    s1 = pd.read_csv(
        SOURCE1_FILE,
        sep="\t"
    )

    s2 = pd.read_csv(
        SOURCE2_FILE,
        sep="\t"
    )

    s3 = pd.read_csv(
        SOURCE3_FILE,
        sep="\t"
    )

    print(
        f"Source 1 records: {len(s1):,}"
    )

    print(
        f"Source 2 records: {len(s2):,}"
    )

    print(
        f"Source 3 records: {len(s3):,}"
    )

    return s1, s2, s3


# ============================================================
# BUILD INDEXES
# ============================================================

def build_indexes(s2, s3):

    print("\n" + "=" * 60)
    print("BUILDING BLOCKING INDEXES")
    print("=" * 60)

    all_sources = pd.concat(
        [
            s2,
            s3
        ],
        ignore_index=True
    )

    name_index = defaultdict(list)

    address_index = defaultdict(list)

    number_index = defaultdict(list)

    country_index = defaultdict(list)

    # --------------------------------------------------------
    # First pass: calculate token frequencies
    # --------------------------------------------------------

    name_frequency = defaultdict(int)

    address_frequency = defaultdict(int)

    for _, row in all_sources.iterrows():

        name_tokens = set(
            tokenize(row["business_name"])
        )

        address_tokens = set(
            tokenize(row["business_address"])
        )

        for token in name_tokens:
            name_frequency[token] += 1

        for token in address_tokens:
            address_frequency[token] += 1


    print(
        f"Unique name tokens: "
        f"{len(name_frequency):,}"
    )

    print(
        f"Unique address tokens: "
        f"{len(address_frequency):,}"
    )


    # --------------------------------------------------------
    # Second pass: build indexes
    # --------------------------------------------------------

    for _, row in all_sources.iterrows():

        entity_id = row["entity_id"]

        country = normalize_text(
            row["country"]
        )

        name_tokens = set(
            tokenize(row["business_name"])
        )

        address_tokens = set(
            tokenize(row["business_address"])
        )

        numbers = set(
            extract_numbers(
                row["business_address"]
            )
        )

        # Country index
        country_index[country].append(
            entity_id
        )

        # Name index
        for token in name_tokens:

            if name_frequency[token] <= MAX_TOKEN_FREQUENCY:

                name_index[token].append(
                    entity_id
                )

        # Address index
        for token in address_tokens:

            if address_frequency[token] <= MAX_TOKEN_FREQUENCY:

                address_index[token].append(
                    entity_id
                )

        # Number index
        for number in numbers:

            number_index[number].append(
                entity_id
            )


    print(
        f"Name index entries: "
        f"{len(name_index):,}"
    )

    print(
        f"Address index entries: "
        f"{len(address_index):,}"
    )

    print(
        f"Number index entries: "
        f"{len(number_index):,}"
    )

    return (
        name_index,
        address_index,
        number_index,
        country_index,
        all_sources
    )


# ============================================================
# CREATE CANDIDATES FOR ONE S1
# ============================================================

def generate_candidates(
    row,
    name_index,
    address_index,
    number_index,
    country_index
):

    candidates = set()

    country = normalize_text(
        row["country"]
    )

    name_tokens = set(
        tokenize(row["business_name"])
    )

    address_tokens = set(
        tokenize(row["business_address"])
    )

    numbers = set(
        extract_numbers(
            row["business_address"]
        )
    )


    # --------------------------------------------------------
    # Country candidates
    # --------------------------------------------------------

    country_candidates = set(
        country_index.get(
            country,
            []
        )
    )


    # --------------------------------------------------------
    # Route 1:
    # Two name tokens
    # --------------------------------------------------------

    rare_name_tokens = [

        token

        for token in name_tokens

        if len(
            name_index.get(
                token,
                []
            )
        ) <= MAX_TOKEN_FREQUENCY
    ]

    for i in range(
        len(rare_name_tokens)
    ):

        for j in range(
            i + 1,
            len(rare_name_tokens)
        ):

            token1 = rare_name_tokens[i]

            token2 = rare_name_tokens[j]

            set1 = set(
                name_index.get(
                    token1,
                    []
                )
            )

            set2 = set(
                name_index.get(
                    token2,
                    []
                )
            )

            candidates.update(
                set1.intersection(set2)
            )


    # --------------------------------------------------------
    # Route 2:
    # Two address tokens
    # --------------------------------------------------------

    rare_address_tokens = [

        token

        for token in address_tokens

        if len(
            address_index.get(
                token,
                []
            )
        ) <= MAX_TOKEN_FREQUENCY
    ]

    for i in range(
        len(rare_address_tokens)
    ):

        for j in range(
            i + 1,
            len(rare_address_tokens)
        ):

            token1 = rare_address_tokens[i]

            token2 = rare_address_tokens[j]

            set1 = set(
                address_index.get(
                    token1,
                    []
                )
            )

            set2 = set(
                address_index.get(
                    token2,
                    []
                )
            )

            candidates.update(
                set1.intersection(set2)
            )


    # --------------------------------------------------------
    # Route 3:
    # Number + address token
    # --------------------------------------------------------

    for number in numbers:

        number_candidates = set(
            number_index.get(
                number,
                []
            )
        )

        for token in rare_address_tokens:

            address_candidates = set(
                address_index.get(
                    token,
                    []
                )
            )

            candidates.update(
                number_candidates.intersection(
                    address_candidates
                )
            )


    # --------------------------------------------------------
    # Route 4:
    # Name token + address token
    # --------------------------------------------------------

    for name_token in rare_name_tokens:

        name_candidates = set(
            name_index.get(
                name_token,
                []
            )
        )

        for address_token in rare_address_tokens:

            address_candidates = set(
                address_index.get(
                    address_token,
                    []
                )
            )

            candidates.update(
                name_candidates.intersection(
                    address_candidates
                )
            )


    # --------------------------------------------------------
    # COUNTRY HARD FILTER
    # --------------------------------------------------------

    candidates = candidates.intersection(
        country_candidates
    )


    return candidates


# ============================================================
# GENERATE TEST CANDIDATES
# ============================================================

def generate_test_candidates(
    s1,
    name_index,
    address_index,
    number_index,
    country_index
):

    print("\n" + "=" * 60)
    print("GENERATING TEST CANDIDATES")
    print("=" * 60)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    total_candidates = 0

    processed = 0

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        # Header
        f.write(
            "source1_entity_id\tcandidate_entity_id\n"
        )

        for _, row in s1.iterrows():

            source1_id = row["entity_id"]

            candidates = generate_candidates(

                row,

                name_index,

                address_index,

                number_index,

                country_index
            )

            # Remove accidental S1 IDs
            candidates = {

                candidate

                for candidate in candidates

                if candidate.startswith("S2-")
                or candidate.startswith("S3-")
            }

            # Write candidates
            for candidate in sorted(candidates):

                f.write(
                    f"{source1_id}\t{candidate}\n"
                )

            total_candidates += len(
                candidates
            )

            processed += 1

            if processed % 10000 == 0:

                print(
                    f"Processed: "
                    f"{processed:,} / "
                    f"{len(s1):,} | "
                    f"Candidates: "
                    f"{total_candidates:,}"
                )


    print("\n" + "=" * 60)
    print("BLOCKING COMPLETE")
    print("=" * 60)

    print(
        f"Source 1 entities: "
        f"{len(s1):,}"
    )

    print(
        f"Total candidate pairs: "
        f"{total_candidates:,}"
    )

    print(
        f"Output:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# MAIN
# ============================================================

def main():

    s1, s2, s3 = load_data()

    (
        name_index,
        address_index,
        number_index,
        country_index,
        all_sources
    ) = build_indexes(
        s2,
        s3
    )

    generate_test_candidates(
        s1,
        name_index,
        address_index,
        number_index,
        country_index
    )


if __name__ == "__main__":

    main()