import os
import pandas as pd
from collections import defaultdict, Counter
from itertools import combinations

from preprocessing import (
    normalize_name,
    normalize_address,
    get_tokens,
    extract_numbers
)


# ============================================================
# FILES
# ============================================================

S1_FILE = "../dataset/train/train_source1.tsv"
S2_FILE = "../dataset/train/train_source2.tsv"
S3_FILE = "../dataset/train/train_source3.tsv"

OUTPUT_DIR = "../output"

CANDIDATE_FILE = os.path.join(
    OUTPUT_DIR,
    "candidate_pairs_v4.tsv"
)


# ============================================================
# BLOCKING SETTINGS
# ============================================================

# Compound block cannot contain extremely common tokens.
MAX_TOKEN_FREQUENCY = 3000

# ============================================================
# HELPER
# ============================================================

def pair_key(a, b):
    if a < b:
        return (a, b)
    return (b, a)


# ============================================================
# BUILD FREQUENCIES
# ============================================================

def calculate_frequencies(s2, s3):

    print("\nCalculating token frequencies...")

    name_freq = Counter()
    address_freq = Counter()
    number_freq = Counter()

    for df_name, df in [("S2", s2), ("S3", s3)]:

        print(f"Scanning {df_name}...")

        for name in df["business_name"]:

            if not isinstance(name, str):
                continue

            tokens = set(
                get_tokens(
                    normalize_name(name)
                )
            )

            for token in tokens:

                if len(token) >= 2:
                    name_freq[token] += 1

        for address in df["business_address"]:

            if not isinstance(address, str):
                continue

            tokens = set(
                get_tokens(
                    normalize_address(address)
                )
            )

            for token in tokens:

                if len(token) >= 2:
                    address_freq[token] += 1

            numbers = set(
                extract_numbers(address)
            )

            for number in numbers:
                number_freq[number] += 1

    print(
        "Name tokens:",
        f"{len(name_freq):,}"
    )

    print(
        "Address tokens:",
        f"{len(address_freq):,}"
    )

    print(
        "Numbers:",
        f"{len(number_freq):,}"
    )

    return name_freq, address_freq, number_freq


# ============================================================
# BUILD BLOCKING INDEXES
# ============================================================

def build_indexes(
    s2,
    s3,
    name_freq,
    address_freq,
    number_freq
):

    print("\n" + "=" * 70)
    print("BUILDING V4 COMPOUND INDEXES")
    print("=" * 70)

    # --------------------------------------------------------
    # Four blocking routes
    # --------------------------------------------------------

    name_pair_index = defaultdict(list)

    address_pair_index = defaultdict(list)

    number_address_index = defaultdict(list)

    name_address_index = defaultdict(list)

    for source, df in [
        ("S2", s2),
        ("S3", s3)
    ]:

        print(
            f"\nIndexing {source}..."
        )

        for idx in range(len(df)):

            country = df.iloc[idx]["country"]

            name = df.iloc[idx]["business_name"]

            address = df.iloc[idx]["business_address"]

            # ------------------------------------------------
            # NAME TOKENS
            # ------------------------------------------------

            if isinstance(name, str):

                name_tokens = set(
                    get_tokens(
                        normalize_name(name)
                    )
                )

                name_tokens = [
                    x for x in name_tokens
                    if len(x) >= 2
                    and name_freq[x] <= MAX_TOKEN_FREQUENCY
                ]

            else:

                name_tokens = []

            # ------------------------------------------------
            # ADDRESS TOKENS
            # ------------------------------------------------

            if isinstance(address, str):

                address_tokens = set(
                    get_tokens(
                        normalize_address(address)
                    )
                )

                address_tokens = [
                    x for x in address_tokens
                    if len(x) >= 2
                    and address_freq[x] <= MAX_TOKEN_FREQUENCY
                ]

                numbers = set(
                    extract_numbers(address)
                )

                numbers = [
                    x for x in numbers
                    if number_freq[x] <= MAX_TOKEN_FREQUENCY
                ]

            else:

                address_tokens = []
                numbers = []

            # =================================================
            # ROUTE 1
            # COUNTRY + TWO NAME TOKENS
            # =================================================

            for a, b in combinations(
                name_tokens,
                2
            ):

                key = (
                    country,
                    *pair_key(a, b)
                )

                name_pair_index[key].append(
                    (source, idx)
                )

            # =================================================
            # ROUTE 2
            # COUNTRY + TWO ADDRESS TOKENS
            # =================================================

            for a, b in combinations(
                address_tokens,
                2
            ):

                key = (
                    country,
                    *pair_key(a, b)
                )

                address_pair_index[key].append(
                    (source, idx)
                )

            # =================================================
            # ROUTE 3
            # COUNTRY + NUMBER + ADDRESS TOKEN
            # =================================================

            for number in numbers:

                for token in address_tokens:

                    key = (
                        country,
                        number,
                        token
                    )

                    number_address_index[key].append(
                        (source, idx)
                    )

            # =================================================
            # ROUTE 4
            # COUNTRY + NAME TOKEN + ADDRESS TOKEN
            # =================================================

            for name_token in name_tokens:

                for address_token in address_tokens:

                    key = (
                        country,
                        name_token,
                        address_token
                    )

                    name_address_index[key].append(
                        (source, idx)
                    )

        print(
            f"Finished {source}"
        )

    print("\nIndexes built.")

    return (
        name_pair_index,
        address_pair_index,
        number_address_index,
        name_address_index
    )


# ============================================================
# GET CANDIDATES FOR ONE S1
# ============================================================

def get_candidates(
    row,
    name_pair_index,
    address_pair_index,
    number_address_index,
    name_address_index,
    name_freq,
    address_freq,
    number_freq
):

    candidates = set()

    country = row["country"]

    name = row["business_name"]

    address = row["business_address"]

    # --------------------------------------------------------
    # NAME TOKENS
    # --------------------------------------------------------

    if isinstance(name, str):

        name_tokens = set(
            get_tokens(
                normalize_name(name)
            )
        )

        name_tokens = [
            x for x in name_tokens
            if len(x) >= 2
            and name_freq.get(
                x,
                999999999
            ) <= MAX_TOKEN_FREQUENCY
        ]

    else:

        name_tokens = []

    # --------------------------------------------------------
    # ADDRESS TOKENS
    # --------------------------------------------------------

    if isinstance(address, str):

        address_tokens = set(
            get_tokens(
                normalize_address(address)
            )
        )

        address_tokens = [
            x for x in address_tokens
            if len(x) >= 2
            and address_freq.get(
                x,
                999999999
            ) <= MAX_TOKEN_FREQUENCY
        ]

        numbers = set(
            extract_numbers(address)
        )

        numbers = [
            x for x in numbers
            if number_freq.get(
                x,
                999999999
            ) <= MAX_TOKEN_FREQUENCY
        ]

    else:

        address_tokens = []
        numbers = []

    # ========================================================
    # ROUTE 1
    # COUNTRY + NAME PAIR
    # ========================================================

    for a, b in combinations(
        name_tokens,
        2
    ):

        key = (
            country,
            *pair_key(a, b)
        )

        for candidate in name_pair_index.get(
            key,
            []
        ):

            candidates.add(candidate)

    # ========================================================
    # ROUTE 2
    # COUNTRY + ADDRESS PAIR
    # ========================================================

    for a, b in combinations(
        address_tokens,
        2
    ):

        key = (
            country,
            *pair_key(a, b)
        )

        for candidate in address_pair_index.get(
            key,
            []
        ):

            candidates.add(candidate)

    # ========================================================
    # ROUTE 3
    # COUNTRY + NUMBER + ADDRESS TOKEN
    # ========================================================

    for number in numbers:

        for token in address_tokens:

            key = (
                country,
                number,
                token
            )

            for candidate in number_address_index.get(
                key,
                []
            ):

                candidates.add(candidate)

    # ========================================================
    # ROUTE 4
    # COUNTRY + NAME + ADDRESS
    # ========================================================

    for name_token in name_tokens:

        for address_token in address_tokens:

            key = (
                country,
                name_token,
                address_token
            )

            for candidate in name_address_index.get(
                key,
                []
            ):

                candidates.add(candidate)

    return candidates


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("V4 FULL DATASET BLOCKING")
    print("=" * 70)

    print("\nLoading S1...")

    s1 = pd.read_csv(
        S1_FILE,
        sep="\t",
        dtype=str
    )

    print("Loading S2...")

    s2 = pd.read_csv(
        S2_FILE,
        sep="\t",
        dtype=str
    )

    print("Loading S3...")

    s3 = pd.read_csv(
        S3_FILE,
        sep="\t",
        dtype=str
    )

    print()
    print(
        "S1:",
        f"{len(s1):,}"
    )

    print(
        "S2:",
        f"{len(s2):,}"
    )

    print(
        "S3:",
        f"{len(s3):,}"
    )

    # --------------------------------------------------------
    # FREQUENCIES
    # --------------------------------------------------------

    (
        name_freq,
        address_freq,
        number_freq
    ) = calculate_frequencies(
        s2,
        s3
    )

    # --------------------------------------------------------
    # INDEXES
    # --------------------------------------------------------

    (
        name_pair_index,
        address_pair_index,
        number_address_index,
        name_address_index
    ) = build_indexes(
        s2,
        s3,
        name_freq,
        address_freq,
        number_freq
    )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if os.path.exists(
        CANDIDATE_FILE
    ):

        print(
            "\nRemoving old V4 candidate file..."
        )

        os.remove(
            CANDIDATE_FILE
        )

    print(
        "\nWriting:",
        CANDIDATE_FILE
    )

    # --------------------------------------------------------
    # STREAM OUTPUT
    # --------------------------------------------------------

    total_pairs = 0

    buffer = []

    BUFFER_SIZE = 100000

    total_s1 = len(s1)

    with open(
        CANDIDATE_FILE,
        "w",
        encoding="utf-8"
    ) as output:

        output.write(
            "source1_entity_id\tcandidate_entity_id\n"
        )

        # ----------------------------------------------------
        # PROCESS S1
        # ----------------------------------------------------

        for count in range(
            total_s1
        ):

            row = s1.iloc[count]

            candidates = get_candidates(
                row,
                name_pair_index,
                address_pair_index,
                number_address_index,
                name_address_index,
                name_freq,
                address_freq,
                number_freq
            )

            s1_id = row["entity_id"]

            # ------------------------------------------------
            # CONVERT TARGET INDEX → ENTITY ID
            # ------------------------------------------------

            for source, idx in candidates:

                if source == "S2":

                    target_id = s2.iloc[
                        idx
                    ]["entity_id"]

                else:

                    target_id = s3.iloc[
                        idx
                    ]["entity_id"]

                buffer.append(
                    f"{s1_id}\t{target_id}\n"
                )

            total_pairs += len(candidates)

            # ------------------------------------------------
            # WRITE EVERY 100K ROWS
            # ------------------------------------------------

            if len(buffer) >= BUFFER_SIZE:

                output.writelines(
                    buffer
                )

                buffer.clear()

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if (
                (count + 1) % 10000 == 0
            ):

                avg = (
                    total_pairs
                    /
                    (count + 1)
                )

                print(
                    f"Processed "
                    f"{count + 1:,}/"
                    f"{total_s1:,} | "
                    f"Pairs: "
                    f"{total_pairs:,} | "
                    f"Avg: "
                    f"{avg:,.2f}"
                )

        # ----------------------------------------------------
        # FINAL BUFFER
        # ----------------------------------------------------

        if buffer:

            output.writelines(
                buffer
            )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("V4 BLOCKING COMPLETE")
    print("=" * 70)

    print(
        "S1 processed:",
        f"{total_s1:,}"
    )

    print(
        "Total candidate pairs:",
        f"{total_pairs:,}"
    )

    print(
        "Average candidates/S1:",
        f"{total_pairs / total_s1:,.2f}"
    )

    print(
        "Candidate file:",
        os.path.abspath(
            CANDIDATE_FILE
        )
    )


if __name__ == "__main__":
    main()