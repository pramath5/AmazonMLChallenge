import os
import pandas as pd
from collections import defaultdict, Counter


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
    "candidate_pairs.tsv"
)


# ============================================================
# V3 SETTINGS
# KEEPING THE SETTINGS THAT GAVE 632.8 AVG CANDIDATES
# ============================================================

MAX_SINGLE_TOKEN = 500
MAX_COMPOUND_BLOCK = 5000


# ============================================================
# HELPER
# ============================================================

def sorted_pair(a, b):

    if a <= b:
        return a, b

    return b, a


# ============================================================
# BUILD INDEXES
# ============================================================

def build_indexes(s2, s3):

    print("\n" + "=" * 70)
    print("BUILDING V3 BLOCKING INDEXES")
    print("=" * 70)

    name_frequency = Counter()
    address_frequency = Counter()
    number_frequency = Counter()

    # --------------------------------------------------------
    # NAME FREQUENCY
    # --------------------------------------------------------

    print("\nCounting name token frequencies...")

    for source, df in [("S2", s2), ("S3", s3)]:

        print(f"Processing {source} names...")

        for value in df["business_name"]:

            if not isinstance(value, str):
                continue

            tokens = set(
                get_tokens(
                    normalize_name(value)
                )
            )

            for token in tokens:

                if len(token) >= 2:
                    name_frequency[token] += 1

    print(
        "Unique name tokens:",
        f"{len(name_frequency):,}"
    )

    # --------------------------------------------------------
    # ADDRESS FREQUENCY
    # --------------------------------------------------------

    print("\nCounting address token frequencies...")

    for source, df in [("S2", s2), ("S3", s3)]:

        print(f"Processing {source} addresses...")

        for value in df["business_address"]:

            if not isinstance(value, str):
                continue

            tokens = set(
                get_tokens(
                    normalize_address(value)
                )
            )

            for token in tokens:

                if len(token) >= 2:
                    address_frequency[token] += 1

    print(
        "Unique address tokens:",
        f"{len(address_frequency):,}"
    )

    # --------------------------------------------------------
    # NUMBER FREQUENCY
    # --------------------------------------------------------

    print("\nCounting address number frequencies...")

    for source, df in [("S2", s2), ("S3", s3)]:

        print(f"Processing {source} numbers...")

        for value in df["business_address"]:

            if not isinstance(value, str):
                continue

            numbers = set(
                extract_numbers(value)
            )

            for number in numbers:
                number_frequency[number] += 1

    print(
        "Unique numbers:",
        f"{len(number_frequency):,}"
    )

    # --------------------------------------------------------
    # INDEXES
    # --------------------------------------------------------

    name_pair_index = defaultdict(list)
    address_pair_index = defaultdict(list)
    number_address_index = defaultdict(list)

    name_single_index = defaultdict(list)
    address_single_index = defaultdict(list)

    # --------------------------------------------------------
    # BUILD TARGET INDEXES
    # --------------------------------------------------------

    for source, df in [
        ("S2", s2),
        ("S3", s3)
    ]:

        print(
            f"\nBuilding indexes for {source}..."
        )

        for idx in range(len(df)):

            name_value = df.iloc[idx]["business_name"]
            address_value = df.iloc[idx]["business_address"]

            # ------------------------------------------------
            # NAME TOKENS
            # ------------------------------------------------

            if isinstance(name_value, str):

                name_tokens = list(
                    set(
                        get_tokens(
                            normalize_name(name_value)
                        )
                    )
                )

                name_tokens = [
                    token
                    for token in name_tokens
                    if len(token) >= 2
                ]

            else:

                name_tokens = []

            # ------------------------------------------------
            # ADDRESS TOKENS
            # ------------------------------------------------

            if isinstance(address_value, str):

                address_tokens = list(
                    set(
                        get_tokens(
                            normalize_address(address_value)
                        )
                    )
                )

                address_tokens = [
                    token
                    for token in address_tokens
                    if len(token) >= 2
                ]

            else:

                address_tokens = []

            # ------------------------------------------------
            # NUMBERS
            # ------------------------------------------------

            if isinstance(address_value, str):

                numbers = list(
                    set(
                        extract_numbers(address_value)
                    )
                )

            else:

                numbers = []

            # =================================================
            # NAME PAIRS
            # =================================================

            for i in range(len(name_tokens)):

                for j in range(i + 1, len(name_tokens)):

                    a = name_tokens[i]
                    b = name_tokens[j]

                    if (
                        name_frequency[a]
                        <= MAX_COMPOUND_BLOCK
                        and
                        name_frequency[b]
                        <= MAX_COMPOUND_BLOCK
                    ):

                        key = sorted_pair(a, b)

                        name_pair_index[key].append(
                            (source, idx)
                        )

            # =================================================
            # ADDRESS PAIRS
            # =================================================

            for i in range(len(address_tokens)):

                for j in range(i + 1, len(address_tokens)):

                    a = address_tokens[i]
                    b = address_tokens[j]

                    if (
                        address_frequency[a]
                        <= MAX_COMPOUND_BLOCK
                        and
                        address_frequency[b]
                        <= MAX_COMPOUND_BLOCK
                    ):

                        key = sorted_pair(a, b)

                        address_pair_index[key].append(
                            (source, idx)
                        )

            # =================================================
            # NUMBER + ADDRESS
            # =================================================

            for number in numbers:

                if (
                    number_frequency[number]
                    > MAX_COMPOUND_BLOCK
                ):
                    continue

                for token in address_tokens:

                    if (
                        address_frequency[token]
                        > MAX_COMPOUND_BLOCK
                    ):
                        continue

                    key = (number, token)

                    number_address_index[key].append(
                        (source, idx)
                    )

            # =================================================
            # RARE NAME TOKEN
            # =================================================

            for token in name_tokens:

                if (
                    name_frequency[token]
                    <= MAX_SINGLE_TOKEN
                ):

                    name_single_index[token].append(
                        (source, idx)
                    )

            # =================================================
            # RARE ADDRESS TOKEN
            # =================================================

            for token in address_tokens:

                if (
                    address_frequency[token]
                    <= MAX_SINGLE_TOKEN
                ):

                    address_single_index[token].append(
                        (source, idx)
                    )

        print(
            f"Finished {source}"
        )

    return (
        name_pair_index,
        address_pair_index,
        number_address_index,
        name_single_index,
        address_single_index,
        name_frequency,
        address_frequency,
        number_frequency
    )


# ============================================================
# PRECOMPUTE COUNTRY ARRAYS
# ============================================================

def build_country_sets(s2, s3):

    print("\nBuilding country filters...")

    s2_countries = {}

    for idx, country in enumerate(s2["country"]):

        s2_countries[idx] = country

    s3_countries = {}

    for idx, country in enumerate(s3["country"]):

        s3_countries[idx] = country

    return s2_countries, s3_countries


# ============================================================
# GET CANDIDATES
# ============================================================

def get_candidates(
    s1_row,
    name_pair_index,
    address_pair_index,
    number_address_index,
    name_single_index,
    address_single_index,
    name_frequency,
    address_frequency,
    number_frequency,
    s2_countries,
    s3_countries
):

    candidates = set()

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    name_value = s1_row["business_name"]
    address_value = s1_row["business_address"]
    country = s1_row["country"]

    if isinstance(name_value, str):

        name_tokens = list(
            set(
                get_tokens(
                    normalize_name(name_value)
                )
            )
        )

        name_tokens = [
            token
            for token in name_tokens
            if len(token) >= 2
        ]

    else:

        name_tokens = []

    if isinstance(address_value, str):

        address_tokens = list(
            set(
                get_tokens(
                    normalize_address(address_value)
                )
            )
        )

        address_tokens = [
            token
            for token in address_tokens
            if len(token) >= 2
        ]

        numbers = list(
            set(
                extract_numbers(address_value)
            )
        )

    else:

        address_tokens = []
        numbers = []

    # ========================================================
    # ROUTE 1 — TWO NAME TOKENS
    # ========================================================

    for i in range(len(name_tokens)):

        for j in range(i + 1, len(name_tokens)):

            a = name_tokens[i]
            b = name_tokens[j]

            if (
                name_frequency.get(a, 999999999)
                <= MAX_COMPOUND_BLOCK
                and
                name_frequency.get(b, 999999999)
                <= MAX_COMPOUND_BLOCK
            ):

                key = sorted_pair(a, b)

                for candidate in name_pair_index.get(
                    key,
                    []
                ):

                    candidates.add(candidate)

    # ========================================================
    # ROUTE 2 — TWO ADDRESS TOKENS
    # ========================================================

    for i in range(len(address_tokens)):

        for j in range(i + 1, len(address_tokens)):

            a = address_tokens[i]
            b = address_tokens[j]

            if (
                address_frequency.get(a, 999999999)
                <= MAX_COMPOUND_BLOCK
                and
                address_frequency.get(b, 999999999)
                <= MAX_COMPOUND_BLOCK
            ):

                key = sorted_pair(a, b)

                for candidate in address_pair_index.get(
                    key,
                    []
                ):

                    candidates.add(candidate)

    # ========================================================
    # ROUTE 3 — NUMBER + ADDRESS TOKEN
    # ========================================================

    for number in numbers:

        if (
            number_frequency.get(
                number,
                999999999
            )
            > MAX_COMPOUND_BLOCK
        ):
            continue

        for token in address_tokens:

            if (
                address_frequency.get(
                    token,
                    999999999
                )
                > MAX_COMPOUND_BLOCK
            ):
                continue

            key = (number, token)

            for candidate in number_address_index.get(
                key,
                []
            ):

                candidates.add(candidate)

    # ========================================================
    # ROUTE 4 — RARE NAME TOKEN
    # ========================================================

    for token in name_tokens:

        if (
            name_frequency.get(
                token,
                999999999
            )
            <= MAX_SINGLE_TOKEN
        ):

            for candidate in name_single_index.get(
                token,
                []
            ):

                candidates.add(candidate)

    # ========================================================
    # ROUTE 5 — RARE ADDRESS TOKEN
    # ========================================================

    for token in address_tokens:

        if (
            address_frequency.get(
                token,
                999999999
            )
            <= MAX_SINGLE_TOKEN
        ):

            for candidate in address_single_index.get(
                token,
                []
            ):

                candidates.add(candidate)

    # ========================================================
    # COUNTRY FILTER
    # ========================================================

    filtered = []

    for source, idx in candidates:

        if source == "S2":

            if s2_countries[idx] == country:

                filtered.append(
                    (source, idx)
                )

        else:

            if s3_countries[idx] == country:

                filtered.append(
                    (source, idx)
                )

    return filtered


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FULL V3 BLOCKING")
    print("=" * 70)

    print("\nLoading data...")

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
    print("S1:", f"{len(s1):,}")
    print("S2:", f"{len(s2):,}")
    print("S3:", f"{len(s3):,}")

    # --------------------------------------------------------
    # BUILD INDEXES
    # --------------------------------------------------------

    (
        name_pair_index,
        address_pair_index,
        number_address_index,
        name_single_index,
        address_single_index,
        name_frequency,
        address_frequency,
        number_frequency
    ) = build_indexes(
        s2,
        s3
    )

    # --------------------------------------------------------
    # COUNTRY LOOKUPS
    # --------------------------------------------------------

    s2_countries, s3_countries = build_country_sets(
        s2,
        s3
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # Remove old candidate file
    if os.path.exists(CANDIDATE_FILE):

        os.remove(
            CANDIDATE_FILE
        )

    # --------------------------------------------------------
    # WRITE HEADER
    # --------------------------------------------------------

    with open(
        CANDIDATE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "source1_entity_id\tcandidate_entity_id\n"
        )

    # --------------------------------------------------------
    # PROCESS ALL S1
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING CANDIDATES FOR ALL S1")
    print("=" * 70)

    total_pairs = 0

    buffer = []

    BUFFER_SIZE = 100000

    total_s1 = len(s1)

    for count, (_, row) in enumerate(
        s1.iterrows(),
        start=1
    ):

        candidates = get_candidates(
            row,
            name_pair_index,
            address_pair_index,
            number_address_index,
            name_single_index,
            address_single_index,
            name_frequency,
            address_frequency,
            number_frequency,
            s2_countries,
            s3_countries
        )

        s1_id = row["entity_id"]

        for source, idx in candidates:

            if source == "S2":

                target_id = s2.iloc[idx]["entity_id"]

            else:

                target_id = s3.iloc[idx]["entity_id"]

            buffer.append(
                f"{s1_id}\t{target_id}\n"
            )

        total_pairs += len(candidates)

        # ----------------------------------------------------
        # WRITE BUFFER
        # ----------------------------------------------------

        if len(buffer) >= BUFFER_SIZE:

            with open(
                CANDIDATE_FILE,
                "a",
                encoding="utf-8"
            ) as f:

                f.writelines(buffer)

            buffer.clear()

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if count % 10000 == 0:

            avg = total_pairs / count

            print(
                f"Processed {count:,}/{total_s1:,} "
                f"({count / total_s1 * 100:.2f}%) | "
                f"Pairs: {total_pairs:,} | "
                f"Avg candidates/S1: {avg:,.2f}"
            )

    # --------------------------------------------------------
    # WRITE REMAINING BUFFER
    # --------------------------------------------------------

    if buffer:

        with open(
            CANDIDATE_FILE,
            "a",
            encoding="utf-8"
        ) as f:

            f.writelines(buffer)

    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("BLOCKING COMPLETE")
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
        "Output file:",
        os.path.abspath(CANDIDATE_FILE)
    )


if __name__ == "__main__":
    main()