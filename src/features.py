import os
import re
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz.fuzz import ratio


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAINING_PAIRS = os.path.join(
    BASE_DIR,
    "output",
    "training_pairs.tsv"
)

SOURCE1_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "train",
    "train_source1.tsv"
)

SOURCE2_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "train",
    "train_source2.tsv"
)

SOURCE3_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "train",
    "train_source3.tsv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "output",
    "training_features.tsv"
)


# ============================================================
# SETTINGS
# ============================================================

CHUNK_SIZE = 100_000


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Keep unicode characters.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# TOKEN FUNCTIONS
# ============================================================

def get_tokens(text):

    if not text:
        return set()

    return set(text.split())


def jaccard(a, b):

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    intersection = len(a & b)
    union = len(a | b)

    if union == 0:
        return 0.0

    return intersection / union


def number_tokens(text):

    return set(
        re.findall(
            r"\d+",
            text
        )
    )


def number_overlap(a, b):

    nums_a = number_tokens(a)
    nums_b = number_tokens(b)

    if not nums_a or not nums_b:
        return 0.0

    return len(nums_a & nums_b) / max(
        len(nums_a),
        len(nums_b)
    )


# ============================================================
# CHARACTER N-GRAM SIMILARITY
# ============================================================

def char_ngram_similarity(
    a,
    b,
    n=3
):

    if not a or not b:
        return 0.0

    if len(a) < n or len(b) < n:
        return ratio(a, b) / 100.0

    grams_a = {
        a[i:i+n]
        for i in range(len(a) - n + 1)
    }

    grams_b = {
        b[i:i+n]
        for i in range(len(b) - n + 1)
    }

    return jaccard(
        grams_a,
        grams_b
    )


# ============================================================
# LOAD SOURCE DATA
# ============================================================

def load_sources():

    print("=" * 60)
    print("LOADING SOURCE DATA")
    print("=" * 60)

    print("Loading Source 1...")

    s1 = pd.read_csv(
        SOURCE1_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    print(
        f"Source 1: {len(s1):,}"
    )

    print("Loading Source 2...")

    s2 = pd.read_csv(
        SOURCE2_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    print(
        f"Source 2: {len(s2):,}"
    )

    print("Loading Source 3...")

    s3 = pd.read_csv(
        SOURCE3_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    )

    print(
        f"Source 3: {len(s3):,}"
    )

    # --------------------------------------------------------
    # Convert to dictionaries.
    #
    # Only the columns needed for feature generation are kept.
    # --------------------------------------------------------

    s1 = s1.set_index("entity_id")
    s2 = s2.set_index("entity_id")
    s3 = s3.set_index("entity_id")

    return s1, s2, s3


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_features(
    s1_row,
    candidate_row
):

    name1 = normalize_text(
        s1_row["business_name"]
    )

    name2 = normalize_text(
        candidate_row["business_name"]
    )

    address1 = normalize_text(
        s1_row["business_address"]
    )

    address2 = normalize_text(
        candidate_row["business_address"]
    )

    country1 = normalize_text(
        s1_row["country"]
    )

    country2 = normalize_text(
        candidate_row["country"]
    )

    # --------------------------------------------------------
    # Tokens
    # --------------------------------------------------------

    name_tokens1 = get_tokens(name1)
    name_tokens2 = get_tokens(name2)

    address_tokens1 = get_tokens(address1)
    address_tokens2 = get_tokens(address2)

    # --------------------------------------------------------
    # Name features
    # --------------------------------------------------------

    name_jaccard = jaccard(
        name_tokens1,
        name_tokens2
    )

    name_char_similarity = char_ngram_similarity(
        name1,
        name2
    )

    name_fuzzy = ratio(
        name1,
        name2
    ) / 100.0

    # --------------------------------------------------------
    # Address features
    # --------------------------------------------------------

    address_jaccard = jaccard(
        address_tokens1,
        address_tokens2
    )

    address_char_similarity = char_ngram_similarity(
        address1,
        address2
    )

    address_fuzzy = ratio(
        address1,
        address2
    ) / 100.0

    # --------------------------------------------------------
    # Number overlap
    # --------------------------------------------------------

    numbers = number_overlap(
        address1,
        address2
    )

    # --------------------------------------------------------
    # Token overlap
    # --------------------------------------------------------

    if name_tokens1 or name_tokens2:

        name_token_overlap = len(
            name_tokens1 & name_tokens2
        ) / max(
            len(name_tokens1),
            len(name_tokens2),
            1
        )

    else:

        name_token_overlap = 0.0

    if address_tokens1 or address_tokens2:

        address_token_overlap = len(
            address_tokens1 & address_tokens2
        ) / max(
            len(address_tokens1),
            len(address_tokens2),
            1
        )

    else:

        address_token_overlap = 0.0

    # --------------------------------------------------------
    # Missing indicators
    # --------------------------------------------------------

    name1_missing = int(
        len(name1) == 0
    )

    name2_missing = int(
        len(name2) == 0
    )

    address1_missing = int(
        len(address1) == 0
    )

    address2_missing = int(
        len(address2) == 0
    )

    # --------------------------------------------------------
    # Country
    # --------------------------------------------------------

    country_match = int(
        country1 != ""
        and country1 == country2
    )

    # --------------------------------------------------------
    # Length ratios
    # --------------------------------------------------------

    name_length_ratio = (
        min(len(name1), len(name2))
        / max(len(name1), len(name2), 1)
    )

    address_length_ratio = (
        min(len(address1), len(address2))
        / max(len(address1), len(address2), 1)
    )

    return [
        name_jaccard,
        name_char_similarity,
        name_fuzzy,
        name_token_overlap,

        address_jaccard,
        address_char_similarity,
        address_fuzzy,
        address_token_overlap,

        numbers,

        country_match,

        name1_missing,
        name2_missing,

        address1_missing,
        address2_missing,

        name_length_ratio,
        address_length_ratio
    ]


# ============================================================
# PROCESS CHUNK
# ============================================================

def process_chunk(
    chunk,
    s1,
    s2,
    s3
):

    features = []

    for row in chunk.itertuples(index=False):

        s1_id = row.source1_entity_id
        candidate_id = row.candidate_entity_id
        label = row.label

        # ----------------------------------------------------
        # Find S1
        # ----------------------------------------------------

        if s1_id not in s1.index:
            continue

        s1_row = s1.loc[s1_id]

        # ----------------------------------------------------
        # Determine S2 / S3
        # ----------------------------------------------------

        if candidate_id.startswith("S2-"):

            if candidate_id not in s2.index:
                continue

            candidate_row = s2.loc[candidate_id]

        elif candidate_id.startswith("S3-"):

            if candidate_id not in s3.index:
                continue

            candidate_row = s3.loc[candidate_id]

        else:

            continue

        # ----------------------------------------------------
        # Calculate features
        # ----------------------------------------------------

        values = calculate_features(
            s1_row,
            candidate_row
        )

        features.append(
            [
                s1_id,
                candidate_id,
                label
            ] + values
        )

    return features


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load source data
    # --------------------------------------------------------

    s1, s2, s3 = load_sources()

    # --------------------------------------------------------
    # Feature column names
    # --------------------------------------------------------

    columns = [
        "source1_entity_id",
        "candidate_entity_id",
        "label",

        "name_jaccard",
        "name_char_similarity",
        "name_fuzzy",
        "name_token_overlap",

        "address_jaccard",
        "address_char_similarity",
        "address_fuzzy",
        "address_token_overlap",

        "number_overlap",

        "country_match",

        "name1_missing",
        "name2_missing",

        "address1_missing",
        "address2_missing",

        "name_length_ratio",
        "address_length_ratio"
    ]

    # --------------------------------------------------------
    # Remove old output
    # --------------------------------------------------------

    if os.path.exists(OUTPUT_FILE):

        os.remove(
            OUTPUT_FILE
        )

    first_chunk = True

    total_processed = 0

    # --------------------------------------------------------
    # Read training pairs in chunks
    # --------------------------------------------------------

    for chunk in pd.read_csv(
        TRAINING_PAIRS,
        sep="\t",
        dtype=str,
        chunksize=CHUNK_SIZE
    ):

        print(
            f"Processing rows "
            f"{total_processed:,} - "
            f"{total_processed + len(chunk):,}"
        )

        # label should be integer
        chunk["label"] = chunk[
            "label"
        ].astype(int)

        feature_rows = process_chunk(
            chunk,
            s1,
            s2,
            s3
        )

        if feature_rows:

            feature_df = pd.DataFrame(
                feature_rows,
                columns=columns
            )

            feature_df.to_csv(
                OUTPUT_FILE,
                sep="\t",
                index=False,
                mode="w" if first_chunk else "a",
                header=first_chunk
            )

            first_chunk = False

        total_processed += len(chunk)

    print("\n" + "=" * 60)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 60)

    print(
        "Rows processed:",
        f"{total_processed:,}"
    )

    print(
        "Output:",
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()