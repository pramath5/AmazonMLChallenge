import pandas as pd
import re
import unicodedata


GROUND_TRUTH = "../dataset/train/train_ground_truth.tsv"
SOURCE1 = "../dataset/train/train_source1.tsv"
SOURCE2 = "../dataset/train/train_source2.tsv"
SOURCE3 = "../dataset/train/train_source3.tsv"


def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def token_set(text):
    return set(normalize_text(text).split())


def jaccard(a, b):
    a = token_set(a)
    b = token_set(b)

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def load_source(path):
    return pd.read_csv(
        path,
        sep="\t",
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country"
        ]
    ).set_index("entity_id")


def get_record(source, entity_id):
    """
    Return one record from an indexed dataframe.
    """
    try:
        return source.loc[entity_id]
    except KeyError:
        return None


def analyze_name_patterns(
    s1,
    s2,
    s3,
    ground_truth
):

    print("\n" + "=" * 70)
    print("NAME MATCHING PATTERNS")
    print("=" * 70)

    total_matches = 0
    exact = 0
    high_similarity = 0
    low_similarity = 0
    empty_name = 0

    examples = []

    for _, row in ground_truth.iterrows():

        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]

        if pd.isna(matched) or str(matched).strip() == "":
            continue

        source1 = get_record(s1, s1_id)

        if source1 is None:
            continue

        s1_name = source1["business_name"]

        for match_id in str(matched).split(","):

            match_id = match_id.strip()

            if match_id.startswith("S2-"):
                target = get_record(s2, match_id)
            elif match_id.startswith("S3-"):
                target = get_record(s3, match_id)
            else:
                continue

            if target is None:
                continue

            target_name = target["business_name"]

            total_matches += 1

            if pd.isna(target_name):
                empty_name += 1
                continue

            normalized_s1 = normalize_text(s1_name)
            normalized_target = normalize_text(target_name)

            sim = jaccard(
                s1_name,
                target_name
            )

            if normalized_s1 == normalized_target:
                exact += 1

            if sim >= 0.5:
                high_similarity += 1

            if sim < 0.3:
                low_similarity += 1

                if len(examples) < 15:
                    examples.append(
                        (
                            s1_name,
                            target_name,
                            sim
                        )
                    )

    print(
        f"Total ground-truth matches analyzed : "
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


def analyze_address_patterns(
    s1,
    s2,
    s3,
    ground_truth
):

    print("\n" + "=" * 70)
    print("ADDRESS MATCHING PATTERNS")
    print("=" * 70)

    total = 0
    both_present = 0
    high_similarity = 0
    low_similarity = 0

    examples = []

    for _, row in ground_truth.iterrows():

        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]

        if pd.isna(matched) or str(matched).strip() == "":
            continue

        source1 = get_record(s1, s1_id)

        if source1 is None:
            continue

        address1 = source1["business_address"]

        for match_id in str(matched).split(","):

            match_id = match_id.strip()

            if match_id.startswith("S2-"):
                target = get_record(s2, match_id)
            elif match_id.startswith("S3-"):
                target = get_record(s3, match_id)
            else:
                continue

            if target is None:
                continue

            address2 = target["business_address"]

            total += 1

            if pd.isna(address1) or pd.isna(address2):
                continue

            both_present += 1

            sim = jaccard(
                address1,
                address2
            )

            if sim >= 0.5:
                high_similarity += 1

            if sim < 0.3:
                low_similarity += 1

                if len(examples) < 15:
                    examples.append(
                        (
                            address1,
                            address2,
                            sim
                        )
                    )

    print(
        f"Total matches                  : "
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


def analyze_country(
    s1,
    s2,
    s3,
    ground_truth
):

    print("\n" + "=" * 70)
    print("COUNTRY CONSISTENCY")
    print("=" * 70)

    total = 0
    same_country = 0
    different_country = 0

    for _, row in ground_truth.iterrows():

        s1_id = row["source1_entity_id"]
        matched = row["matched_entity_ids"]

        if pd.isna(matched) or str(matched).strip() == "":
            continue

        source1 = get_record(s1, s1_id)

        if source1 is None:
            continue

        country1 = source1["country"]

        for match_id in str(matched).split(","):

            match_id = match_id.strip()

            if match_id.startswith("S2-"):
                target = get_record(s2, match_id)
            elif match_id.startswith("S3-"):
                target = get_record(s3, match_id)
            else:
                continue

            if target is None:
                continue

            country2 = target["country"]

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


def main():

    print("Loading training data...")

    s1 = load_source(SOURCE1)

    print("Source 1 loaded.")

    s2 = load_source(SOURCE2)

    print("Source 2 loaded.")

    s3 = load_source(SOURCE3)

    print("Source 3 loaded.")

    ground_truth = pd.read_csv(
        GROUND_TRUTH,
        sep="\t",
        usecols=[
            "source1_entity_id",
            "matched_entity_ids"
        ]
    )

    print("Ground truth loaded.")

    analyze_name_patterns(
        s1,
        s2,
        s3,
        ground_truth
    )

    analyze_address_patterns(
        s1,
        s2,
        s3,
        ground_truth
    )

    analyze_country(
        s1,
        s2,
        s3,
        ground_truth
    )


if __name__ == "__main__":
    main()