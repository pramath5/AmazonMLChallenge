import re
import unicodedata
import pandas as pd


# ---------------------------------------------------------
# 1. Basic text normalization
# ---------------------------------------------------------

def normalize_text(text):
    """
    Unicode-safe text normalization.

    Latin accents:
        É -> e
        ó -> o

    Indic and other non-Latin scripts:
        preserved without breaking their characters.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower().strip()

    # NFKD separates Latin accents from their base characters.
    text = unicodedata.normalize("NFKD", text)

    result = []

    for char in text:

        category = unicodedata.category(char)

        # Combining mark
        if category.startswith("M"):

            # Keep combining marks for non-Latin scripts.
            if result:
                previous = result[-1]

                previous_name = unicodedata.name(
                    previous,
                    ""
                )

                if "LATIN" in previous_name:
                    # Remove Latin accent marks.
                    continue

            result.append(char)

        else:
            result.append(char)

    text = "".join(result)

    # Replace punctuation/symbols with spaces,
    # but KEEP Unicode letters, numbers,
    # combining marks and whitespace.
    cleaned = []

    for char in text:

        category = unicodedata.category(char)

        if (
            category[0] in ("L", "N", "M")
            or char == "_"
            or char.isspace()
        ):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ---------------------------------------------------------
# 2. Name normalization
# ---------------------------------------------------------

def normalize_name(name):
    """
    Normalize a business name.

    We intentionally DO NOT remove non-English scripts.
    Hindi, Tamil, Telugu, etc. can contain useful information.
    """

    return normalize_text(name)


# ---------------------------------------------------------
# 3. Address normalization
# ---------------------------------------------------------

ADDRESS_REPLACEMENTS = {
    "street": "st",
    "avenue": "ave",
    "road": "rd",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "highway": "hwy",
    "parkway": "pkwy",
    "place": "pl",
    "court": "ct",
    "circle": "cir",
    "terrace": "ter",
    "saint": "st",
}


def normalize_address(address):
    """
    Normalize a business address.

    Example:

        "3315 Fremont Street, Peoria, IL"

        ->
        "3315 fremont st peoria il"
    """

    text = normalize_text(address)

    if not text:
        return ""

    tokens = text.split()

    normalized_tokens = []

    for token in tokens:

        replacement = ADDRESS_REPLACEMENTS.get(
            token,
            token
        )

        normalized_tokens.append(replacement)

    return " ".join(normalized_tokens)


# ---------------------------------------------------------
# 4. Extract numbers from addresses
# ---------------------------------------------------------

def extract_numbers(text):
    """
    Extract numeric components.

    Example:

        "1056-1060 Belden Ave, Akron, OH"

        ->
        {"1056", "1060"}
    """

    if pd.isna(text):
        return set()

    text = str(text)

    numbers = re.findall(
        r"\d+",
        text
    )

    return set(numbers)


# ---------------------------------------------------------
# 5. Extract tokens
# ---------------------------------------------------------

def get_tokens(text):
    """
    Return normalized tokens as a set.
    """

    text = normalize_text(text)

    if not text:
        return set()

    return set(text.split())


# ---------------------------------------------------------
# 6. Jaccard similarity
# ---------------------------------------------------------

def jaccard_similarity(text1, text2):
    """
    Token-based Jaccard similarity.

    J(A,B) = |A intersection B| / |A union B|
    """

    tokens1 = get_tokens(text1)
    tokens2 = get_tokens(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    return len(intersection) / len(union)


# ---------------------------------------------------------
# 7. Number overlap
# ---------------------------------------------------------

def number_overlap(text1, text2):
    """
    Measure overlap between numeric components.

    Example:

        "1056 Belden Avenue"
        "1056c Belden Ave"

        -> strong number overlap
    """

    numbers1 = extract_numbers(text1)
    numbers2 = extract_numbers(text2)

    if not numbers1 or not numbers2:
        return 0.0

    intersection = numbers1 & numbers2

    return len(intersection) / min(
        len(numbers1),
        len(numbers2)
    )


# ---------------------------------------------------------
# 8. Character n-grams
# ---------------------------------------------------------

def character_ngrams(text, n=3):
    """
    Create character n-grams.

    Example:

        "payne"

        3-grams:
        pay
        ayn
        yne
    """

    text = normalize_text(text)

    if len(text) < n:
        return {text} if text else set()

    return {
        text[i:i + n]
        for i in range(len(text) - n + 1)
    }


def character_ngram_similarity(
    text1,
    text2,
    n=3
):
    """
    Jaccard similarity over character n-grams.
    """

    grams1 = character_ngrams(
        text1,
        n
    )

    grams2 = character_ngrams(
        text2,
        n
    )

    if not grams1 or not grams2:
        return 0.0

    return len(grams1 & grams2) / len(
        grams1 | grams2
    )


# ---------------------------------------------------------
# 9. Create all useful normalized fields
# ---------------------------------------------------------

def preprocess_dataframe(df):
    """
    Add normalized columns to a dataframe.

    Original columns remain unchanged.
    """

    df = df.copy()

    df["name_normalized"] = (
        df["business_name"]
        .apply(normalize_name)
    )

    df["address_normalized"] = (
        df["business_address"]
        .apply(normalize_address)
    )

    df["name_tokens"] = (
        df["name_normalized"]
        .apply(lambda x: x.split())
    )

    df["address_tokens"] = (
        df["address_normalized"]
        .apply(lambda x: x.split())
    )

    df["address_numbers"] = (
        df["business_address"]
        .apply(extract_numbers)
    )

    return df


# ---------------------------------------------------------
# 10. Small test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING PREPROCESSING")
    print("=" * 60)

    names = [
        "Payne Énterprises",
        "PAYNE-ENRTPRMISES",
        "Raj Investments LLP",
        "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
        "एसएस फूड प्राइवेट लिमिटेड",
    ]

    print("\nNAME NORMALIZATION")
    print("-" * 60)

    for name in names:
        print(
            f"{name}"
            f" -> "
            f"{normalize_name(name)}"
        )

    addresses = [
        "3315 Fremont Street, Peoria, IL",
        "Fremont St, Peoria, Illinois",
        "1056-1060 Belden Avenue, Akron, OH",
        "Af-684, Ghaziabad, Uttar Pradesh",
    ]

    print("\nADDRESS NORMALIZATION")
    print("-" * 60)

    for address in addresses:
        print(
            f"{address}"
            f" -> "
            f"{normalize_address(address)}"
        )

    print("\nSIMILARITY TESTS")
    print("-" * 60)

    print(
        "Name Jaccard:",
        jaccard_similarity(
            "Payne Enterprises",
            "Payne Enterprises LLC"
        )
    )

    print(
        "Address Jaccard:",
        jaccard_similarity(
            "3315 Fremont Street, Peoria, IL",
            "Fremont St, Peoria, Illinois"
        )
    )

    print(
        "Number overlap:",
        number_overlap(
            "1056 Belden Avenue, Akron, OH",
            "1056c Belden Ave, Akron, Ohio"
        )
    )

    print(
        "Character similarity:",
        character_ngram_similarity(
            "Payne Enterprises",
            "Payne Enterpires"
        )
    )