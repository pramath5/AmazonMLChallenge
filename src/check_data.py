import pandas as pd


def load_data():
    train_s1=pd.read_csv(
        "D:/Projects/AmazonMLChallenge/dataset/train/train_source1.tsv",
        sep="\t"
    )

    train_s2=pd.read_csv(
        "D:/Projects/AmazonMLChallenge/dataset/train/train_source2.tsv",
        sep="\t"
    )

    train_s3=pd.read_csv(
        "D:/Projects/AmazonMLChallenge/dataset/train/train_source3.tsv",
        sep="\t"
    )

    ground_truth=pd.read_csv(
        "D:/Projects/AmazonMLChallenge/dataset/train/train_ground_truth.tsv",
        sep="\t"
    )
    return train_s1,train_s2,train_s3,ground_truth


def inspect_data(df,name):
    print("\n",name)

    # print("\nColumns:")
    # print(df.columns.tolist())

    # print("\nFirst 5 rows:")
    # print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())


def main():
    train_s1, train_s2, train_s3, ground_truth=load_data()

    inspect_data(train_s1,"SOURCE 1")
    inspect_data(train_s2,"SOURCE 2")
    inspect_data(train_s3,"SOURCE 3")
    inspect_data(ground_truth,"GROUND TRUTH")
    print("\n\nDATASET SIZES")

    print("Source 1:", len(train_s1))
    print("Source 2:", len(train_s2))
    print("Source 3:", len(train_s3))
    print("Ground Truth:", len(ground_truth))
    print("\n\nMISSING VALUE PERCENTAGES")

    for name, df in [
        ("Source 1", train_s1),
        ("Source 2", train_s2),
        ("Source 3", train_s3),
    ]:
        print(f"\n{name}")
        missing = df.isnull().sum()
        percentage = (missing / len(df)) * 100
        print(
            pd.DataFrame({
                "missing": missing,
                "percentage": percentage.round(2)
            })
        )

    print("\n\nCOUNTRY DISTRIBUTION")
    for name, df in [
        ("Source 1", train_s1),
        ("Source 2", train_s2),
        ("Source 3", train_s3),
    ]:
        print(f"\n{name}")
        print(df["country"].value_counts())

    print("\n\nGROUND TRUTH MATCH DISTRIBUTION")
    match_counts = ground_truth["matched_entity_ids"].fillna("").apply(
        lambda x: 0 if x == "" else len(x.split(","))
    )

    print(match_counts.describe())

    print("\nNumber of matches per S1:")
    print(match_counts.value_counts().sort_index())

    singleton_count = (match_counts == 0).sum()

    print("\nSingleton / no-match S1 entities:", singleton_count)
    print(
        "Percentage:",
        round(singleton_count / len(ground_truth) * 100, 2),
        "%"
    )
    print("\n\nMATCH SOURCE DISTRIBUTION")

    s2_matches = 0
    s3_matches = 0

    for value in ground_truth["matched_entity_ids"].dropna():
        ids = value.split(",")

        for entity_id in ids:
            if entity_id.startswith("S2-"):
                s2_matches += 1
            elif entity_id.startswith("S3-"):
                s3_matches += 1

    print("S2 matches:", s2_matches)
    print("S3 matches:", s3_matches)

if __name__=="__main__":
    main() 