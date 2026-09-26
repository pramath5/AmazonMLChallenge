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

def main():

    s1, s2, s3, ground_truth = load_data()

    # Create fast ID lookups
    s1_lookup = s1.set_index("entity_id")
    s2_lookup = s2.set_index("entity_id")
    s3_lookup = s3.set_index("entity_id")

    # Take the first few S1 entities that have matches
    examples = ground_truth[
        ground_truth["matched_entity_ids"].notna()
    ].head(10)

    for _, row in examples.iterrows():

        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"].split(",")

        print("\n" + "=" * 100)

        s1_record = s1_lookup.loc[s1_id]

        print("SOURCE 1")
        print("ID:", s1_id)
        print("Name:", s1_record["business_name"])
        print("Address:", s1_record["business_address"])
        print("Country:", s1_record["country"])

        print("\nMATCHES")

        for match_id in matched_ids:

            if match_id.startswith("S2-"):
                record = s2_lookup.loc[match_id]
                source = "S2"

            else:
                record = s3_lookup.loc[match_id]
                source = "S3"

            print("\n", source, match_id)
            print("Name:", record["business_name"])
            print("Address:", record["business_address"])
            print("Country:", record["country"])


if __name__ == "__main__":
    main()