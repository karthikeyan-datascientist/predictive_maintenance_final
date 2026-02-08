import os
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
TARGET_COL = "Engine Condition"

FEATURE_COLS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp"
]


def main():
    print("Loading raw dataset from Hugging Face...")
    ds = load_dataset(HF_DATASET_REPO, split="train")
    df = ds.to_pandas()

    print("Selecting relevant columns...")
    df = df[FEATURE_COLS + [TARGET_COL]].copy()

    print("Basic cleaning...")
    df = df.drop_duplicates()
    df = df.dropna()

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    print("Splitting into train and test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test

    os.makedirs("artifacts", exist_ok=True)

    train_path = "artifacts/train_data.csv"
    test_path = "artifacts/test_data.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Uploading processed datasets to Hugging Face...")
    api = HfApi(token=os.getenv("HF_TOKEN"))

    api.upload_file(
        path_or_fileobj=train_path,
        path_in_repo="processed/train_data.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message="Upload processed train dataset"
    )

    api.upload_file(
        path_or_fileobj=test_path,
        path_in_repo="processed/test_data.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message="Upload processed test dataset"
    )

    print("✅ Data preparation complete.")


if __name__ == "__main__":
    main()
