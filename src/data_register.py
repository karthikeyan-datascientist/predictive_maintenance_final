import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")   
REPO_TYPE = "dataset"

# This is a LOCAL folder path (Drive path in Colab)
RAW_DATA_FOLDER = "data"


def main():
    api = HfApi(token=os.getenv("HF_TOKEN"))

    try:
        api.repo_info(repo_id=HF_DATASET_REPO, repo_type=REPO_TYPE)
        print(f"Dataset repo '{HF_DATASET_REPO}' already exists.")
    except RepositoryNotFoundError:
        print(f"Dataset repo '{HF_DATASET_REPO}' not found. Creating...")
        create_repo(
            repo_id=HF_DATASET_REPO,
            repo_type=REPO_TYPE,
            private=False,
            token=os.getenv("HF_TOKEN")
        )
        print(f"Dataset repo '{HF_DATASET_REPO}' created.")

    # Upload local folder "data/"
    print("Uploading raw dataset folder from local path:", RAW_DATA_FOLDER)

    api.upload_folder(
        folder_path=RAW_DATA_FOLDER,
        repo_id=HF_DATASET_REPO,
        repo_type=REPO_TYPE,
        commit_message="Upload raw dataset"
    )

    print("✅ Data registration complete.")


if __name__ == "__main__":
    main()
