import os
from huggingface_hub import HfApi

HF_SPACE_REPO = os.getenv("HF_SPACE_REPO")

def main():
    api = HfApi(token=os.getenv("HF_TOKEN"))

    # Create Hugging Face Space as DOCKER (not streamlit)
    api.create_repo(
        repo_id=HF_SPACE_REPO,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True
    )

    # Upload deployment folder (Dockerfile + app + requirements)
    api.upload_folder(
        folder_path="deployment",
        repo_id=HF_SPACE_REPO,
        repo_type="space",
        commit_message="Streamlit app"
    )

    print("✅ Hugging Face Space deployment complete.")

if __name__ == "__main__":
    main()
