import os
import json
import joblib
import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

HF_DATASET_REPO = os.getenv("HF_DATASET_REPO")
HF_MODEL_REPO = os.getenv("HF_MODEL_REPO")

TARGET_COL = "Engine Condition"


def evaluate(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
    }


def main():
    print("Loading processed train/test from Hugging Face...")
    train_df = load_dataset(HF_DATASET_REPO, data_files="processed/train_data.csv", split="train").to_pandas()
    test_df = load_dataset(HF_DATASET_REPO, data_files="processed/test_data.csv", split="train").to_pandas()

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]

    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    model_configs = {
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {"max_depth": [3, 5, 10], "min_samples_split": [2, 10]}
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "max_depth": [5, 10]}
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(random_state=42),
            "params": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
        },
        "GradientBoosting": {
            "model": GradientBoostingClassifier(random_state=42),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, eval_metric="logloss", use_label_encoder=False),
            "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
        }
    }

    results = []

    print("Running hyperparameter tuning + experiment tracking...")
    for model_name, cfg in model_configs.items():
        for params in ParameterGrid(cfg["params"]):
            model = cfg["model"].set_params(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            metrics = evaluate(y_test, preds)

            results.append({
                "model": model_name,
                "params": params,
                **metrics
            })

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)

    os.makedirs("artifacts", exist_ok=True)

    results_df.to_csv("artifacts/experiment_results.csv", index=False)

    best_row = results_df.iloc[0]
    best_model_name = best_row["model"]
    best_params = best_row["params"]

    print("Best model:", best_model_name)
    print("Best params:", best_params)

    best_model = model_configs[best_model_name]["model"].set_params(**best_params)
    best_model.fit(X_train, y_train)

    final_preds = best_model.predict(X_test)
    final_metrics = evaluate(y_test, final_preds)

    joblib.dump(best_model, "artifacts/best_model.joblib")

    with open("artifacts/final_metrics.json", "w") as f:
        json.dump({
            "best_model": best_model_name,
            "best_params": best_params,
            "test_metrics": final_metrics
        }, f, indent=2)

    print("Uploading best model to Hugging Face Model Hub...")
    api = HfApi(token=os.getenv("HF_TOKEN"))

    api.create_repo(repo_id=HF_MODEL_REPO, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj="artifacts/best_model.joblib",
        path_in_repo="best_model.joblib",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message=f"Upload best model: {best_model_name}"
    )

    api.upload_file(
        path_or_fileobj="artifacts/final_metrics.json",
        path_in_repo="final_metrics.json",
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        commit_message="Upload final metrics"
    )

    print("✅ Model training + registration complete.")


if __name__ == "__main__":
    main()
