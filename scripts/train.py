"""
CS5100 Final Project - Model Training
Trains XGBoost and Logistic Regression classifiers on transaction data
using TF-IDF features. Supports training on different dataset sizes
to compare the effect of data volume on accuracy.

Usage:
    python scripts/train.py                                    # default: training_data.csv (100K)
    python scripts/train.py --data data/scale_test.csv         # 2M dataset
    python scripts/train.py --tag 100k                         # custom tag for saved models
"""

import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import argparse

# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Train transaction classifiers")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training CSV (default: data/training_data.csv)")
    parser.add_argument("--tag", type=str, default=None,
                        help="Tag for saved model files (e.g. '100k', '2m')")
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else PROJECT_ROOT / "data" / "training_data.csv"
    tag = args.tag or data_path.stem  # e.g. "training_data" or "scale_test"

    # load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"  Records: {len(df):,}")
    print(f"  Categories: {df['category'].nunique()}")

    # encode labels
    label_names = sorted(df["category"].unique())
    label_to_idx = {label: idx for idx, label in enumerate(label_names)}
    df["label"] = df["category"].map(label_to_idx)

    print(f"\nLabel mapping:")
    for label, idx in label_to_idx.items():
        count = len(df[df["category"] == label])
        print(f"  {idx}: {label} ({count:,})")

    # train/test split
    X_text = df["description"]
    y = df["label"]

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train_text):,} | Test: {len(X_test_text):,}")

    # build TF-IDF features
    print("\nBuilding TF-IDF features...")
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=10000,
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    print(f"  Vocabulary: {len(vectorizer.vocabulary_):,} features")

    # train XGBoost
    print("\nTraining XGBoost...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=len(label_names),
        eval_metric="mlogloss",
        tree_method="hist",
        device="cpu",
        n_jobs=-1,
        random_state=42,
        verbosity=1,
    )

    t0 = time.time()
    xgb_model.fit(X_train_tfidf, y_train)
    xgb_time = time.time() - t0
    print(f"  Done in {xgb_time:.1f}s")

    # train Logistic Regression
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )

    t0 = time.time()
    lr_model.fit(X_train_tfidf, y_train)
    lr_time = time.time() - t0
    print(f"  Done in {lr_time:.1f}s")

    # save with tag
    print(f"\nSaving models (tag: {tag})...")
    joblib.dump(vectorizer, MODEL_DIR / f"tfidf_vectorizer_{tag}.pkl")
    joblib.dump(xgb_model, MODEL_DIR / f"xgboost_model_{tag}.pkl")
    joblib.dump(lr_model, MODEL_DIR / f"logreg_model_{tag}.pkl")
    joblib.dump(label_names, MODEL_DIR / f"label_names_{tag}.pkl")
    joblib.dump(label_to_idx, MODEL_DIR / f"label_to_idx_{tag}.pkl")

    # also save as default (no tag) for categorize.py to use
    joblib.dump(vectorizer, MODEL_DIR / "tfidf_vectorizer.pkl")
    joblib.dump(xgb_model, MODEL_DIR / "xgboost_model.pkl")
    joblib.dump(lr_model, MODEL_DIR / "logreg_model.pkl")
    joblib.dump(label_names, MODEL_DIR / "label_names.pkl")
    joblib.dump(label_to_idx, MODEL_DIR / "label_to_idx.pkl")

    print(f"  Saved to {MODEL_DIR}")

    # summary
    print(f"\n{'=' * 45}")
    print(f"  Training Summary — {tag}")
    print(f"{'=' * 45}")
    print(f"  Dataset:         {data_path.name}")
    print(f"  Records:         {len(df):,}")
    print(f"  Train/Test:      {len(X_train_text):,} / {len(X_test_text):,}")
    print(f"  Vocabulary:      {len(vectorizer.vocabulary_):,}")
    print(f"  XGBoost time:    {xgb_time:.1f}s")
    print(f"  LogReg time:     {lr_time:.1f}s")
    print(f"{'=' * 45}")
    print("\nRun evaluate.py to see results.")


if __name__ == "__main__":
    main()