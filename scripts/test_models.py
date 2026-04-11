"""
CS5100 Final Project - Model Testing
Load saved models and test them on custom transaction descriptions.

Usage:
    python scripts/test_models.py
    python scripts/test_models.py --description "STARBUCKS #4421 SAN JOSE"
"""

import argparse
import time
from pathlib import Path

import joblib

# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

# ---------- Load models ----------

print("Loading models...")
vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
xgb_model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
lr_model = joblib.load(MODEL_DIR / "logreg_model.pkl")
label_names = joblib.load(MODEL_DIR / "label_names.pkl")
print("Models loaded.\n")


def predict(description: str):
    """Run both models on a single description and print results."""
    # transform text to TF-IDF
    features = vectorizer.transform([description])

    # XGBoost prediction
    start = time.time()
    xgb_pred = xgb_model.predict(features)[0]
    xgb_time = (time.time() - start) * 1000

    xgb_label = label_names[xgb_pred]

    # Logistic Regression prediction with probabilities
    start = time.time()
    lr_pred = lr_model.predict(features)[0]
    lr_proba = lr_model.predict_proba(features)[0]
    lr_time = (time.time() - start) * 1000

    lr_label = label_names[lr_pred]
    lr_confidence = lr_proba[lr_pred]

    # display results
    print(f"  Description:  {description}")
    print(f"  XGBoost:      {xgb_label} ({xgb_time:.4f}ms)")
    print(f"  LogReg:       {lr_label} (confidence: {lr_confidence:.4f}, {lr_time:.4f}ms)")

    # show top 3 categories from LogReg
    top3_idx = lr_proba.argsort()[-3:][::-1]
    print(f"  Top 3:")
    for idx in top3_idx:
        print(f"    {label_names[idx]}: {lr_proba[idx]:.4f}")

    # flag disagreement
    if xgb_label != lr_label:
        print(f"  ⚠ DISAGREEMENT: XGBoost says {xgb_label}, LogReg says {lr_label}")

    print()


def interactive_mode():
    """Run models interactively — type descriptions to test."""
    print("=" * 55)
    print("  Transaction Categorizer — Interactive Testing")
    print("  Type a transaction description and press Enter.")
    print("  Type 'quit' to exit.")
    print("=" * 55)
    print()

    while True:
        try:
            description = input("Transaction > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not description or description.lower() == "quit":
            print("Exiting.")
            break

        predict(description)


def main():
    parser = argparse.ArgumentParser(description="Test trained models")
    parser.add_argument("--description", type=str, default=None,
                        help="Single description to test")
    args = parser.parse_args()

    if args.description:
        predict(args.description)
    else:
        # run some sample tests first
        print("--- Sample Predictions ---\n")

        samples = [
            "STARBUCKS #4421 SAN JOSE",
            "AMZN MKTP US*3A1B7",
            "SHELL OIL 57442 SANTA CLARA",
            "NETFLIX.COM",
            "DIRECT DEPOSIT PAYROLL",
            "CVS/PHARMACY #8841",
            "COSTCO WHSE #0321",
            "UBER EATS ORDER #4821",
            "SQ *BOBA GUYS SF",
            "PG&E WEB ONLINE",
            "WALMART SC #1234",
            "TARGET.COM *7821",
            "VENMO PAYMENT RECEIVED",
            "ATM WITHDRAWAL #9912",
        ]

        for desc in samples:
            predict(desc)

        # then enter interactive mode
        print("\n--- Interactive Mode ---\n")
        interactive_mode()


if __name__ == "__main__":
    main()