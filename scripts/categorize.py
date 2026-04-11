"""
CS5100 Final Project - Transaction Categorizer (Local Model)
Reads a CSV of bank transactions, categorizes them using the trained
Logistic Regression model, and writes results to PostgreSQL and CSV.

Usage:
    1. Ensure models are trained (run scripts/train.py first)
    2. Ensure .env has DB credentials
    3. python scripts/categorize.py
    4. python scripts/categorize.py --input data/my_transactions.csv
"""

import argparse
import csv
import os
import time
from collections import Counter
from pathlib import Path

import joblib
import psycopg2
from dotenv import load_dotenv

# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

MODEL_DIR = PROJECT_ROOT / "models" / "saved"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Database config ----------

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "dbname": os.environ.get("DB_NAME", "postgres"),
    "user": os.environ.get("DB_USER", "finance"),
    "password": os.environ.get("DB_PASSWORD", "finance123"),
    "port": os.environ.get("DB_PORT", "5432"),
}


def load_model(tag: str = None):
    """Load the trained model, vectorizer, and label names."""
    suffix = f"_{tag}" if tag else ""
    print(f"Loading model{f' ({tag})' if tag else ''}...")
    vectorizer = joblib.load(MODEL_DIR / f"tfidf_vectorizer{suffix}.pkl")
    model = joblib.load(MODEL_DIR / f"logreg_model{suffix}.pkl")
    label_names = joblib.load(MODEL_DIR / f"label_names{suffix}.pkl")
    print(f"  Model loaded. Categories: {label_names}")
    return vectorizer, model, label_names


def categorize(descriptions: list[str], vectorizer, model, label_names) -> list[dict]:
    """Categorize all descriptions using the trained model."""
    features = vectorizer.transform(descriptions)

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    results = []
    for i, (pred_idx, proba) in enumerate(zip(predictions, probabilities)):
        label = label_names[pred_idx]
        confidence = float(proba[pred_idx])
        results.append({
            "category": label,
            "confidence": round(confidence, 4),
        })

    return results


def save_to_csv(results: list[dict], output_file: Path):
    """Write categorized transactions to a CSV file."""
    fieldnames = ["transaction_id", "date", "description", "amount",
                  "transaction_type", "category", "confidence"]
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"  CSV: {output_file}")


def save_to_postgres(results: list[dict]):
    """Insert categorized transactions into PostgreSQL, skipping duplicates."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    insert_query = """
        INSERT INTO transactions (transaction_id, date, description, amount,
                                  transaction_type, category, merchant, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (transaction_id) DO NOTHING
    """

    inserted = 0
    for r in results:
        cur.execute(insert_query, (
            r["transaction_id"],
            r["date"],
            r["description"],
            float(r["amount"]),
            r["transaction_type"],
            r["category"],
            r.get("merchant", r["description"]),
            float(r["confidence"]),
        ))
        if cur.rowcount > 0:
            inserted += 1

    conn.commit()
    cur.close()
    conn.close()

    print(f"  PostgreSQL: {inserted} new, {len(results) - inserted} skipped")


def main():
    parser = argparse.ArgumentParser(description="Categorize transactions")
    parser.add_argument("--input", type=str, default=None,
                        help="Input CSV (default: data/synthetic_transactions.csv)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV (default: output/categorized_transactions.csv)")
    parser.add_argument("--no-db", action="store_true",
                        help="Skip PostgreSQL, only write CSV")
    parser.add_argument("--tag", type=str, default=None,
                        help="Model tag to load (e.g. '2m', '100k')")
    args = parser.parse_args()

    input_file = Path(args.input) if args.input else PROJECT_ROOT / "data" / "synthetic_transactions.csv"
    output_file = Path(args.output) if args.output else OUTPUT_DIR / "categorized_transactions.csv"

    # load model
    vectorizer, model, label_names = load_model(args.tag)

    # load transactions
    with open(input_file, newline="") as f:
        reader = csv.DictReader(f)
        transactions = list(reader)

    print(f"\nLoaded {len(transactions):,} transactions from {input_file.name}")

    # categorize
    descriptions = [txn["description"] for txn in transactions]

    t0 = time.time()
    categorized = categorize(descriptions, vectorizer, model, label_names)
    elapsed = time.time() - t0

    print(f"Categorized {len(categorized):,} transactions in {elapsed:.3f}s "
          f"({elapsed / len(categorized) * 1000:.4f}ms per transaction)")

    # merge results
    results = []
    for txn, cat in zip(transactions, categorized):
        results.append({
            "transaction_id": txn.get("transaction_id", ""),
            "date": txn["date"],
            "description": txn["description"],
            "amount": txn["amount"],
            "transaction_type": txn["transaction_type"],
            "category": cat["category"],
            "confidence": cat["confidence"],
        })

    # print predictions
    for r in results[:10]:
        print(f"  {r['description'][:45]:<45} → {r['category']:<20} ({r['confidence']:.4f})")
    if len(results) > 10:
        print(f"  ... and {len(results) - 10} more")

    # save
    print(f"\nSaving results...")
    save_to_csv(results, output_file)

    if not args.no_db:
        try:
            save_to_postgres(results)
        except Exception as e:
            print(f"  PostgreSQL error: {e}")
            print("  Results saved to CSV only.")

    # summary
    cat_counts = Counter(r["category"] for r in results)
    print(f"\n--- Category Breakdown ---")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")

    print("\nDone!")


if __name__ == "__main__":
    main()