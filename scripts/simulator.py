"""
CS5100 Final Project - Transaction Simulator
Samples from seed_transactions.csv and scales up to a full training dataset
with realistic variations: personas, seasonal patterns, weekend boosts,
description mutations, and amount jitter.

Usage:
    python scripts/simulator.py --records 100000 --output data/training_data.csv
"""

import argparse
import csv
import random
import string
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED_FILE = PROJECT_ROOT / "data" / "seed_transactions.csv"

# ---------- Persona spending profiles ----------

PERSONAS = {
    "student": {
        "weight": 0.40,
        "category_bias": {
            "Food & Dining": 0.28,
            "Groceries": 0.08,
            "Transportation": 0.10,
            "Shopping": 0.16,
            "Entertainment": 0.22,
            "Health & Pharmacy": 0.04,
            "Utilities": 0.06,
            "Income": 0.04,
            "Other": 0.02,
        },
    },
    "professional": {
        "weight": 0.35,
        "category_bias": {
            "Food & Dining": 0.18,
            "Groceries": 0.14,
            "Transportation": 0.14,
            "Shopping": 0.14,
            "Entertainment": 0.10,
            "Health & Pharmacy": 0.10,
            "Utilities": 0.10,
            "Income": 0.06,
            "Other": 0.04,
        },
    },
    "family": {
        "weight": 0.25,
        "category_bias": {
            "Food & Dining": 0.14,
            "Groceries": 0.22,
            "Transportation": 0.14,
            "Shopping": 0.14,
            "Entertainment": 0.06,
            "Health & Pharmacy": 0.10,
            "Utilities": 0.10,
            "Income": 0.06,
            "Other": 0.04,
        },
    },
}

# ---------- Seasonal multipliers (index 0 = January) ----------

SEASONAL = {
    "Food & Dining":     [0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.1, 1.0, 1.0, 1.0, 1.1, 1.3],
    "Groceries":         [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.5],
    "Transportation":    [1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.3, 1.2, 1.0, 1.0, 1.0, 0.9],
    "Shopping":          [0.8, 0.8, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1, 1.0, 1.0, 1.8, 2.0],
    "Entertainment":     [1.0, 1.0, 1.0, 1.0, 1.0, 1.2, 1.3, 1.2, 1.0, 1.0, 1.0, 1.1],
    "Health & Pharmacy": [1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.1],
    "Utilities":         [1.3, 1.2, 1.0, 0.9, 0.9, 1.0, 1.2, 1.3, 1.1, 0.9, 1.0, 1.2],
    "Income":            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "Other":             [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
}

WEEKEND_BOOST = {
    "Food & Dining": 1.4,
    "Entertainment": 1.3,
    "Shopping": 1.2,
}

# ---------- Load seeds ----------


def load_seeds(seed_file: Path) -> dict[str, list[dict]]:
    """Load seed transactions grouped by category."""
    seeds_by_category = {}

    with open(seed_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row["category"]
            if category not in seeds_by_category:
                seeds_by_category[category] = []
            seeds_by_category[category].append({
                "description": row["description"],
                "amount_min": float(row["amount_min"]),
                "amount_max": float(row["amount_max"]),
            })

    return seeds_by_category


# ---------- Description mutation ----------


def mutate_description(description: str) -> str:
    """Apply random mutations to a seed description to create variation."""
    mutated = description

    roll = random.random()

    if roll < 0.25:
        # swap store/order number with a new random one
        for i, char in enumerate(mutated):
            if char.isdigit():
                start = i
                while i < len(mutated) and (mutated[i].isdigit() or mutated[i] in ".-"):
                    i += 1
                old_num = mutated[start:i]
                new_num = "".join(random.choices(string.digits, k=len(old_num)))
                mutated = mutated[:start] + new_num + mutated[i:]
                break

    elif roll < 0.40:
        # append a random location
        locations = [
            "SAN JOSE", "SANTA CLARA", "SUNNYVALE", "CUPERTINO", "MOUNTAIN VIEW",
            "PALO ALTO", "MILPITAS", "FREMONT", "SAN FRANCISCO", "OAKLAND",
            "CAMPBELL", "LOS GATOS", "SARATOGA", "GILROY", "REDWOOD CITY",
            "MENLO PARK", "BERKELEY", "HAYWARD", "DUBLIN", "PLEASANTON",
        ]
        if not any(mutated.endswith(loc) for loc in locations):
            mutated = mutated.rstrip() + " " + random.choice(locations)

    elif roll < 0.50:
        # change capitalization
        choice = random.random()
        if choice < 0.33:
            mutated = mutated.upper()
        elif choice < 0.66:
            mutated = mutated.title()

    elif roll < 0.55:
        # introduce a minor typo (swap two adjacent characters)
        if len(mutated) > 4:
            pos = random.randint(1, len(mutated) - 3)
            mutated = mutated[:pos] + mutated[pos + 1] + mutated[pos] + mutated[pos + 2:]

    # else (45%): keep description as-is

    return mutated.strip()[:80]


# ---------- Transaction generation ----------


def generate_transaction_id() -> str:
    return f"TXN-{uuid.uuid4().hex[:12].upper()}"


def pick_persona() -> str:
    r = random.random()
    cumulative = 0
    for name, data in PERSONAS.items():
        cumulative += data["weight"]
        if r <= cumulative:
            return name
    return "student"


def pick_category(persona: str, month: int, is_weekend: bool) -> str:
    bias = PERSONAS[persona]["category_bias"]
    categories = list(bias.keys())
    weights = list(bias.values())

    adjusted = []
    for cat, w in zip(categories, weights):
        seasonal = SEASONAL.get(cat, [1.0] * 12)[month - 1]
        weekend = WEEKEND_BOOST.get(cat, 1.0) if is_weekend else 1.0
        adjusted.append(w * seasonal * weekend)

    total = sum(adjusted)
    adjusted = [w / total for w in adjusted]

    return random.choices(categories, weights=adjusted, k=1)[0]


def generate_amount(seed: dict, month: int, category: str) -> float:
    base = random.uniform(seed["amount_min"], seed["amount_max"])
    seasonal = SEASONAL.get(category, [1.0] * 12)[month - 1]
    jitter = random.uniform(0.90, 1.10)
    return round(base * seasonal * jitter, 2)


def generate_date(start: datetime, end: datetime) -> datetime:
    total_days = (end - start).days
    offset = random.randint(0, total_days)
    return start + timedelta(days=offset)


# ---------- Main simulator ----------


def simulate(seeds_by_category, num_records, start_date, end_date):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    transactions = []
    skipped = 0

    for _ in range(num_records):
        persona = pick_persona()
        txn_date = generate_date(start, end)
        month = txn_date.month
        is_weekend = txn_date.weekday() >= 5

        category = pick_category(persona, month, is_weekend)

        if category not in seeds_by_category or not seeds_by_category[category]:
            skipped += 1
            continue

        seed = random.choice(seeds_by_category[category])

        description = mutate_description(seed["description"])
        amount = generate_amount(seed, month, category)
        transaction_type = "credit" if category == "Income" else "debit"

        if transaction_type == "debit":
            amount = -amount

        transactions.append({
            "transaction_id": generate_transaction_id(),
            "date": txn_date.strftime("%Y-%m-%d"),
            "description": description,
            "amount": amount,
            "transaction_type": transaction_type,
            "category": category,
        })

    transactions.sort(key=lambda t: t["date"])

    if skipped > 0:
        print(f"  Skipped {skipped} transactions (no seeds for category)")

    return transactions


def main():
    parser = argparse.ArgumentParser(description="Simulate bank transactions from seeds")
    parser.add_argument("--records", type=int, default=100000,
                        help="Number of records to generate (default: 100000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: data/training_data.csv)")
    parser.add_argument("--start-date", type=str, default="2023-01-01",
                        help="Start date (default: 2023-01-01)")
    parser.add_argument("--end-date", type=str, default="2025-12-31",
                        help="End date (default: 2025-12-31)")

    args = parser.parse_args()

    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "training_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading seeds from {SEED_FILE}...")
    seeds_by_category = load_seeds(SEED_FILE)

    total_seeds = sum(len(v) for v in seeds_by_category.values())
    print(f"Loaded {total_seeds} seeds across {len(seeds_by_category)} categories\n")

    for cat, seeds in seeds_by_category.items():
        print(f"  {cat}: {len(seeds)} seeds")

    print(f"\nGenerating {args.records:,} transactions from {args.start_date} to {args.end_date}...")
    transactions = simulate(seeds_by_category, args.records, args.start_date, args.end_date)

    fieldnames = ["transaction_id", "date", "description", "amount", "transaction_type", "category"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)

    print(f"\nDone! Wrote {len(transactions):,} transactions to {output_path}")

    cat_counts = Counter(t["category"] for t in transactions)
    print("\n--- Category Distribution ---")
    for cat, count in cat_counts.most_common():
        pct = count / len(transactions) * 100
        print(f"  {cat}: {count:,} ({pct:.1f}%)")

    dates = [t["date"] for t in transactions]
    print(f"\nDate range: {min(dates)} to {max(dates)}")

    unique_desc = len(set(t["description"] for t in transactions))
    print(f"Unique descriptions: {unique_desc:,} out of {len(transactions):,}")


if __name__ == "__main__":
    main()