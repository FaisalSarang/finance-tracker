"""
CS5100 Final Project - Seed Transaction Generator
Uses Groq/Llama to generate 1,000 diverse, realistic bank transaction
descriptions across all 9 categories. One API call per category (9 total).
These seeds are used by the simulator to scale up to a full training dataset.

Usage:
    1. Ensure .env has GROQ_API_KEY
    2. pip install python-dotenv requests
    3. python scripts/generate_seeds.py
"""

import csv
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import requests
from dotenv import load_dotenv

# load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Set GROQ_API_KEY in your .env file before running.")

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
MAX_RETRIES = 5

# output path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_FILE = DATA_DIR / "seed_transactions.csv"

# ---------- One call per category ----------

CATEGORIES = {
    "Food & Dining": {
        "count": 180,
        "hint": (
            "restaurants, fast food, delivery apps (Uber Eats, DoorDash, Grubhub), "
            "coffee shops (Starbucks, Philz, Peet's), boba tea, pizza, takeout, "
            "food trucks, bars, diners, bakeries, sushi, ramen, tacos, pho, "
            "sandwich shops, ice cream. Mix chains and local Bay Area spots."
        ),
    },
    "Groceries": {
        "count": 120,
        "hint": (
            "supermarkets (Safeway, Albertsons), wholesale clubs (Costco, Sam's Club), "
            "Trader Joe's, Whole Foods, Walmart grocery, ethnic stores (H Mart, 99 Ranch, "
            "Nijiya), farmers markets, Instacart, Amazon Fresh. Include Bay Area locations."
        ),
    },
    "Transportation": {
        "count": 120,
        "hint": (
            "gas stations (Shell, Chevron, Arco, 76), rideshare (Uber, Lyft), "
            "public transit (Caltrain, BART, VTA, Clipper), parking meters/garages, "
            "toll bridges (FasTrak), car washes, EV charging (ChargePoint, Tesla), "
            "auto repair, oil changes, DMV fees."
        ),
    },
    "Shopping": {
        "count": 130,
        "hint": (
            "online retail (Amazon, eBay, Etsy, Temu, Shein), electronics (Best Buy, "
            "Micro Center, Apple Store), clothing (Nike, Ross, Marshalls, TJ Maxx, Uniqlo, "
            "H&M, Zara), department stores (Target, Macy's, Nordstrom), home goods "
            "(IKEA, Home Depot, Lowe's), dollar stores, thrift stores, bookstores."
        ),
    },
    "Entertainment": {
        "count": 120,
        "hint": (
            "streaming (Netflix, Spotify, Hulu, Disney+, HBO Max, YouTube Premium, "
            "Apple TV+, Peacock, Paramount+), gaming (Steam, PlayStation, Xbox, Nintendo, "
            "Epic Games), movie theaters (AMC, Regal, Century), concerts, Ticketmaster, "
            "theme parks, arcades, bowling, escape rooms, museums, Kindle/Audible."
        ),
    },
    "Health & Pharmacy": {
        "count": 100,
        "hint": (
            "pharmacies (CVS, Walgreens, Rite Aid), gyms (24 Hour Fitness, Planet Fitness, "
            "Equinox, CrossFit, Bay Club), doctors/copays, dentists, optometrists, "
            "urgent care, Kaiser, mental health, GoodRx, vitamin shops (GNC), "
            "health insurance premiums."
        ),
    },
    "Utilities": {
        "count": 100,
        "hint": (
            "electric/gas (PG&E, ConEd, SCE), water (SJWC, EBMUD), internet (Xfinity, "
            "AT&T, Spectrum, Google Fiber), phone (T-Mobile, Verizon, Mint Mobile), "
            "cloud storage (iCloud, Google One, Dropbox), VPN (NordVPN, ExpressVPN), "
            "trash/recycling, sewer."
        ),
    },
    "Income": {
        "count": 80,
        "hint": (
            "payroll deposits (ADP, Gusto, direct deposit), freelance payments (PayPal, "
            "Stripe), Venmo/Zelle/CashApp received, tax refunds (IRS), cashback rewards, "
            "bank interest, stock dividends, merchant refund credits, scholarship deposits, "
            "rental income, side gig payments."
        ),
    },
    "Other": {
        "count": 50,
        "hint": (
            "ATM withdrawals, bank fees (overdraft, monthly maintenance), wire transfers, "
            "money orders, rent/mortgage payments, insurance premiums (auto, renters), "
            "charity donations, government fees, pet expenses (vet, Petco, PetSmart), "
            "laundry, dry cleaning, storage units, shipping (UPS, FedEx, USPS)."
        ),
    },
}

SYSTEM_PROMPT = """You are generating realistic bank transaction descriptions for a machine learning training dataset.
Generate descriptions that look EXACTLY like they appear on a real bank or credit card statement.

Rules:
- Use ALL CAPS or mixed case as real banks do
- Include store numbers, order numbers, location codes randomly
- Use real abbreviations: SQ *, TST*, POS DEBIT, ACH, DD, WF, etc.
- Add location suffixes sometimes: city names, state codes, zip fragments
- Include messy/truncated versions: STRBCKS instead of STARBUCKS, AMZN instead of AMAZON
- Mix in order numbers: #1234, *R4KL9, REF-8821
- Some should be cryptic and hard to categorize
- No two descriptions should be identical
- Each must be a single line, no commas within the description
- Include a realistic amount range (min and max as floats)

Return ONLY a JSON array where each element is:
{"description": "THE TRANSACTION TEXT", "amount_min": 5.00, "amount_max": 25.00}

No markdown, no backticks, no explanation. Return ONLY the JSON array."""


def call_groq_with_retry(category: str, count: int, hint: str) -> list[dict]:
    """Generate all seeds for a category in a single API call with retries."""
    user_prompt = (
        f"Generate exactly {count} unique bank transaction descriptions "
        f"for the category: {category}\n\n"
        f"Types of merchants/transactions to include: {hint}\n\n"
        f"Make them diverse - mix well-known chains with local businesses, "
        f"vary the formatting style, and include some that are tricky to categorize. "
        f"Include Bay Area / California specific merchants where appropriate.\n\n"
        f"Remember: return exactly {count} items in a JSON array."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.9,
        "max_tokens": 8192,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                GROQ_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                },
                json=payload,
            )
            response.raise_for_status()

            raw = response.json()["choices"][0]["message"]["content"]
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            return json.loads(cleaned)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) + random.uniform(0, 1)
                print(f"  Rate limited. Retrying in {wait:.1f}s...")
                time.sleep(wait)
            else:
                raise
        except json.JSONDecodeError:
            if attempt < MAX_RETRIES - 1:
                print(f"  JSON parse error. Retrying...")
                time.sleep(2)
            else:
                raise


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_seeds = []

    print("=== Generating Seed Transactions (9 API calls) ===\n")

    for i, (category, config) in enumerate(CATEGORIES.items(), 1):
        count = config["count"]
        hint = config["hint"]

        print(f"[{i}/9] {category} ({count} seeds)...", end=" ", flush=True)

        try:
            batch = call_groq_with_retry(category, count, hint)

            seeds = []
            for item in batch:
                seeds.append({
                    "description": item["description"],
                    "category": category,
                    "amount_min": float(item.get("amount_min", 5.00)),
                    "amount_max": float(item.get("amount_max", 50.00)),
                })

            all_seeds.extend(seeds)
            print(f"got {len(seeds)}")

        except Exception as e:
            print(f"ERROR: {e}")

        # pause between calls
        if i < 9:
            time.sleep(120)

    random.shuffle(all_seeds)

    fieldnames = ["description", "category", "amount_min", "amount_max"]
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_seeds)

    print(f"\n=== Done! Wrote {len(all_seeds)} seeds to {OUTPUT_FILE} ===")

    cat_counts = Counter(s["category"] for s in all_seeds)
    print("\n--- Category Distribution ---")
    for cat, count in cat_counts.most_common():
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()