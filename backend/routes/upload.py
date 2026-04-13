"""CSV bulk upload endpoint.

Accepts any reasonable CSV file, auto-detects columns, parses dates in
multiple formats, categorizes via ML, and inserts into the database.
"""

from __future__ import annotations

import csv
import io
import re
import time
import uuid
from datetime import date, datetime

from fastapi import APIRouter, UploadFile, File, HTTPException

from backend.models.schemas import UploadResponse
from backend.services import ml_service, db_service

router = APIRouter(tags=["upload"])


# ── Flexible column detection ──

# Each key is our internal field name; the value is a list of patterns
# that might appear as a CSV header (all compared lowercase, stripped).
HEADER_PATTERNS: dict[str, list[str]] = {
    "date": [
        "date", "transaction date", "trans date", "posted date",
        "posting date", "value date", "effective date", "txn date",
    ],
    "description": [
        "description", "memo", "transaction description", "details",
        "narrative", "particulars", "reference", "payee", "name",
        "merchant", "merchant name",
    ],
    "amount": [
        "amount", "sum", "value", "transaction amount", "debit amount",
        "credit amount", "debit", "credit",
    ],
    "transaction_type": [
        "type", "transaction type", "trans type", "txn type",
        "dr/cr", "debit/credit",
    ],
    "category": [
        "category", "tag", "label", "group",
    ],
    "transaction_id": [
        "transaction id", "transaction_id", "txn id", "reference number",
        "ref", "id", "check number",
    ],
}


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace, replace _ and - with space."""
    return re.sub(r"[\s_\-]+", " ", s.strip().lower())


def _detect_columns(fieldnames: list[str]) -> dict[str, str]:
    """Map CSV column names → our internal field names.

    Special handling: if separate debit/credit amount columns are found,
    both are mapped as '_debit_amount' / '_credit_amount' so the row
    parser can merge them.
    """
    mapping: dict[str, str] = {}           # csv_col → our_field
    used_fields: set[str] = set()

    # First pass: check for split debit / credit amount columns
    debit_col = None
    credit_col = None
    for raw_col in fieldnames:
        norm = _normalize(raw_col)
        if any(k in norm for k in ("debit amount", "debit amt", "withdrawals")):
            debit_col = raw_col
        elif any(k in norm for k in ("credit amount", "credit amt", "deposits")):
            credit_col = raw_col

    # If we found split columns, map them specially and mark amount as handled
    if debit_col and credit_col:
        mapping[debit_col] = "_debit_amount"
        mapping[credit_col] = "_credit_amount"
        used_fields.add("amount")

    for raw_col in fieldnames:
        if raw_col in mapping:
            continue
        norm = _normalize(raw_col)
        for our_field, patterns in HEADER_PATTERNS.items():
            if our_field in used_fields:
                continue
            if norm in patterns or any(p in norm for p in patterns):
                mapping[raw_col] = our_field
                used_fields.add(our_field)
                break

    return mapping


# ── Date parsing ──

DATE_FORMATS = [
    "%Y-%m-%d",        # 2025-01-03
    "%m/%d/%Y",        # 01/03/2025
    "%m/%d/%y",        # 01/03/25
    "%d/%m/%Y",        # 03/01/2025  (ambiguous, tried after US format)
    "%m-%d-%Y",        # 01-03-2025
    "%d-%m-%Y",        # 03-01-2025
    "%Y/%m/%d",        # 2025/01/03
    "%b %d, %Y",       # Jan 03, 2025
    "%B %d, %Y",       # January 03, 2025
    "%d %b %Y",        # 03 Jan 2025
    "%d %B %Y",        # 03 January 2025
]


def _parse_date(raw: str) -> date | None:
    """Try multiple date formats, return date object or None."""
    cleaned = raw.strip()
    if not cleaned:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    return None


# ── Amount parsing ──

def _parse_amount(raw: str) -> float | None:
    """Parse a currency string like '$1,234.56' or '(100.00)' into a float."""
    s = raw.strip()
    if not s:
        return None
    # Detect negative from parens: (100.00)
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    # Remove currency symbols and commas
    s = re.sub(r"[^\d.\-+]", "", s)
    if not s:
        return None
    try:
        val = float(s)
        return -val if neg else val
    except ValueError:
        return None


# ── Row processing ──

def _parse_csv(text: str) -> tuple[list[dict], dict[str, str]]:
    """Parse CSV text. Returns (rows, column_mapping)."""
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(status_code=400, detail="CSV has no headers")

    col_map = _detect_columns(list(reader.fieldnames))

    mapped_fields = set(col_map.values())

    if "date" not in mapped_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Could not detect a date column. Found headers: {reader.fieldnames}",
        )
    if "description" not in mapped_fields:
        raise HTTPException(
            status_code=400,
            detail=f"Could not detect a description column. Found headers: {reader.fieldnames}",
        )
    has_amount = "amount" in mapped_fields or (
        "_debit_amount" in mapped_fields and "_credit_amount" in mapped_fields
    )
    if not has_amount:
        raise HTTPException(
            status_code=400,
            detail=f"Could not detect an amount column. Found headers: {reader.fieldnames}",
        )

    rows: list[dict] = []
    has_split_amounts = "_debit_amount" in col_map.values()

    for row in reader:
        mapped: dict[str, str] = {}
        for csv_col, our_field in col_map.items():
            val = row.get(csv_col, "").strip()
            if val:
                mapped[our_field] = val

        # Merge split debit/credit columns into a single amount
        if has_split_amounts and "amount" not in mapped:
            debit_val = mapped.pop("_debit_amount", "")
            credit_val = mapped.pop("_credit_amount", "")
            if debit_val:
                mapped["amount"] = f"-{debit_val}" if not debit_val.startswith("-") else debit_val
                mapped["transaction_type"] = "debit"
            elif credit_val:
                mapped["amount"] = credit_val
                mapped["transaction_type"] = "credit"
        else:
            # Clean up internal keys if present
            mapped.pop("_debit_amount", None)
            mapped.pop("_credit_amount", None)

        if mapped.get("date") and mapped.get("description") and mapped.get("amount"):
            rows.append(mapped)

    return rows, col_map


def _clean_row(row: dict) -> dict | None:
    """Normalize a parsed row into the shape create_transaction expects.

    Returns None if the row can't be parsed (bad date or amount).
    """
    # Parse date
    parsed_date = _parse_date(row["date"])
    if parsed_date is None:
        return None

    # Parse amount
    amount = _parse_amount(row["amount"])
    if amount is None:
        return None

    # Infer transaction type
    txn_type_raw = row.get("transaction_type", "").strip().lower()
    if txn_type_raw in ("credit", "deposit", "cr"):
        txn_type = "credit"
    elif txn_type_raw in ("debit", "withdrawal", "purchase", "dr"):
        txn_type = "debit"
    else:
        txn_type = "credit" if amount > 0 else "debit"

    # Use existing transaction_id or generate one
    txn_id = row.get("transaction_id", "").strip()
    if not txn_id:
        txn_id = f"CSV-{uuid.uuid4().hex[:12].upper()}"

    return {
        "transaction_id": txn_id,
        "date": parsed_date,                  # datetime.date object for asyncpg
        "description": row["description"],
        "amount": abs(amount),
        "transaction_type": txn_type,
        "category": row.get("category") or None,
        "merchant": row.get("description", "")[:50],
    }


# ── Endpoint ──

@router.post("/upload", response_model=UploadResponse)
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    start = time.time()

    content = await file.read()
    # Try UTF-8 with BOM first, fall back to latin-1
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    raw_rows, col_map = _parse_csv(text)
    if not raw_rows:
        raise HTTPException(status_code=400, detail="No valid rows found in CSV")

    # Clean all rows (skip unparseable ones)
    cleaned: list[dict] = []
    skipped_parse = 0
    for r in raw_rows:
        row = _clean_row(r)
        if row is not None:
            cleaned.append(row)
        else:
            skipped_parse += 1

    if not cleaned:
        raise HTTPException(status_code=400, detail="No rows could be parsed. Check date and amount formats.")

    # Auto-categorize rows without a category
    uncategorized_indices = [i for i, r in enumerate(cleaned) if not r.get("category")]
    if uncategorized_indices:
        descriptions = [cleaned[i]["description"] for i in uncategorized_indices]
        predictions = ml_service.categorize(descriptions)
        for i, pred in zip(uncategorized_indices, predictions):
            cleaned[i]["category"] = pred["category"]
            cleaned[i]["confidence"] = pred["confidence"]

    # Bulk insert into database
    inserted, dup_skipped = await db_service.bulk_insert(cleaned)

    # Build category breakdown for the response
    from collections import Counter
    cat_counts = Counter(r["category"] for r in cleaned)
    cat_amounts: dict[str, float] = {}
    for r in cleaned:
        cat_amounts[r["category"]] = cat_amounts.get(r["category"], 0) + float(r["amount"])

    grand_total = sum(cat_amounts.values()) or 1
    categories = []
    for cat, count in cat_counts.most_common():
        amt = cat_amounts[cat]
        categories.append({
            "category": cat,
            "count": count,
            "total": round(amt, 2),
            "average": round(amt / count, 2),
            "percentage": round(amt / grand_total * 100, 1),
        })

    return UploadResponse(
        total_parsed=len(raw_rows),
        categorized=len(uncategorized_indices),
        inserted=inserted,
        duplicates_skipped=dup_skipped,
        time_seconds=round(time.time() - start, 2),
        categories=categories,
    )
