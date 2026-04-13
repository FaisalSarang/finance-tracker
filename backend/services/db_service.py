"""Database query helpers for transactions and budgets.

All queries use asyncpg parameterized placeholders ($1, $2, ...).
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from backend.database import execute, fetch, fetchrow


# ── Transactions ──

async def get_transactions(
    category: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    sort_by: str = "date_desc",
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
) -> list[dict]:
    """Fetch transactions with optional filters."""

    conditions = []
    params = []
    idx = 1

    if category:
        conditions.append(f"category = ${idx}")
        params.append(category)
        idx += 1
    if start_date:
        conditions.append(f"date >= ${idx}")
        params.append(start_date)
        idx += 1
    if end_date:
        conditions.append(f"date <= ${idx}")
        params.append(end_date)
        idx += 1
    if search:
        conditions.append(f"LOWER(description) LIKE ${idx}")
        params.append(f"%{search.lower()}%")
        idx += 1

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    sort_map = {
        "date_desc": "date DESC, id DESC",
        "date_asc": "date ASC, id ASC",
        "amount_desc": "ABS(amount) DESC",
        "amount_asc": "ABS(amount) ASC",
    }
    order = sort_map.get(sort_by, "date DESC, id DESC")

    params.extend([limit, offset])
    sql = f"""
        SELECT id, transaction_id, date, description, amount,
               transaction_type, category, merchant, confidence, created_at
        FROM transactions {where}
        ORDER BY {order}
        LIMIT ${idx} OFFSET ${idx + 1}
    """
    return await fetch(sql, *params)


async def create_transaction(data: dict) -> dict | None:
    """Insert a single transaction. Returns the row or None if duplicate."""
    sql = """
        INSERT INTO transactions (transaction_id, date, description, amount,
                                  transaction_type, category, merchant, confidence)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (transaction_id) DO NOTHING
        RETURNING id, transaction_id, date, description, amount,
                  transaction_type, category, merchant, confidence, created_at
    """
    return await fetchrow(
        sql,
        data["transaction_id"],
        data["date"],
        data["description"],
        float(data["amount"]),
        data["transaction_type"],
        data.get("category"),
        data.get("merchant", data["description"]),
        data.get("confidence"),
    )


async def update_transaction(txn_id: int, data: dict) -> dict | None:
    """Update a transaction by id."""
    sets = []
    params = []
    idx = 1
    for key in ("category", "merchant", "description", "amount", "date", "transaction_type"):
        if key in data and data[key] is not None:
            sets.append(f"{key} = ${idx}")
            params.append(data[key] if key != "amount" else float(data[key]))
            idx += 1

    if not sets:
        return None

    params.append(txn_id)
    sql = f"""
        UPDATE transactions SET {', '.join(sets)}
        WHERE id = ${idx}
        RETURNING id, transaction_id, date, description, amount,
                  transaction_type, category, merchant, confidence, created_at
    """
    return await fetchrow(sql, *params)


async def delete_transaction(txn_id: int) -> bool:
    result = await execute("DELETE FROM transactions WHERE id = $1", txn_id)
    return result == "DELETE 1"


async def bulk_insert(results: list[dict]) -> tuple[int, int]:
    """Bulk insert transactions. Returns (inserted, skipped)."""
    inserted = 0
    for r in results:
        row = await create_transaction(r)
        if row is not None:
            inserted += 1
    return inserted, len(results) - inserted


# ── Stats ──

async def get_stats(
    category: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict:
    conditions = []
    params = []
    idx = 1
    if category:
        conditions.append(f"category = ${idx}")
        params.append(category)
        idx += 1
    if start_date:
        conditions.append(f"date >= ${idx}")
        params.append(start_date)
        idx += 1
    if end_date:
        conditions.append(f"date <= ${idx}")
        params.append(end_date)
        idx += 1
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    sql = f"""
        SELECT
            COUNT(*) as total_transactions,
            COALESCE(ABS(SUM(CASE WHEN transaction_type = 'debit' THEN amount ELSE 0 END)), 0) as total_spent,
            COALESCE(SUM(CASE WHEN transaction_type = 'credit' THEN amount ELSE 0 END), 0) as total_income,
            COUNT(DISTINCT category) as category_count,
            MIN(date) as earliest_date,
            MAX(date) as latest_date
        FROM transactions {where}
    """
    row = await fetchrow(sql, *params)
    if not row:
        return {}
    return {
        **row,
        "net_balance": float(row["total_income"]) - float(row["total_spent"]),
    }


async def get_category_breakdown(
    start_date: date | None = None,
    end_date: date | None = None,
) -> list[dict]:
    conditions = ["transaction_type = 'debit'"]
    params = []
    idx = 1
    if start_date:
        conditions.append(f"date >= ${idx}")
        params.append(start_date)
        idx += 1
    if end_date:
        conditions.append(f"date <= ${idx}")
        params.append(end_date)
        idx += 1
    where = f"WHERE {' AND '.join(conditions)}"

    rows = await fetch(f"""
        SELECT category, COUNT(*) as count, ABS(SUM(amount)) as total,
               ABS(AVG(amount)) as average
        FROM transactions {where}
        GROUP BY category ORDER BY total DESC
    """, *params)

    grand_total = sum(float(r["total"]) for r in rows) or 1
    return [
        {**r, "percentage": round(float(r["total"]) / grand_total * 100, 1)}
        for r in rows
    ]


async def get_monthly_trends(
    start_date: date | None = None,
    end_date: date | None = None,
    category: str | None = None,
) -> list[dict]:
    conditions = ["transaction_type = 'debit'"]
    params = []
    idx = 1
    if start_date:
        conditions.append(f"date >= ${idx}")
        params.append(start_date)
        idx += 1
    if end_date:
        conditions.append(f"date <= ${idx}")
        params.append(end_date)
        idx += 1
    if category:
        conditions.append(f"category = ${idx}")
        params.append(category)
        idx += 1
    where = f"WHERE {' AND '.join(conditions)}"

    return await fetch(f"""
        SELECT TO_CHAR(date, 'YYYY-MM') as month, category,
               ABS(SUM(amount)) as total, COUNT(*) as count
        FROM transactions {where}
        GROUP BY TO_CHAR(date, 'YYYY-MM'), category
        ORDER BY month
    """, *params)


# ── Budgets ──

async def ensure_budgets_table():
    """Create the budgets table if it doesn't exist."""
    await execute("""
        CREATE TABLE IF NOT EXISTS budgets (
            id SERIAL PRIMARY KEY,
            category VARCHAR(50) NOT NULL,
            monthly_limit DECIMAL(10,2) NOT NULL,
            month VARCHAR(7) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(category, month)
        )
    """)


async def get_budgets(month: str) -> list[dict]:
    """Get budgets for a month with spent amounts calculated from transactions."""
    budgets = await fetch(
        "SELECT id, category, monthly_limit, month, created_at FROM budgets WHERE month = $1",
        month,
    )

    result = []
    for b in budgets:
        spent_rows = await fetch("""
            SELECT COALESCE(ABS(SUM(amount)), 0) as spent
            FROM transactions
            WHERE category = $1 AND transaction_type = 'debit'
              AND TO_CHAR(date, 'YYYY-MM') = $2
        """, b["category"], month)
        spent = float(spent_rows[0]["spent"]) if spent_rows else 0
        limit_val = float(b["monthly_limit"])
        result.append({
            "id": b["id"],
            "category": b["category"],
            "monthly_limit": limit_val,
            "month": b["month"],
            "spent": spent,
            "remaining": max(0, limit_val - spent),
            "percentage": round((spent / limit_val * 100) if limit_val > 0 else 0, 1),
        })
    return result


async def create_budget(category: str, monthly_limit: Decimal, month: str) -> dict | None:
    return await fetchrow("""
        INSERT INTO budgets (category, monthly_limit, month)
        VALUES ($1, $2, $3)
        ON CONFLICT (category, month) DO UPDATE SET monthly_limit = EXCLUDED.monthly_limit
        RETURNING id, category, monthly_limit, month, created_at
    """, category, float(monthly_limit), month)


async def update_budget(budget_id: int, monthly_limit: Decimal) -> dict | None:
    return await fetchrow("""
        UPDATE budgets SET monthly_limit = $1 WHERE id = $2
        RETURNING id, category, monthly_limit, month, created_at
    """, float(monthly_limit), budget_id)
