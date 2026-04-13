"""Build spending context string for the chatbot.

Ported from scripts/chatbot.py get_spending_context() — async version.
Now includes budget data so the AI can reference spending limits.
"""

from backend.database import fetch
from backend.services.db_service import get_budgets


async def get_spending_context() -> str:
    """Query DB and build a formatted spending summary for the LLM."""

    try:
        summary = await fetch("""
            SELECT category, COUNT(*) as count, ABS(SUM(amount)) as total,
                   ABS(AVG(amount)) as avg_amount
            FROM transactions WHERE transaction_type = 'debit'
            GROUP BY category ORDER BY total DESC
        """)

        income = await fetch("""
            SELECT SUM(amount) as total, COUNT(*) as count
            FROM transactions WHERE transaction_type = 'credit'
        """)

        recent = await fetch("""
            SELECT date, description, category, amount, transaction_type
            FROM transactions ORDER BY date DESC LIMIT 10
        """)

        monthly = await fetch("""
            SELECT TO_CHAR(date, 'YYYY-MM') as month, ABS(SUM(amount)) as total,
                   COUNT(*) as count
            FROM transactions WHERE transaction_type = 'debit'
            GROUP BY TO_CHAR(date, 'YYYY-MM') ORDER BY month
        """)

        # Get budgets for all months that have them
        budget_months = await fetch("""
            SELECT DISTINCT month FROM budgets ORDER BY month DESC
        """)

        parts = []

        if summary:
            lines = ["SPENDING BY CATEGORY:"]
            total_spent = 0
            for r in summary:
                t = float(r["total"])
                total_spent += t
                lines.append(
                    f"  {r['category']}: ${t:.2f} ({r['count']} txns, "
                    f"avg ${float(r['avg_amount']):.2f})"
                )
            lines.append(f"  TOTAL: ${total_spent:.2f}")
            parts.append("\n".join(lines))

        if income and income[0]["total"]:
            parts.append(
                f"INCOME: ${float(income[0]['total']):.2f} ({income[0]['count']} deposits)"
            )

        if recent:
            lines = ["RECENT TRANSACTIONS:"]
            for r in recent:
                amt = float(r["amount"])
                sign = "+" if r["transaction_type"] == "credit" else "-"
                lines.append(
                    f"  {r['date']} | {sign}${abs(amt):.2f} | "
                    f"{r['category']} | {r['description'][:35]}"
                )
            parts.append("\n".join(lines))

        if monthly:
            lines = ["MONTHLY SPENDING:"]
            for r in monthly:
                lines.append(f"  {r['month']}: ${float(r['total']):.2f} ({r['count']} txns)")
            parts.append("\n".join(lines))

        # Add budget information
        if budget_months:
            lines = ["BUDGETS:"]
            for bm in budget_months:
                month_str = bm["month"]
                budgets = await get_budgets(month_str)
                if budgets:
                    lines.append(f"  Month: {month_str}")
                    for b in budgets:
                        limit_val = b["monthly_limit"]
                        spent = b["spent"]
                        remaining = b["remaining"]
                        pct = b["percentage"]
                        status = "OVER BUDGET" if pct > 100 else f"{pct:.0f}% used"
                        lines.append(
                            f"    {b['category']}: limit ${limit_val:.2f}, "
                            f"spent ${spent:.2f}, remaining ${remaining:.2f} ({status})"
                        )
            if len(lines) > 1:
                parts.append("\n".join(lines))

        return "\n\n".join(parts) if parts else "No transaction data available."

    except Exception as e:
        return f"Error fetching data: {e}"
