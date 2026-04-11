"""
CS5100 Final Project - Conversational Finance Assistant
A Streamlit chatbot that lets you ask natural language questions about
your spending. Queries PostgreSQL for data, sends context to Ollama
for intelligent responses.

Usage:
    pip install streamlit psycopg2-binary requests
    streamlit run app/chatbot.py
"""

import json
import os
from datetime import datetime
from pathlib import Path

import psycopg2
import requests
import streamlit as st
from dotenv import load_dotenv

# ---------- Config ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DB_CONFIG = {
    "host": os.environ.get("DB_HOST", "localhost"),
    "dbname": os.environ.get("DB_NAME", "postgres"),
    "user": os.environ.get("DB_USER", "finance"),
    "password": os.environ.get("DB_PASSWORD", "finance123"),
    "port": os.environ.get("DB_PORT", "5432"),
}

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """You are a helpful financial assistant. The user will ask questions about their
spending and transactions. You will receive their actual transaction data from a database
as context. Use this data to give specific, accurate answers.

Rules:
- Always reference actual numbers from the data provided
- Be concise and conversational
- If the data doesn't contain enough information to answer, say so
- Format currency as $X.XX
- When listing transactions, keep it brief — show top 5 unless asked for more
- If asked for advice, base it on their actual spending patterns
- Do not make up transactions or amounts that aren't in the data"""


# ---------- Database queries ----------


def get_db_connection():
    """Get a PostgreSQL connection."""
    return psycopg2.connect(**DB_CONFIG)


def query_db(sql: str, params: tuple = None) -> list[dict]:
    """Execute a query and return results as list of dicts."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(sql, params or ())
    columns = [desc[0] for desc in cur.description]
    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def get_spending_summary() -> str:
    """Get overall spending summary."""
    rows = query_db("""
        SELECT category,
               COUNT(*) as count,
               ABS(SUM(amount)) as total,
               ABS(AVG(amount)) as avg_amount,
               MIN(date) as earliest,
               MAX(date) as latest
        FROM transactions
        WHERE transaction_type = 'debit'
        GROUP BY category
        ORDER BY total DESC
    """)

    if not rows:
        return "No transactions found in the database."

    lines = ["SPENDING SUMMARY BY CATEGORY:"]
    grand_total = 0
    for r in rows:
        total = float(r["total"])
        grand_total += total
        lines.append(
            f"  {r['category']}: ${total:.2f} "
            f"({r['count']} transactions, avg ${float(r['avg_amount']):.2f})"
        )
    lines.append(f"\n  TOTAL SPENDING: ${grand_total:.2f}")

    return "\n".join(lines)


def get_income_summary() -> str:
    """Get income summary."""
    rows = query_db("""
        SELECT COUNT(*) as count,
               SUM(amount) as total,
               AVG(amount) as avg_amount
        FROM transactions
        WHERE transaction_type = 'credit'
    """)

    if not rows or rows[0]["total"] is None:
        return "No income transactions found."

    r = rows[0]
    return (
        f"INCOME SUMMARY:\n"
        f"  Total income: ${float(r['total']):.2f}\n"
        f"  Transactions: {r['count']}\n"
        f"  Average: ${float(r['avg_amount']):.2f}"
    )


def get_recent_transactions(limit: int = 10) -> str:
    """Get most recent transactions."""
    rows = query_db("""
        SELECT date, description, category, amount, transaction_type
        FROM transactions
        ORDER BY date DESC
        LIMIT %s
    """, (limit,))

    if not rows:
        return "No transactions found."

    lines = [f"RECENT TRANSACTIONS (last {limit}):"]
    for r in rows:
        amount = float(r["amount"])
        sign = "+" if r["transaction_type"] == "credit" else "-"
        lines.append(
            f"  {r['date']} | {sign}${abs(amount):.2f} | "
            f"{r['category']} | {r['description'][:40]}"
        )

    return "\n".join(lines)


def get_category_transactions(category: str, limit: int = 10) -> str:
    """Get transactions for a specific category."""
    rows = query_db("""
        SELECT date, description, amount, merchant
        FROM transactions
        WHERE LOWER(category) = LOWER(%s)
        ORDER BY date DESC
        LIMIT %s
    """, (category, limit))

    if not rows:
        return f"No transactions found for category: {category}"

    total = sum(abs(float(r["amount"])) for r in rows)
    lines = [f"TRANSACTIONS — {category} (showing {len(rows)}, total: ${total:.2f}):"]
    for r in rows:
        lines.append(
            f"  {r['date']} | ${abs(float(r['amount'])):.2f} | "
            f"{r['merchant'] or r['description'][:40]}"
        )

    return "\n".join(lines)


def get_monthly_spending() -> str:
    """Get spending by month."""
    rows = query_db("""
        SELECT TO_CHAR(date, 'YYYY-MM') as month,
               ABS(SUM(amount)) as total,
               COUNT(*) as count
        FROM transactions
        WHERE transaction_type = 'debit'
        GROUP BY TO_CHAR(date, 'YYYY-MM')
        ORDER BY month
    """)

    if not rows:
        return "No spending data found."

    lines = ["MONTHLY SPENDING:"]
    for r in rows:
        lines.append(f"  {r['month']}: ${float(r['total']):.2f} ({r['count']} transactions)")

    return "\n".join(lines)


def get_top_merchants(limit: int = 10) -> str:
    """Get top merchants by spending."""
    rows = query_db("""
        SELECT merchant,
               COUNT(*) as count,
               ABS(SUM(amount)) as total
        FROM transactions
        WHERE transaction_type = 'debit' AND merchant IS NOT NULL
        GROUP BY merchant
        ORDER BY total DESC
        LIMIT %s
    """, (limit,))

    if not rows:
        return "No merchant data found."

    lines = [f"TOP {limit} MERCHANTS BY SPENDING:"]
    for r in rows:
        lines.append(f"  {r['merchant']}: ${float(r['total']):.2f} ({r['count']} transactions)")

    return "\n".join(lines)


def get_all_data_context(user_question: str) -> str:
    """Build a comprehensive data context based on the user's question."""
    context_parts = []

    question_lower = user_question.lower()

    # always include spending summary
    context_parts.append(get_spending_summary())

    # add relevant data based on keywords
    if any(word in question_lower for word in ["income", "earn", "salary", "deposit", "credit"]):
        context_parts.append(get_income_summary())

    if any(word in question_lower for word in ["recent", "latest", "last"]):
        context_parts.append(get_recent_transactions(15))

    if any(word in question_lower for word in ["month", "trend", "over time", "compare"]):
        context_parts.append(get_monthly_spending())

    if any(word in question_lower for word in ["merchant", "store", "where", "shop"]):
        context_parts.append(get_top_merchants())

    # check for specific category mentions
    categories = [
        "food", "dining", "groceries", "transportation", "shopping",
        "entertainment", "health", "pharmacy", "utilities", "income",
    ]
    for cat in categories:
        if cat in question_lower:
            # map to full category name
            cat_map = {
                "food": "Food & Dining", "dining": "Food & Dining",
                "groceries": "Groceries", "transportation": "Transportation",
                "shopping": "Shopping", "entertainment": "Entertainment",
                "health": "Health & Pharmacy", "pharmacy": "Health & Pharmacy",
                "utilities": "Utilities", "income": "Income",
            }
            full_cat = cat_map.get(cat)
            if full_cat:
                context_parts.append(get_category_transactions(full_cat, 15))

    # if no specific context matched, add general data
    if len(context_parts) == 1:
        context_parts.append(get_income_summary())
        context_parts.append(get_recent_transactions(10))
        context_parts.append(get_monthly_spending())

    return "\n\n".join(context_parts)


# ---------- Ollama ----------


def ask_ollama(user_question: str, data_context: str, chat_history: list[dict]) -> str:
    """Send question + data context to Ollama and get a response."""

    # build messages with history
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # add recent chat history (last 6 messages for context)
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    # add current question with data
    user_message = (
        f"Here is my transaction data:\n\n{data_context}\n\n"
        f"My question: {user_question}"
    )
    messages.append({"role": "user", "content": user_message})

    try:
        response = requests.post(
            OLLAMA_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": OLLAMA_MODEL,
                "messages": messages,
                "temperature": 0.3,
                "max_tokens": 1024,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.ConnectionError:
        return (
            "I can't connect to the AI model. Make sure Ollama is running:\n"
            "```\nollama serve\n```"
        )
    except Exception as e:
        return f"Error getting response: {e}"


# ---------- Streamlit UI ----------


def main():
    st.set_page_config(
        page_title="Finance Assistant",
        page_icon="💰",
        layout="wide",
    )

    st.title("💰 AI Finance Assistant")
    st.caption("Ask questions about your spending in plain English.")

    # sidebar with quick stats
    with st.sidebar:
        st.header("Quick Stats")

        try:
            summary = query_db("""
                SELECT
                    COUNT(*) as total_txns,
                    ABS(SUM(CASE WHEN transaction_type = 'debit' THEN amount ELSE 0 END)) as total_spent,
                    SUM(CASE WHEN transaction_type = 'credit' THEN amount ELSE 0 END) as total_income,
                    COUNT(DISTINCT category) as categories
                FROM transactions
            """)

            if summary and summary[0]["total_txns"] > 0:
                s = summary[0]
                st.metric("Total Transactions", f"{s['total_txns']:,}")
                st.metric("Total Spent", f"${float(s['total_spent']):,.2f}")
                st.metric("Total Income", f"${float(s['total_income']):,.2f}")
                net = float(s["total_income"]) - float(s["total_spent"])
                st.metric("Net Balance", f"${net:,.2f}", delta=f"${net:,.2f}")
            else:
                st.warning("No transactions in database. Run categorize.py first.")

        except Exception as e:
            st.error(f"Database error: {e}")

        st.divider()
        st.subheader("Sample Questions")
        sample_questions = [
            "How much did I spend on food?",
            "What are my top spending categories?",
            "Show me my recent transactions",
            "Where do I spend the most money?",
            "How does this month compare to last month?",
            "What subscriptions am I paying for?",
            "Give me tips to save money",
            "How much did I earn vs spend?",
        ]
        for q in sample_questions:
            if st.button(q, key=q, use_container_width=True):
                st.session_state.pending_question = q

    # chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # handle pending question from sidebar
    pending = st.session_state.pop("pending_question", None)

    # chat input
    user_input = st.chat_input("Ask about your finances...")
    question = pending or user_input

    if question:
        # show user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # get data context and response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing your transactions..."):
                data_context = get_all_data_context(question)
                response = ask_ollama(question, data_context, st.session_state.messages)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()