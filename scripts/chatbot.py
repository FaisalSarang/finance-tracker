"""
CS5100 Final Project - AI Finance Tracker
Full Streamlit application with:
  - Landing page with navigation
  - CSV upload and AI categorization
  - Transaction history from PostgreSQL
  - Floating chatbot (Ollama)

Usage:
    streamlit run app/chatbot.py
"""

import csv
import io
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
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

MODEL_DIR = PROJECT_ROOT / "models" / "saved"
OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
OLLAMA_MODEL = "llama3.1:8b"

SYSTEM_PROMPT = """You are a helpful financial assistant. The user will ask questions about their
spending and transactions. You will receive their actual transaction data from a database
as context. Use this data to give specific, accurate answers.

Rules:
- Always reference actual numbers from the data provided
- Be concise and conversational — keep responses under 150 words
- Format currency as $X.XX
- When listing transactions, show top 5 unless asked for more
- If asked for advice, base it on their actual spending patterns
- Do not make up transactions or amounts that aren't in the data"""


# ---------- Page config ----------

st.set_page_config(
    page_title="AI Finance Tracker",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- Custom CSS ----------

st.markdown("""
<style>
    /* hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* global font */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* landing page styles */
    .landing-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 70vh;
        text-align: center;
    }

    .landing-title {
        font-size: 3.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
        letter-spacing: -1px;
    }

    .landing-subtitle {
        font-size: 1.15rem;
        color: #6b7280;
        margin-bottom: 3rem;
        max-width: 500px;
    }

    .button-row {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        flex-wrap: wrap;
    }

    .nav-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #e0e7ff;
        border-radius: 16px;
        padding: 2.5rem 2rem;
        width: 280px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
    }

    .nav-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 32px rgba(99, 102, 241, 0.15);
        border-color: #818cf8;
    }

    .nav-card-icon {
        font-size: 2.5rem;
        margin-bottom: 0.8rem;
    }

    .nav-card-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.4rem;
    }

    .nav-card-desc {
        font-size: 0.9rem;
        color: #6b7280;
    }

    /* page header */
    .page-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #eef2ff;
    }

    .page-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1a1a2e;
        margin: 0;
    }

    /* stat cards */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }

    .stat-card {
        flex: 1;
        min-width: 180px;
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #e0e7ff;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
    }

    .stat-label {
        font-size: 0.8rem;
        font-weight: 500;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-top: 0.2rem;
    }

    .stat-value.positive { color: #059669; }
    .stat-value.negative { color: #dc2626; }

    /* upload area */
    .upload-zone {
        border: 2px dashed #c7d2fe;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: #f8fafc;
        margin-bottom: 1.5rem;
    }

    /* parsing results */
    .parse-success {
        background: #ecfdf5;
        border: 1px solid #6ee7b7;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }

    .parse-error {
        background: #fef2f2;
        border: 1px solid #fca5a5;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* chatbot toggle */
    .chat-toggle {
        position: fixed;
        bottom: 24px;
        right: 24px;
        width: 56px;
        height: 56px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        font-size: 1.5rem;
        cursor: pointer;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
        z-index: 1000;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: transform 0.2s;
    }

    .chat-toggle:hover {
        transform: scale(1.1);
    }

    /* category badge */
    .cat-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
    }

    .cat-food { background: #f97316; }
    .cat-groceries { background: #22c55e; }
    .cat-transport { background: #3b82f6; }
    .cat-shopping { background: #ec4899; }
    .cat-entertainment { background: #8b5cf6; }
    .cat-health { background: #14b8a6; }
    .cat-utilities { background: #64748b; }
    .cat-income { background: #059669; }
    .cat-other { background: #9ca3af; }

    /* back button */
    .back-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: #6366f1;
        font-weight: 500;
        cursor: pointer;
        margin-bottom: 1rem;
        font-size: 0.95rem;
    }

    /* transaction table */
    .txn-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
    }

    .txn-table th {
        background: #f1f5f9;
        padding: 0.8rem 1rem;
        text-align: left;
        font-weight: 600;
        font-size: 0.85rem;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .txn-table td {
        padding: 0.7rem 1rem;
        border-bottom: 1px solid #f1f5f9;
        font-size: 0.9rem;
    }

    .amount-debit { color: #dc2626; font-weight: 600; }
    .amount-credit { color: #059669; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ---------- Database ----------


def get_db_connection():
    return psycopg2.connect(**DB_CONFIG)


def query_db(sql: str, params: tuple = None) -> list[dict]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(sql, params or ())
    columns = [desc[0] for desc in cur.description]
    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    cur.close()
    conn.close()
    return rows


def insert_transactions(results: list[dict]):
    """Insert categorized transactions into PostgreSQL."""
    conn = get_db_connection()
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
    return inserted


def get_transaction_count() -> int:
    try:
        rows = query_db("SELECT COUNT(*) as cnt FROM transactions")
        return rows[0]["cnt"] if rows else 0
    except Exception:
        return 0


# ---------- Model ----------


@st.cache_resource
def load_model():
    """Load trained model (cached across reruns)."""
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    model = joblib.load(MODEL_DIR / "logreg_model.pkl")
    label_names = joblib.load(MODEL_DIR / "label_names.pkl")
    return vectorizer, model, label_names


def categorize_transactions(descriptions: list[str]) -> list[dict]:
    """Categorize descriptions using trained model."""
    vectorizer, model, label_names = load_model()
    features = vectorizer.transform(descriptions)
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)

    results = []
    for pred_idx, proba in zip(predictions, probabilities):
        results.append({
            "category": label_names[pred_idx],
            "confidence": round(float(proba[pred_idx]), 4),
        })
    return results


# ---------- Ollama ----------


def get_spending_context() -> str:
    """Build spending context for the chatbot."""
    try:
        summary = query_db("""
            SELECT category, COUNT(*) as count, ABS(SUM(amount)) as total,
                   ABS(AVG(amount)) as avg_amount
            FROM transactions WHERE transaction_type = 'debit'
            GROUP BY category ORDER BY total DESC
        """)

        income = query_db("""
            SELECT SUM(amount) as total, COUNT(*) as count
            FROM transactions WHERE transaction_type = 'credit'
        """)

        recent = query_db("""
            SELECT date, description, category, amount, transaction_type
            FROM transactions ORDER BY date DESC LIMIT 10
        """)

        monthly = query_db("""
            SELECT TO_CHAR(date, 'YYYY-MM') as month, ABS(SUM(amount)) as total,
                   COUNT(*) as count
            FROM transactions WHERE transaction_type = 'debit'
            GROUP BY TO_CHAR(date, 'YYYY-MM') ORDER BY month
        """)

        parts = []

        if summary:
            lines = ["SPENDING BY CATEGORY:"]
            total_spent = 0
            for r in summary:
                t = float(r["total"])
                total_spent += t
                lines.append(f"  {r['category']}: ${t:.2f} ({r['count']} txns, avg ${float(r['avg_amount']):.2f})")
            lines.append(f"  TOTAL: ${total_spent:.2f}")
            parts.append("\n".join(lines))

        if income and income[0]["total"]:
            parts.append(f"INCOME: ${float(income[0]['total']):.2f} ({income[0]['count']} deposits)")

        if recent:
            lines = ["RECENT TRANSACTIONS:"]
            for r in recent:
                amt = float(r["amount"])
                sign = "+" if r["transaction_type"] == "credit" else "-"
                lines.append(f"  {r['date']} | {sign}${abs(amt):.2f} | {r['category']} | {r['description'][:35]}")
            parts.append("\n".join(lines))

        if monthly:
            lines = ["MONTHLY SPENDING:"]
            for r in monthly:
                lines.append(f"  {r['month']}: ${float(r['total']):.2f} ({r['count']} txns)")
            parts.append("\n".join(lines))

        return "\n\n".join(parts) if parts else "No transaction data available."

    except Exception as e:
        return f"Error fetching data: {e}"


def ask_ollama(question: str, chat_history: list[dict]) -> str:
    """Send question to Ollama with spending context."""
    context = get_spending_context()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"Transaction data:\n\n{context}\n\nQuestion: {question}"
    })

    try:
        response = requests.post(
            OLLAMA_URL,
            headers={"Content-Type": "application/json"},
            json={"model": OLLAMA_MODEL, "messages": messages, "temperature": 0.3, "max_tokens": 1024},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        return "Cannot connect to Ollama. Make sure it's running: `ollama serve`"
    except Exception as e:
        return f"Error: {e}"


# ---------- Category colors ----------

CAT_COLORS = {
    "Food & Dining": "#f97316",
    "Groceries": "#22c55e",
    "Transportation": "#3b82f6",
    "Shopping": "#ec4899",
    "Entertainment": "#8b5cf6",
    "Health & Pharmacy": "#14b8a6",
    "Utilities": "#64748b",
    "Income": "#059669",
    "Other": "#9ca3af",
}


# ---------- Pages ----------


def landing_page():
    """Main landing page with two navigation cards."""
    txn_count = get_transaction_count()

    st.markdown("""
    <div class="landing-container">
        <div class="landing-title">💰 AI Finance Tracker</div>
        <div class="landing-subtitle">
            Upload your bank statement and let AI categorize your spending instantly.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
                        border: 1px solid #e0e7ff; border-radius: 16px;
                        padding: 2.5rem 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">📄</div>
                <div style="font-size: 1.15rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.4rem;">
                    Add Bank Statement
                </div>
                <div style="font-size: 0.85rem; color: #6b7280;">
                    Upload a CSV and categorize transactions with AI
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Upload CSV →", key="nav_upload", use_container_width=True):
                st.session_state.page = "upload"
                st.rerun()

        with c2:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f8fafc 0%, #ecfdf5 100%);
                        border: 1px solid #a7f3d0; border-radius: 16px;
                        padding: 2.5rem 1.5rem; text-align: center;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">📊</div>
                <div style="font-size: 1.15rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.4rem;">
                    View Transaction History
                </div>
                <div style="font-size: 0.85rem; color: #6b7280;">
                    {txn_count:,} transactions in database
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("View History →", key="nav_history", use_container_width=True):
                st.session_state.page = "history"
                st.rerun()


def upload_page():
    """CSV upload and categorization page."""

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("""
    <div class="page-header">
        <h1>📄 Add Bank Statement</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("Upload a CSV file with columns: `transaction_id`, `date`, `description`, `amount`, `transaction_type`")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded_file is not None:
        # parse CSV
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.uploaded_df = df
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        # validate columns
        required = ["date", "description", "amount", "transaction_type"]
        missing = [c for c in required if c not in df.columns]

        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.markdown("Expected columns: `transaction_id`, `date`, `description`, `amount`, `transaction_type`")
            return

        # show preview
        st.markdown(f"""
        <div class="parse-success">
            ✅ Successfully parsed <strong>{len(df):,}</strong> transactions
            from <strong>{uploaded_file.name}</strong>
        </div>
        """, unsafe_allow_html=True)

        # preview table
        st.markdown("**Preview (first 10 rows):**")
        st.dataframe(df.head(10), use_container_width=True)

        # categorize button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🤖 Categorize with AI", use_container_width=True, type="primary"):
                with st.spinner("Categorizing transactions..."):
                    descriptions = df["description"].tolist()

                    t0 = time.time()
                    categories = categorize_transactions(descriptions)
                    elapsed = time.time() - t0

                    # merge results
                    results = []
                    for i, (_, row) in enumerate(df.iterrows()):
                        results.append({
                            "transaction_id": row.get("transaction_id", f"TXN-UPLOAD-{i+1:06d}"),
                            "date": row["date"],
                            "description": row["description"],
                            "amount": float(row["amount"]),
                            "transaction_type": row["transaction_type"],
                            "category": categories[i]["category"],
                            "confidence": categories[i]["confidence"],
                            "merchant": row["description"],
                        })

                    st.session_state.categorized = results
                    st.session_state.categorize_time = elapsed

                    st.success(f"Categorized {len(results):,} transactions in {elapsed:.3f}s")

        # show categorized results
        if "categorized" in st.session_state:
            results = st.session_state.categorized

            st.markdown("---")
            st.markdown("### Categorization Results")

            # category summary
            cat_counts = {}
            cat_totals = {}
            for r in results:
                cat = r["category"]
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
                cat_totals[cat] = cat_totals.get(cat, 0) + abs(r["amount"])

            cols = st.columns(min(len(cat_counts), 4))
            for i, (cat, count) in enumerate(sorted(cat_counts.items(), key=lambda x: -cat_totals[x[0]])):
                with cols[i % len(cols)]:
                    color = CAT_COLORS.get(cat, "#6b7280")
                    st.markdown(f"""
                    <div style="background: {color}15; border-left: 4px solid {color};
                                border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.8rem;">
                        <div style="font-weight: 600; color: {color};">{cat}</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #1a1a2e;">
                            ${cat_totals[cat]:,.2f}
                        </div>
                        <div style="font-size: 0.8rem; color: #6b7280;">{count} transactions</div>
                    </div>
                    """, unsafe_allow_html=True)

            # results table
            results_df = pd.DataFrame(results)
            display_df = results_df[["date", "description", "amount", "category", "confidence"]].copy()
            display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.2%}")
            st.dataframe(display_df, use_container_width=True)

            # save buttons
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("💾 Save to Database", use_container_width=True, type="primary"):
                    try:
                        inserted = insert_transactions(results)
                        st.success(f"Saved {inserted} new transactions to database ({len(results) - inserted} duplicates skipped)")
                    except Exception as e:
                        st.error(f"Database error: {e}")


def history_page():
    """Transaction history page with stats and table."""

    if st.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.markdown("""
    <div class="page-header">
        <h1>📊 Transaction History</h1>
    </div>
    """, unsafe_allow_html=True)

    try:
        # stat cards
        stats = query_db("""
            SELECT
                COUNT(*) as total_txns,
                ABS(SUM(CASE WHEN transaction_type = 'debit' THEN amount ELSE 0 END)) as total_spent,
                SUM(CASE WHEN transaction_type = 'credit' THEN amount ELSE 0 END) as total_income,
                COUNT(DISTINCT category) as categories,
                MIN(date) as earliest,
                MAX(date) as latest
            FROM transactions
        """)

        if not stats or stats[0]["total_txns"] == 0:
            st.info("No transactions in the database yet. Upload a bank statement first!")
            return

        s = stats[0]
        total_spent = float(s["total_spent"])
        total_income = float(s["total_income"])
        net = total_income - total_spent

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Transactions</div>
                <div class="stat-value">{s['total_txns']:,}</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Spent</div>
                <div class="stat-value negative">-${total_spent:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Total Income</div>
                <div class="stat-value positive">+${total_income:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            color_class = "positive" if net >= 0 else "negative"
            sign = "+" if net >= 0 else "-"
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">Net Balance</div>
                <div class="stat-value {color_class}">{sign}${abs(net):,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # spending by category
        st.markdown("### Spending by Category")

        cat_data = query_db("""
            SELECT category, COUNT(*) as count, ABS(SUM(amount)) as total
            FROM transactions WHERE transaction_type = 'debit'
            GROUP BY category ORDER BY total DESC
        """)

        if cat_data:
            cat_df = pd.DataFrame(cat_data)
            cat_df["total"] = cat_df["total"].astype(float)

            cols = st.columns(min(len(cat_data), 4))
            for i, row in enumerate(cat_data):
                with cols[i % len(cols)]:
                    color = CAT_COLORS.get(row["category"], "#6b7280")
                    pct = float(row["total"]) / total_spent * 100 if total_spent > 0 else 0
                    st.markdown(f"""
                    <div style="background: {color}15; border-left: 4px solid {color};
                                border-radius: 8px; padding: 0.8rem 1rem; margin-bottom: 0.8rem;">
                        <div style="font-weight: 600; color: {color};">{row['category']}</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: #1a1a2e;">
                            ${float(row['total']):,.2f}
                        </div>
                        <div style="font-size: 0.8rem; color: #6b7280;">
                            {row['count']} transactions · {pct:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # filters
        st.markdown("### All Transactions")

        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])

        with filter_col1:
            categories = ["All"] + [r["category"] for r in cat_data] if cat_data else ["All"]
            selected_cat = st.selectbox("Category", categories)

        with filter_col2:
            sort_options = {"Newest first": "date DESC", "Oldest first": "date ASC",
                           "Highest amount": "ABS(amount) DESC", "Lowest amount": "ABS(amount) ASC"}
            selected_sort = st.selectbox("Sort by", list(sort_options.keys()))

        with filter_col3:
            limit = st.selectbox("Show", [25, 50, 100, 250], index=0)

        # build query
        where_clause = ""
        params = []
        if selected_cat != "All":
            where_clause = "WHERE category = %s"
            params.append(selected_cat)

        order_clause = sort_options[selected_sort]

        txns = query_db(f"""
            SELECT date, description, category, amount, transaction_type, confidence
            FROM transactions {where_clause}
            ORDER BY {order_clause}
            LIMIT %s
        """, tuple(params + [limit]))

        if txns:
            txn_df = pd.DataFrame(txns)
            txn_df["amount"] = txn_df["amount"].apply(
                lambda x: f"+${abs(float(x)):,.2f}" if float(x) > 0 else f"-${abs(float(x)):,.2f}"
            )
            txn_df["confidence"] = txn_df["confidence"].apply(
                lambda x: f"{float(x):.0%}" if x else "N/A"
            )
            txn_df = txn_df.rename(columns={
                "date": "Date",
                "description": "Description",
                "category": "Category",
                "amount": "Amount",
                "transaction_type": "Type",
                "confidence": "AI Confidence",
            })

            st.dataframe(txn_df, use_container_width=True, hide_index=True)
        else:
            st.info("No transactions match the selected filters.")

    except Exception as e:
        st.error(f"Database error: {e}")
        st.info("Make sure PostgreSQL is running and the transactions table exists.")


def chatbot_widget():
    """Floating chatbot at the bottom of the page."""

    # only show on history and upload pages
    if st.session_state.get("page", "home") == "home":
        return

    st.markdown("---")

    with st.expander("💬 **AI Finance Assistant** — Ask about your spending", expanded=False):
        # chat history
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []

        # display messages
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # sample questions as pills
        if not st.session_state.chat_messages:
            st.markdown("**Try asking:**")
            sample_cols = st.columns(2)
            samples = [
                "How much did I spend on food?",
                "What are my top categories?",
                "Give me tips to save money",
                "Show my recent transactions",
            ]
            for i, q in enumerate(samples):
                with sample_cols[i % 2]:
                    if st.button(q, key=f"sample_{i}", use_container_width=True):
                        st.session_state.pending_chat = q
                        st.rerun()

        # handle pending question
        pending = st.session_state.pop("pending_chat", None)

        # chat input
        user_input = st.chat_input("Ask about your finances...")
        question = pending or user_input

        if question:
            st.session_state.chat_messages.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = ask_ollama(question, st.session_state.chat_messages)
                    st.markdown(response)

            st.session_state.chat_messages.append({"role": "assistant", "content": response})


# ---------- Main ----------


def main():
    # initialize page state
    if "page" not in st.session_state:
        st.session_state.page = "home"

    # route to pages
    if st.session_state.page == "home":
        landing_page()
    elif st.session_state.page == "upload":
        upload_page()
        chatbot_widget()
    elif st.session_state.page == "history":
        history_page()
        chatbot_widget()


if __name__ == "__main__":
    main()