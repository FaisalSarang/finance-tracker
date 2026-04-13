# AI-Powered Personal Finance Tracker

**CS5100 — Foundations of Artificial Intelligence | Final Project**

An intelligent personal finance application that uses trained ML classifiers to automatically categorize bank transactions, provides interactive spending analytics, and includes an AI chatbot for natural-language financial queries. The project compares three AI approaches: TF-IDF + Logistic Regression, XGBoost, and LLM prompt engineering (Ollama).

**Team:** Faisal Riyaz Sarang, Juan Franco Boeta, Tridev Prabhu

---

## Key Results

| Model | Accuracy | F1 (Macro) | Inference (ms/txn) |
|---|---|---|---|
| Logistic Regression | **99.74%** | **0.9977** | **0.0001** |
| XGBoost | 99.10% | 0.9925 | 0.0043 |
| Ollama (Llama 3.1 8B) | 83.08% | 0.8319 | 578.96 |

Evaluated on the same 19,416-record test set. The trained Logistic Regression classifier outperforms a state-of-the-art LLM while being 5.8 million times faster at inference.

---

## Application Architecture

```
                   ┌─────────────────────────────────────────────┐
                   │           Next.js 16 Frontend               │
                   │  Dashboard · Transactions · Analytics       │
                   │  Budgets · Add Transaction · AI Chat        │
                   └──────────────────┬──────────────────────────┘
                                      │  REST API
                   ┌──────────────────▼──────────────────────────┐
                   │           FastAPI Backend                    │
                   │  ┌─────────┐ ┌──────────┐ ┌─────────────┐  │
                   │  │ ML Svc  │ │ DB Svc   │ │ Ollama Svc  │  │
                   │  │ TF-IDF  │ │ asyncpg  │ │ llama3.1:8b │  │
                   │  └────┬────┘ └────┬─────┘ └──────┬──────┘  │
                   └───────┼──────────┼───────────────┼──────────┘
                           │          │               │
                    ┌──────▼──┐  ┌────▼─────┐  ┌─────▼──────┐
                    │ .pkl    │  │PostgreSQL│  │  Ollama    │
                    │ models  │  │          │  │  (local)   │
                    └─────────┘  └──────────┘  └────────────┘
```

---

## Project Structure

```
finance-tracker/
├── .env.example                      # Environment variable template
├── .gitignore
├── requirements.txt                  # All Python dependencies
├── README.md
│
├── backend/                          # FastAPI REST API
│   ├── main.py                       # App entry point, lifespan, CORS
│   ├── config.py                     # Settings from .env
│   ├── database.py                   # asyncpg connection pool
│   ├── requirements.txt              # Backend-only dependencies
│   ├── models/
│   │   └── schemas.py                # Pydantic request/response models
│   ├── routes/
│   │   ├── transactions.py           # CRUD /api/transactions
│   │   ├── upload.py                 # CSV upload with auto-categorization
│   │   ├── categorize.py             # Single-description ML categorize
│   │   ├── stats.py                  # /api/stats, /api/analytics/*
│   │   ├── budgets.py                # /api/budgets CRUD
│   │   └── chat.py                   # /api/chat (Ollama)
│   └── services/
│       ├── ml_service.py             # TF-IDF + LogReg model loading
│       ├── db_service.py             # All database queries
│       ├── ollama_service.py         # Ollama LLM client
│       └── context_builder.py        # Builds spending context for chatbot
│
├── frontend/                         # Next.js 16 + TailwindCSS v4
│   ├── package.json
│   ├── src/
│   │   ├── app/                      # App Router pages
│   │   │   ├── page.tsx              # Dashboard
│   │   │   ├── layout.tsx            # Root layout (sidebar, slicers)
│   │   │   ├── globals.css           # Obsidian Mint design system
│   │   │   ├── transactions/         # Transaction list with search/sort
│   │   │   ├── add/                  # Manual entry + CSV drag-drop upload
│   │   │   ├── analytics/            # Category donut + monthly trends
│   │   │   ├── budgets/              # Budget management
│   │   │   └── chat/                 # Full-page AI chatbot
│   │   ├── components/
│   │   │   ├── ui/                   # Card, Badge, ProgressBar, Skeleton
│   │   │   ├── layout/               # Sidebar, MobileNav, SlicerBar
│   │   │   ├── dashboard/            # StatCards, SpendingChart, etc.
│   │   │   ├── transactions/         # TransactionTable
│   │   │   ├── analytics/            # CategoryDonut, MonthlyTrends
│   │   │   ├── budgets/              # BudgetCard, BudgetForm
│   │   │   └── chat/                 # ChatPanel, FloatingWidget
│   │   └── lib/
│   │       ├── api.ts                # Centralized API client
│   │       ├── types.ts              # TypeScript interfaces
│   │       ├── constants.ts          # Categories, colors, nav items
│   │       ├── utils.ts              # Formatting helpers
│   │       └── FilterContext.tsx      # Global month/category slicer state
│   └── ...
│
├── scripts/                          # ML pipeline scripts
│   ├── generate_seeds.py             # Groq API → seed descriptions
│   ├── simulator.py                  # Scales seeds → training data
│   ├── train.py                      # Trains XGBoost + LogReg
│   ├── evaluate.py                   # Three-way model comparison
│   ├── categorize.py                 # Production categorization pipeline
│   ├── chatbot.py                    # Standalone Ollama chatbot (CLI)
│   └── test_models.py               # Interactive model testing
│
├── data/
│   ├── seed_transactions.csv         # 1,252 LLM-generated seed descriptions
│   ├── synthetic_transactions.csv    # 25 test transactions
│   └── training_data.csv             # 100K simulated training records
│
├── models/
│   └── saved/                        # Trained .pkl files (git-ignored)
│
└── output/
    ├── 100k/                         # Evaluation results (100K dataset)
    │   ├── evaluation_results.json
    │   └── evaluation_charts/        # Comparison charts (PNG)
    └── 2M/                           # Evaluation results (2M dataset)
        ├── evaluation_results.json
        └── evaluation_charts/
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker Desktop (for PostgreSQL)
- Ollama (for AI chatbot)

### 1. Clone and install Python dependencies

```bash
git clone https://github.com/YOUR_USERNAME/finance-tracker.git
cd finance-tracker
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Create `.env` file

Copy the template and fill in your values:

```bash
cp .env.example .env
```

Get a free Groq API key at [console.groq.com](https://console.groq.com) (only needed for seed generation).

### 3. Start PostgreSQL

Make sure Docker Desktop is running, then:

```bash
docker run -d \
  --name finance-db \
  -e POSTGRES_USER=finance \
  -e POSTGRES_PASSWORD=finance123 \
  -e POSTGRES_DB=postgres \
  -p 5432:5432 \
  postgres:16
```

Create the table using any SQL client (connect to `localhost:5432`):

```sql
CREATE TABLE IF NOT EXISTS transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) UNIQUE NOT NULL,
    date DATE NOT NULL,
    description TEXT NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    transaction_type VARCHAR(20),
    category VARCHAR(50),
    merchant VARCHAR(100),
    confidence DECIMAL(3,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS budgets (
    id SERIAL PRIMARY KEY,
    month VARCHAR(7) NOT NULL,
    category VARCHAR(50) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    UNIQUE(month, category)
);
```

### 4. Install Ollama

Download from [ollama.com/download](https://ollama.com/download), then:

```bash
ollama pull llama3.1:8b
```

### 5. Start the backend

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

API available at `http://localhost:8000` — Swagger docs at `http://localhost:8000/docs`

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

App available at `http://localhost:3000`

---

## ML Pipeline

These scripts are used to train and evaluate the ML models. The trained models are then used by the backend API for real-time transaction categorization.

### Step 1: Generate seed transactions

Uses Groq API (Llama 3.3 70B) to create 1,252 diverse, realistic bank transaction descriptions across 8 categories.

```bash
python scripts/generate_seeds.py
```

### Step 2: Simulate training data

Samples from seeds and applies mutations (store number swaps, location suffixes, capitalization changes, typos) to generate large-scale training data.

```bash
python scripts/simulator.py --records 100000
python scripts/simulator.py --records 2000000 --output data/large_training_data.csv
```

### Step 3: Train models

Trains XGBoost and Logistic Regression with TF-IDF features (unigrams + bigrams, max 10K features).

```bash
python scripts/train.py --tag 100k
python scripts/train.py --data data/large_training_data.csv --tag 2m
```

### Step 4: Evaluate models

Three-way comparison: XGBoost vs Logistic Regression vs Ollama. All evaluated on the same test set.

```bash
python scripts/evaluate.py
```

### Step 5: Categorize new transactions

Production pipeline — categorizes CSV transactions using the trained Logistic Regression model and inserts into PostgreSQL.

```bash
python scripts/categorize.py
python scripts/categorize.py --input data/my_transactions.csv
python scripts/categorize.py --no-db    # skip database insert
```

---

## Web Application Features

### Dashboard
- Summary stats (total spent, transaction count, top category)
- Spending trend area chart
- Recent transactions list
- Budget progress overview

### Transactions
- Searchable, sortable transaction table
- Pagination
- Global month/year and category filters

### Add Transaction
- Manual entry form with auto-categorization (ML model predicts category on blur)
- CSV drag-and-drop upload — accepts any bank CSV format (fuzzy header detection, multi-format date parsing, split debit/credit column support)

### Analytics
- Category breakdown donut chart
- Monthly spending trends line chart
- Detailed category-by-category table

### Budgets
- Set monthly budget limits per category
- Visual progress bars with over-budget warnings
- Filtered by global month slicer

### AI Chatbot
- Conversational assistant powered by Ollama (Llama 3.1 8B)
- Has access to your real transaction and budget data
- Ask questions like "How much did I spend on food?", "Am I over budget?", "Give me tips to save money"
- Floating widget accessible from any page + dedicated full-page chat

---

## Data Pipeline

### Categories (8)

Food & Dining · Groceries · Transportation · Shopping · Entertainment · Health & Pharmacy · Utilities · Income

### Seed Generation
- 1,252 seeds generated via Groq API (Llama 3.3 70B)
- Realistic messy bank descriptions with category labels and amount ranges

### Simulation
- Three personas (student, professional, family) with different spending biases
- Seasonal multipliers, weekend boosts, description mutations
- 37,905 unique descriptions from 1,252 seeds (100K dataset)
- 425,344 unique descriptions (2M dataset)

### Feature Engineering
- TF-IDF vectorization: unigrams + bigrams, max 10K features, min df=3, sublinear TF
- Vocabulary: ~5,400 features

---

## Model Details

### Logistic Regression (Best Performer)
- Multinomial with L-BFGS solver
- Wins because transaction categorization is fundamentally a keyword-matching problem
- Training: ~1 second | Inference: 0.0001ms/txn

### XGBoost
- 200 estimators, max depth 6, learning rate 0.1
- Slightly lower accuracy due to overfitting on noise
- Training: ~8s (100K) | Inference: 0.004ms/txn

### Ollama / LLM Prompt Engineering
- Llama 3.1 8B, zero-shot classification
- Struggles with Groceries vs Shopping (0.73 F1), Utilities (0.78 F1)
- Inference: 579ms/txn (5.8M times slower than LogReg)

### Scaling Analysis

| Dataset | LogReg | XGBoost | Improvement |
|---|---|---|---|
| 100K | 99.54% | 98.73% | — |
| 2M | 99.74% | 99.10% | +0.20% / +0.37% |

20x more data yielded <0.4% improvement — data quality (diverse seeds) matters more than quantity.

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| Frontend | Next.js 16, TailwindCSS v4, Recharts | Web application UI |
| Backend | FastAPI, asyncpg | REST API server |
| ML Models | scikit-learn (TF-IDF + LogReg), XGBoost | Transaction categorization |
| AI Chatbot | Ollama / Llama 3.1 8B | Conversational financial assistant |
| Seed Generation | Groq API / Llama 3.3 70B | Generating training data seeds |
| Database | PostgreSQL 16 (Docker) | Transaction & budget storage |
| Data Fetching | SWR | Client-side caching & revalidation |

---

## License

This project was developed for CS5100 — Foundations of Artificial Intelligence at Northeastern University, Khoury College of Computer Sciences.
