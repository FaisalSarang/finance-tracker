# AI-Powered Personal Finance Tracker

**CS5100 — Foundations of Artificial Intelligence | Final Project**

An intelligent transaction categorization system that compares three AI approaches: trained ML classifiers, XGBoost, and LLM prompt engineering. Includes a conversational chatbot for querying spending data in natural language.

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

## Project Architecture

```
CSV Upload → Trained Model (categorization) → PostgreSQL (storage) → Power BI (dashboard)
                                                    ↓
                                            Streamlit Chatbot (Ollama)
```

---

## Project Structure

```
finance-tracker/
├── .env                          # API keys and database credentials (not in git)
├── .gitignore
├── requirements.txt
├── README.md
│
├── data/
│   ├── seed_transactions.csv     # 1,252 LLM-generated seed descriptions
│   ├── synthetic_transactions.csv # 25 test transactions
│   ├── training_data.csv         # 100K simulated records (generated)
│   └── scale_test.csv            # 2M simulated records (generated)
│
├── scripts/
│   ├── generate_seeds.py         # generates seed descriptions using Groq
│   ├── simulator.py              # scales seeds to training datasets
│   ├── train.py                  # trains XGBoost + Logistic Regression
│   ├── evaluate.py               # three-way model comparison
│   ├── categorize.py             # production categorization pipeline
│   └── test_models.py            # interactive model testing
│
├── models/
│   ├── train.py                  # (alternate location)
│   └── saved/                    # trained model files (.pkl)
│
├── app/
│   └── chatbot.py                # Streamlit conversational assistant
│
└── output/
    ├── categorized_transactions.csv
    ├── evaluation_results.json
    ├── evaluation_charts/        # comparison charts (PNG)
    └── confusion_matrices/       # confusion matrix plots
```

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- Docker Desktop
- Ollama
- Power BI Desktop (optional, for dashboard)

### 1. Clone and install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/finance-tracker.git
cd finance-tracker
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 2. Create `.env` file

Create a file named `.env` in the project root:

```
GROQ_API_KEY=your-groq-api-key
DB_HOST=localhost
DB_NAME=postgres
DB_USER=finance
DB_PASSWORD=finance123
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Start PostgreSQL

Make sure Docker Desktop is running, then:

```bash
docker run -d --name finance-db -e POSTGRES_USER=finance -e POSTGRES_PASSWORD=finance123 -e POSTGRES_DB=transactions -p 5432:5432 postgres:16
```

Create the table using DBeaver or any SQL client (connect to `localhost:5432`, database `postgres`, user `finance`, password `finance123`):

```sql
CREATE TABLE transactions (
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
```

### 4. Install Ollama

Download from [ollama.com/download](https://ollama.com/download), install, restart your terminal, then:

```bash
ollama pull llama3.1:8b
```

---

## Pipeline — Step by Step

### Step 1: Generate seed transactions

Uses Groq API to create 1,252 diverse, realistic bank transaction descriptions across 8 categories.

```bash
python scripts/generate_seeds.py
```

Output: `data/seed_transactions.csv`

### Step 2: Simulate training data

Samples from seeds and applies mutations (store number swaps, location suffixes, capitalization changes, typos) to generate large-scale training data.

```bash
# 100K records
python scripts/simulator.py --records 100000

# 2M records
python scripts/simulator.py --records 2000000 --output data/scale_test.csv
```

Output: `data/training_data.csv`

### Step 3: Train models

Trains XGBoost and Logistic Regression with TF-IDF features.

```bash
# Train on 100K
python scripts/train.py --tag 100k

# Train on 2M
python scripts/train.py --data data/scale_test.csv --tag 2m
```

Output: `models/saved/*.pkl`

### Step 4: Evaluate models

Three-way comparison: XGBoost vs Logistic Regression vs Ollama (prompt engineering). All evaluated on the same test set.

```bash
python scripts/evaluate.py
```

Output: `output/evaluation_results.json`, `output/evaluation_charts/`

### Step 5: Categorize transactions

Run the production pipeline on new transaction data. Uses the trained Logistic Regression model (no API needed).

```bash
python scripts/categorize.py
python scripts/categorize.py --input data/my_transactions.csv
python scripts/categorize.py --no-db    # skip PostgreSQL
```

Output: `output/categorized_transactions.csv` + PostgreSQL insert

### Step 6: Launch chatbot

Interactive conversational assistant that queries PostgreSQL and uses Ollama for natural language responses.

```bash
streamlit run app/chatbot.py
```

Opens at `http://localhost:8501`

---

## Data Pipeline

### Seed Generation

- 1,252 seed transactions generated via Groq API (Llama 3.3 70B)
- 8 categories: Food & Dining, Groceries, Transportation, Shopping, Entertainment, Health & Pharmacy, Utilities, Income
- Each seed includes realistic messy bank description, category label, and amount range

### Simulation

- Three personas (student, professional, family) with different spending biases
- Seasonal multipliers (more shopping in Nov/Dec, higher utilities in winter)
- Weekend boosts (more food and entertainment spending)
- Description mutations: store number swaps, location suffixes, capitalization changes, typos
- 37,905 unique descriptions from 1,252 seeds (100K dataset)
- 425,344 unique descriptions (2M dataset)

### Feature Engineering

- TF-IDF vectorization with unigrams + bigrams
- Max 10,000 features, min document frequency of 3
- Sublinear TF scaling
- Vocabulary size: ~5,400 features

---

## Model Details

### Logistic Regression (Best Performer)

- Multinomial classification with L-BFGS solver
- Why it wins: transaction categorization is a linear keyword-matching problem. "STARBUCKS" maps directly to Food & Dining — no complex feature interactions needed.
- Training time: ~1 second
- Inference: 0.0001ms per transaction

### XGBoost

- 200 estimators, max depth 6, learning rate 0.1
- Slightly lower accuracy due to overfitting on noise (store numbers, location codes)
- Training time: ~8 seconds (100K), ~2 minutes (2M)
- Inference: 0.004ms per transaction

### Ollama / LLM Prompt Engineering

- Llama 3.1 8B running locally via Ollama
- Zero-shot classification with structured prompting
- Struggles with: Groceries vs Shopping (0.73 F1), Utilities (0.78 F1)
- Inference: 579ms per transaction (5.8 million times slower than LogReg)

### Data Scaling Analysis

| Dataset | LogReg Accuracy | XGBoost Accuracy | Improvement |
|---|---|---|---|
| 100K | 99.54% | 98.73% | — |
| 2M | 99.74% | 99.10% | +0.20% / +0.37% |

20x more data yielded less than 0.4% improvement, demonstrating that data quality (diverse seeds) matters more than data quantity.

---

## Chatbot

The Streamlit chatbot provides a conversational interface to query spending data:

- Connects to PostgreSQL for live transaction data
- Uses Ollama (Llama 3.1 8B) for natural language understanding
- Context-aware: automatically fetches relevant data based on the question
- Maintains conversation history for follow-up questions

Sample questions:
- "How much did I spend on food?"
- "What are my top spending categories?"
- "Give me tips to save money"
- "How does this month compare to last month?"

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| ML Models | scikit-learn, XGBoost | Transaction categorization |
| Feature Engineering | TF-IDF (scikit-learn) | Text to numeric features |
| LLM (Evaluation) | Ollama / Llama 3.1 8B | Prompt engineering baseline |
| LLM (Chatbot) | Ollama / Llama 3.1 8B | Conversational assistant |
| Seed Generation | Groq API / Llama 3.3 70B | Generating training data seeds |
| Database | PostgreSQL 16 (Docker) | Transaction storage |
| Dashboard | Power BI | Spending visualizations |
| Chatbot UI | Streamlit | Web interface |
| Containerization | Docker | PostgreSQL deployment |

---

## Requirements

```
python-dotenv
requests
psycopg2-binary
scikit-learn
xgboost
pandas
matplotlib
seaborn
joblib
streamlit
plotly
```

---

## License

This project was developed for CS5100 — Foundations of Artificial Intelligence at Northeastern University, Khoury College of Computer Sciences.
