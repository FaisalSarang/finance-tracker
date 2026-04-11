"""
CS5100 Final Project - Three-Way Model Evaluation
Compares XGBoost, Logistic Regression, and Ollama (local LLM prompt engineering)
on the SAME full test set for a fair comparison.

Prerequisites:
    1. Install Ollama and pull a model: ollama pull llama3.1:8b
    2. Train models first: python scripts/train.py

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --ollama-model llama3.1:8b
    python scripts/evaluate.py --clear-cache
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split

# ---------- Paths ----------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "training_data.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "saved"
OUTPUT_DIR = PROJECT_ROOT / "output"
CHART_DIR = OUTPUT_DIR / "evaluation_charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Ollama config ----------

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"

SYSTEM_PROMPT = """You are a financial transaction categorizer. You will receive a numbered list of
bank transaction descriptions. For EACH transaction, return a JSON object with these fields:

{
  "index": the transaction number,
  "category": "one of the categories listed below"
}

Allowed categories (use EXACTLY these names):
- Entertainment
- Food & Dining
- Groceries
- Health & Pharmacy
- Income
- Shopping
- Transportation
- Utilities

Return ONLY a JSON array of objects. No markdown, no explanation, no backticks."""


# ---------- Ollama API ----------


def call_ollama_batch(descriptions: list[str], model: str) -> list[str]:
    """Send a batch of descriptions to Ollama and return predicted categories."""
    lines = [f'{i + 1}. "{desc}"' for i, desc in enumerate(descriptions)]
    user_prompt = "Transactions:\n" + "\n".join(lines)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "max_tokens": 4096,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            raw = response.json()["choices"][0]["message"]["content"]
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            results = json.loads(cleaned)
            results.sort(key=lambda r: r.get("index", 0))

            return [r["category"] for r in results]

        except (json.JSONDecodeError, KeyError) as e:
            if attempt < max_retries - 1:
                print(f" parse error, retrying...", end="", flush=True)
                time.sleep(1)
            else:
                return ["_INVALID_"] * len(descriptions)

        except requests.exceptions.ConnectionError:
            print("\n\nERROR: Cannot connect to Ollama.")
            print("Make sure Ollama is running: ollama serve")
            raise


def evaluate_ollama(descriptions: list[str], model: str, cache_file: Path, batch_size: int = 25) -> tuple:
    """Run Ollama on ALL descriptions, with caching and progress tracking."""

    # load from cache if available
    if cache_file.exists():
        print(f"  Loading cached predictions from {cache_file.name}...")
        with open(cache_file) as f:
            cached = json.load(f)
        return cached["predictions"], cached["total_time"]

    all_preds = []
    total_time = 0
    total_batches = (len(descriptions) + batch_size - 1) // batch_size
    failed_batches = 0

    print(f"  Running {total_batches} batches ({len(descriptions):,} transactions)...")
    print(f"  Estimated time: {total_batches * 2 / 60:.0f}–{total_batches * 4 / 60:.0f} minutes\n")

    for start in range(0, len(descriptions), batch_size):
        batch = descriptions[start : start + batch_size]
        batch_num = (start // batch_size) + 1
        pct = batch_num / total_batches * 100

        print(f"\r  [{batch_num}/{total_batches}] {pct:.1f}%", end="", flush=True)

        t0 = time.time()
        preds = call_ollama_batch(batch, model)
        elapsed = time.time() - t0
        total_time += elapsed

        # check for failures
        if preds[0] == "_INVALID_":
            failed_batches += 1

        all_preds.extend(preds)

        # periodic progress update every 50 batches
        if batch_num % 50 == 0:
            elapsed_min = total_time / 60
            rate = batch_num / total_time
            remaining = (total_batches - batch_num) / rate / 60
            print(f" | {elapsed_min:.1f}min elapsed, ~{remaining:.1f}min remaining", end="", flush=True)

    print(f"\n\n  Completed in {total_time / 60:.1f} minutes")
    if failed_batches > 0:
        print(f"  {failed_batches} batches failed to parse")

    # save cache
    with open(cache_file, "w") as f:
        json.dump({
            "predictions": all_preds,
            "total_time": total_time,
            "model": model,
            "total_records": len(descriptions),
        }, f)
    print(f"  Cached to {cache_file.name}")

    return all_preds, total_time


# ---------- Plotting ----------


def plot_metric_comparison(results: dict, output_dir: Path):
    """Bar chart comparing metrics across models."""
    models = list(results.keys())
    metrics = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    metric_labels = ["Accuracy", "F1 (Macro)", "F1 (Weighted)", "Precision (Macro)", "Recall (Macro)"]

    x = np.arange(len(metric_labels))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, model_name in enumerate(models):
        values = [results[model_name]["metrics"][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i % len(colors)])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Classification Metrics (Same Test Set)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "metric_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: metric_comparison.png")


def plot_f1_per_category(results: dict, label_names: list[str], output_dir: Path):
    """Grouped bar chart showing F1 per category."""
    models = list(results.keys())
    x = np.arange(len(label_names))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, model_name in enumerate(models):
        f1s = results[model_name]["f1_per_category"]
        values = [f1s.get(cat, 0) for cat in label_names]
        ax.bar(x + i * width, values, width, label=model_name, color=colors[i % len(colors)])

    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score per Category — Model Comparison (Same Test Set)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(label_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "f1_per_category.png", dpi=150)
    plt.close()
    print(f"  Saved: f1_per_category.png")


def plot_inference_speed(results: dict, output_dir: Path):
    """Bar chart comparing inference speed (log scale)."""
    models = list(results.keys())
    speeds = [results[m]["metrics"]["inference_per_txn_ms"] for m in models]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(models, speeds, color=colors[:len(models)])

    for bar, val in zip(bars, speeds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{val:.2f}ms", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Inference Time per Transaction (ms) — log scale")
    ax.set_title("Inference Speed Comparison (Same Test Set)")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "inference_speed.png", dpi=150)
    plt.close()
    print(f"  Saved: inference_speed.png")


def plot_confusion_matrices(results: dict, label_names: list[str], output_dir: Path):
    """Side-by-side confusion matrices."""
    models = list(results.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 7))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        cm = np.array(results[model_name]["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names, ax=ax)
        ax.set_title(f"{model_name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices — All Models (Same Test Set)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: confusion_matrices_comparison.png")


# ---------- Main ----------


def main():
    parser = argparse.ArgumentParser(description="Evaluate all models on the same test set")
    parser.add_argument("--ollama-model", type=str, default="llama3.1:8b",
                        help="Ollama model (default: llama3.1:8b)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete cached Ollama predictions and re-run")
    parser.add_argument("--data", type=str, default=None,
                        help="Training data CSV (default: data/training_data.csv)")
    args = parser.parse_args()

    data_path = Path(args.data) if args.data else DATA_FILE
    cache_file = OUTPUT_DIR / f"ollama_cache_{args.ollama_model.replace(':', '_')}.json"

    if args.clear_cache and cache_file.exists():
        cache_file.unlink()
        print(f"Cleared cache: {cache_file.name}\n")

    # load data and create the same split as training
    print("Loading data...")
    df = pd.read_csv(data_path)
    label_names = sorted(df["category"].unique())

    X_text = df["description"]
    y = df["category"]
    _, X_test_text, _, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    X_test_text = X_test_text.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    test_descriptions = X_test_text.tolist()
    test_labels = y_test.tolist()

    print(f"Test set: {len(test_descriptions):,} transactions")
    print(f"All three models will be evaluated on this exact set.\n")

    # load trained models
    print("Loading trained models...")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    xgb_model = joblib.load(MODEL_DIR / "xgboost_model.pkl")
    lr_model = joblib.load(MODEL_DIR / "logreg_model.pkl")

    X_test_tfidf = vectorizer.transform(test_descriptions)

    results = {}

    # ---------- XGBoost ----------
    print("\n=== XGBoost ===")
    t0 = time.time()
    xgb_preds = [label_names[i] for i in xgb_model.predict(X_test_tfidf)]
    xgb_time = time.time() - t0

    results["XGBoost"] = {
        "metrics": {
            "accuracy": accuracy_score(test_labels, xgb_preds),
            "f1_macro": f1_score(test_labels, xgb_preds, average="macro"),
            "f1_weighted": f1_score(test_labels, xgb_preds, average="weighted"),
            "precision_macro": precision_score(test_labels, xgb_preds, average="macro"),
            "recall_macro": recall_score(test_labels, xgb_preds, average="macro"),
            "inference_total_s": round(xgb_time, 3),
            "inference_per_txn_ms": round(xgb_time / len(test_labels) * 1000, 4),
            "test_size": len(test_labels),
        },
        "f1_per_category": dict(zip(
            label_names,
            f1_score(test_labels, xgb_preds, average=None, labels=label_names),
        )),
        "confusion_matrix": confusion_matrix(test_labels, xgb_preds, labels=label_names).tolist(),
    }
    print(f"  Accuracy: {results['XGBoost']['metrics']['accuracy']:.4f}")

    # ---------- Logistic Regression ----------
    print("\n=== Logistic Regression ===")
    t0 = time.time()
    lr_preds = [label_names[i] for i in lr_model.predict(X_test_tfidf)]
    lr_time = time.time() - t0

    results["Logistic Regression"] = {
        "metrics": {
            "accuracy": accuracy_score(test_labels, lr_preds),
            "f1_macro": f1_score(test_labels, lr_preds, average="macro"),
            "f1_weighted": f1_score(test_labels, lr_preds, average="weighted"),
            "precision_macro": precision_score(test_labels, lr_preds, average="macro"),
            "recall_macro": recall_score(test_labels, lr_preds, average="macro"),
            "inference_total_s": round(lr_time, 3),
            "inference_per_txn_ms": round(lr_time / len(test_labels) * 1000, 4),
            "test_size": len(test_labels),
        },
        "f1_per_category": dict(zip(
            label_names,
            f1_score(test_labels, lr_preds, average=None, labels=label_names),
        )),
        "confusion_matrix": confusion_matrix(test_labels, lr_preds, labels=label_names).tolist(),
    }
    print(f"  Accuracy: {results['Logistic Regression']['metrics']['accuracy']:.4f}")

    # ---------- Ollama ----------
    ollama_label = f"Ollama ({args.ollama_model})"
    print(f"\n=== {ollama_label} ===")

    try:
        ollama_preds_raw, ollama_time = evaluate_ollama(
            test_descriptions, args.ollama_model, cache_file
        )

        # clean predictions
        valid_categories = set(label_names)
        ollama_preds = []
        invalid_count = 0
        for pred in ollama_preds_raw:
            if pred in valid_categories:
                ollama_preds.append(pred)
            else:
                ollama_preds.append("_INVALID_")
                invalid_count += 1

        # for fair comparison, only evaluate on valid predictions
        valid_idx = [i for i, p in enumerate(ollama_preds) if p != "_INVALID_"]
        filtered_preds = [ollama_preds[i] for i in valid_idx]
        filtered_labels = [test_labels[i] for i in valid_idx]

        print(f"  Valid predictions: {len(filtered_preds):,} / {len(test_descriptions):,}")
        if invalid_count > 0:
            print(f"  Invalid predictions: {invalid_count} (excluded from metrics)")

        results[ollama_label] = {
            "metrics": {
                "accuracy": accuracy_score(filtered_labels, filtered_preds),
                "f1_macro": f1_score(filtered_labels, filtered_preds, average="macro", zero_division=0),
                "f1_weighted": f1_score(filtered_labels, filtered_preds, average="weighted", zero_division=0),
                "precision_macro": precision_score(filtered_labels, filtered_preds, average="macro", zero_division=0),
                "recall_macro": recall_score(filtered_labels, filtered_preds, average="macro", zero_division=0),
                "inference_total_s": round(ollama_time, 3),
                "inference_per_txn_ms": round(ollama_time / len(test_descriptions) * 1000, 4),
                "test_size": len(filtered_labels),
                "invalid_predictions": invalid_count,
            },
            "f1_per_category": dict(zip(
                label_names,
                f1_score(filtered_labels, filtered_preds, average=None, labels=label_names, zero_division=0),
            )),
            "confusion_matrix": confusion_matrix(filtered_labels, filtered_preds, labels=label_names).tolist(),
        }
        print(f"  Accuracy: {results[ollama_label]['metrics']['accuracy']:.4f}")

    except requests.exceptions.ConnectionError:
        print("\n  Skipping — Ollama not running. Start with: ollama serve")

    # ---------- Charts ----------
    print("\n--- Generating Charts ---")
    plot_metric_comparison(results, CHART_DIR)
    plot_f1_per_category(results, label_names, CHART_DIR)
    plot_inference_speed(results, CHART_DIR)
    plot_confusion_matrices(results, label_names, CHART_DIR)

    # ---------- Save results ----------
    serializable = {}
    for model_name, data in results.items():
        serializable[model_name] = {
            "metrics": {k: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
                        for k, v in data["metrics"].items()},
            "f1_per_category": {k: round(float(v), 4) for k, v in data["f1_per_category"].items()},
        }

    with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
        json.dump(serializable, f, indent=2)

    # ---------- Final comparison ----------
    models = list(results.keys())

    print("\n" + "=" * 80)
    print("                     FINAL MODEL COMPARISON (SAME TEST SET)")
    print("=" * 80)

    header = f"{'Metric':<25}" + "".join(f"{m:>25}" for m in models)
    print(header)
    print("-" * 80)

    for display, key in [
        ("Accuracy", "accuracy"),
        ("F1 (Macro)", "f1_macro"),
        ("F1 (Weighted)", "f1_weighted"),
        ("Precision (Macro)", "precision_macro"),
        ("Recall (Macro)", "recall_macro"),
        ("Inference (ms/txn)", "inference_per_txn_ms"),
        ("Total Inference (s)", "inference_total_s"),
        ("Test Samples", "test_size"),
    ]:
        row = f"{display:<25}"
        for m in models:
            val = results[m]["metrics"].get(key, "N/A")
            if isinstance(val, float):
                row += f"{val:>25.4f}"
            else:
                row += f"{val:>25}"
        print(row)

    print("=" * 80)

    print("\n" + "=" * 80)
    print("                  F1 SCORE PER CATEGORY (SAME TEST SET)")
    print("=" * 80)
    header = f"{'Category':<25}" + "".join(f"{m:>25}" for m in models)
    print(header)
    print("-" * 80)

    for cat in label_names:
        row = f"{cat:<25}"
        for m in models:
            val = results[m]["f1_per_category"].get(cat, 0)
            row += f"{val:>25.4f}"
        print(row)

    print("=" * 80)

    print(f"\nResults: {OUTPUT_DIR / 'evaluation_results.json'}")
    print(f"Charts:  {CHART_DIR}")


if __name__ == "__main__":
    main()