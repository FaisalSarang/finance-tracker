"""ML model loading and transaction categorization.

Ported from scripts/categorize.py — loads the trained TF-IDF + Logistic Regression
pipeline and exposes a categorize() function.
"""

import joblib
from backend.config import MODEL_DIR

_vectorizer = None
_model = None
_label_names = None


def load():
    """Load model artifacts. Called once at FastAPI startup."""
    global _vectorizer, _model, _label_names
    _vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    _model = joblib.load(MODEL_DIR / "logreg_model.pkl")
    _label_names = joblib.load(MODEL_DIR / "label_names.pkl")


def categorize(descriptions: list[str]) -> list[dict]:
    """Categorize transaction descriptions.

    Returns list of {category: str, confidence: float}.
    """
    features = _vectorizer.transform(descriptions)
    predictions = _model.predict(features)
    probabilities = _model.predict_proba(features)

    results = []
    for pred_idx, proba in zip(predictions, probabilities):
        results.append({
            "category": _label_names[pred_idx],
            "confidence": round(float(proba[pred_idx]), 4),
        })
    return results
