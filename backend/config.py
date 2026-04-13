"""Centralized configuration from .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Database
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "finance")
DB_PASSWORD = os.getenv("DB_PASSWORD", "finance123")
DB_PORT = int(os.getenv("DB_PORT", "5432"))

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Ollama
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/v1/chat/completions")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# ML models
MODEL_DIR = PROJECT_ROOT / "models" / "saved"

# CORS
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
