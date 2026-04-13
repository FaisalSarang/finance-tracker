"""Pydantic request/response schemas."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ── Requests ──

class TransactionCreate(BaseModel):
    transaction_id: Optional[str] = None
    date: date
    description: str
    amount: Decimal
    transaction_type: Literal["debit", "credit"]
    category: Optional[str] = None
    merchant: Optional[str] = None
    notes: Optional[str] = None


class TransactionUpdate(BaseModel):
    category: Optional[str] = None
    merchant: Optional[str] = None
    description: Optional[str] = None
    amount: Optional[Decimal] = None
    date: Optional[date] = None
    transaction_type: Optional[Literal["debit", "credit"]] = None


class CategorizeRequest(BaseModel):
    descriptions: list[str]


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = Field(default_factory=list)


class BudgetCreate(BaseModel):
    category: str
    monthly_limit: Decimal
    month: str  # "YYYY-MM"


class BudgetUpdate(BaseModel):
    monthly_limit: Decimal


# ── Responses ──

class TransactionResponse(BaseModel):
    id: int
    transaction_id: str
    date: date
    description: str
    amount: float
    transaction_type: str
    category: Optional[str] = None
    merchant: Optional[str] = None
    confidence: Optional[float] = None
    created_at: Optional[datetime] = None


class StatsResponse(BaseModel):
    total_transactions: int
    total_spent: float
    total_income: float
    net_balance: float
    category_count: int
    earliest_date: Optional[date] = None
    latest_date: Optional[date] = None


class CategoryBreakdown(BaseModel):
    category: str
    count: int
    total: float
    average: float
    percentage: float


class MonthlyTrend(BaseModel):
    month: str
    total: float
    count: int
    category: Optional[str] = None


class CategorizeResponseItem(BaseModel):
    category: str
    confidence: float


class ChatResponse(BaseModel):
    reply: str


class BudgetResponse(BaseModel):
    id: int
    category: str
    monthly_limit: float
    month: str
    spent: float
    remaining: float
    percentage: float


class UploadResponse(BaseModel):
    total_parsed: int
    categorized: int
    inserted: int
    duplicates_skipped: int
    time_seconds: float
    categories: list[CategoryBreakdown]
