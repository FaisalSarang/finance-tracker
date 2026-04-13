"""Dashboard stats and analytics endpoints."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter

from backend.models.schemas import StatsResponse, CategoryBreakdown, MonthlyTrend
from backend.services import db_service

router = APIRouter(tags=["stats"])


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    category: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
):
    return await db_service.get_stats(category, start_date, end_date)


@router.get("/analytics/by-category", response_model=list[CategoryBreakdown])
async def category_breakdown(
    start_date: date | None = None,
    end_date: date | None = None,
):
    return await db_service.get_category_breakdown(start_date, end_date)


@router.get("/analytics/monthly", response_model=list[MonthlyTrend])
async def monthly_trends(
    start_date: date | None = None,
    end_date: date | None = None,
    category: str | None = None,
):
    return await db_service.get_monthly_trends(start_date, end_date, category)
