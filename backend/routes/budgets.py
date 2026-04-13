"""Budget CRUD endpoints."""

from fastapi import APIRouter, HTTPException, Query

from backend.models.schemas import BudgetCreate, BudgetUpdate, BudgetResponse
from backend.services import db_service

router = APIRouter(tags=["budgets"])


@router.get("/budgets", response_model=list[BudgetResponse])
async def list_budgets(
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$", description="Month in YYYY-MM format"),
):
    return await db_service.get_budgets(month)


@router.post("/budgets", response_model=BudgetResponse, status_code=201)
async def create_budget(body: BudgetCreate):
    row = await db_service.create_budget(body.category, body.monthly_limit, body.month)
    if row is None:
        raise HTTPException(status_code=500, detail="Failed to create budget")
    # Return with spent/remaining info
    budgets = await db_service.get_budgets(body.month)
    for b in budgets:
        if b["category"] == body.category:
            return b
    return row


@router.put("/budgets/{budget_id}", response_model=BudgetResponse)
async def update_budget(budget_id: int, body: BudgetUpdate):
    row = await db_service.update_budget(budget_id, body.monthly_limit)
    if row is None:
        raise HTTPException(status_code=404, detail="Budget not found")
    # Return with spent/remaining info
    budgets = await db_service.get_budgets(row["month"])
    for b in budgets:
        if b["id"] == budget_id:
            return b
    return row
