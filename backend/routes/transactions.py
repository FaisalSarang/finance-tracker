"""Transaction CRUD endpoints."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException, Query

from backend.models.schemas import (
    TransactionCreate,
    TransactionUpdate,
    TransactionResponse,
)
from backend.services import db_service, ml_service

import uuid

router = APIRouter(tags=["transactions"])


@router.get("/transactions", response_model=list[TransactionResponse])
async def list_transactions(
    category: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    sort_by: str = Query("date_desc", pattern="^(date_desc|date_asc|amount_desc|amount_asc)$"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: str | None = None,
):
    rows = await db_service.get_transactions(
        category=category,
        start_date=start_date,
        end_date=end_date,
        sort_by=sort_by,
        limit=limit,
        offset=offset,
        search=search,
    )
    return rows


@router.post("/transactions", response_model=TransactionResponse, status_code=201)
async def create_transaction(body: TransactionCreate):
    data = body.model_dump()

    # Auto-generate transaction_id if not provided
    if not data.get("transaction_id"):
        data["transaction_id"] = f"TXN-{uuid.uuid4().hex[:12].upper()}"

    # Auto-categorize if no category provided
    if not data.get("category"):
        results = ml_service.categorize([data["description"]])
        data["category"] = results[0]["category"]
        data["confidence"] = results[0]["confidence"]

    row = await db_service.create_transaction(data)
    if row is None:
        raise HTTPException(status_code=409, detail="Duplicate transaction_id")
    return row


@router.put("/transactions/{txn_id}", response_model=TransactionResponse)
async def update_transaction(txn_id: int, body: TransactionUpdate):
    data = body.model_dump(exclude_unset=True)
    row = await db_service.update_transaction(txn_id, data)
    if row is None:
        raise HTTPException(status_code=404, detail="Transaction not found or no changes")
    return row


@router.delete("/transactions/{txn_id}", status_code=204)
async def delete_transaction(txn_id: int):
    deleted = await db_service.delete_transaction(txn_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Transaction not found")
