"""ML categorization endpoint."""

from fastapi import APIRouter

from backend.models.schemas import CategorizeRequest, CategorizeResponseItem
from backend.services import ml_service

router = APIRouter(tags=["categorize"])


@router.post("/categorize", response_model=list[CategorizeResponseItem])
async def categorize(body: CategorizeRequest):
    return ml_service.categorize(body.descriptions)
