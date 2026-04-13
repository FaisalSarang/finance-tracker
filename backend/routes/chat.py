"""AI chatbot endpoint (Ollama)."""

from fastapi import APIRouter

from backend.models.schemas import ChatRequest, ChatResponse
from backend.services import ollama_service
from backend.services.context_builder import get_spending_context

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
async def chat(body: ChatRequest):
    context = await get_spending_context()
    reply = await ollama_service.ask(body.message, body.history, context)
    return {"reply": reply}
