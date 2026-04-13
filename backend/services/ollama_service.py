"""Async Ollama chat integration.

Ported from scripts/chatbot.py ask_ollama() — uses httpx for async HTTP.
"""

import httpx
from backend.config import OLLAMA_URL, OLLAMA_MODEL

SYSTEM_PROMPT = (
    "You are a helpful financial assistant. The user will ask questions about their "
    "spending and transactions. You will receive their actual transaction data from a "
    "database as context. Use this data to give specific, accurate answers.\n\n"
    "Rules:\n"
    "- Always reference actual numbers from the data provided\n"
    "- Be concise and conversational — keep responses under 150 words\n"
    "- Format currency as $X.XX\n"
    "- When listing transactions, show top 5 unless asked for more\n"
    "- If asked for advice, base it on their actual spending patterns\n"
    "- Do not make up transactions or amounts that aren't in the data"
)


async def ask(question: str, history: list[dict], spending_context: str) -> str:
    """Send a question to Ollama with financial context."""

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Include last 6 messages of history
    for msg in history[-6:]:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg.get("content", ""),
        })

    # Final user message with context
    messages.append({
        "role": "user",
        "content": f"Transaction data:\n\n{spending_context}\n\nQuestion: {question}",
    })

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 1024,
                    "stream": False,
                },
            )

            if resp.status_code != 200:
                body = resp.text
                return f"Ollama returned status {resp.status_code}. Make sure the model '{OLLAMA_MODEL}' is pulled (`ollama pull {OLLAMA_MODEL}`)."

            data = resp.json()
            return data["choices"][0]["message"]["content"]

    except httpx.ConnectError:
        return "Cannot connect to Ollama. Make sure it's running: `ollama serve`"
    except httpx.ReadTimeout:
        return "The AI took too long to respond. Try a shorter question."
    except Exception as e:
        return f"Chat error: {e}"
