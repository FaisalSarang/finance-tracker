"""FastAPI application entry point.

Lifespan: preloads the ML model and opens the asyncpg pool on startup,
then tears them down on shutdown.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import CORS_ORIGINS
from backend.database import init_pool, close_pool
from backend.services import ml_service
from backend.services.db_service import ensure_budgets_table

from backend.routes import transactions, stats, categorize, chat, budgets, upload


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    await init_pool()
    ml_service.load()
    await ensure_budgets_table()
    print("Backend ready  |  DB pool open  |  ML model loaded")
    yield
    # ── Shutdown ──
    await close_pool()
    print("Backend shut down gracefully")


app = FastAPI(
    title="Finance Tracker API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──
app.include_router(transactions.router, prefix="/api")
app.include_router(stats.router, prefix="/api")
app.include_router(categorize.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(budgets.router, prefix="/api")
app.include_router(upload.router, prefix="/api")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
