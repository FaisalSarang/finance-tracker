"""Async PostgreSQL connection pool using asyncpg."""

from __future__ import annotations

import asyncpg
from backend.config import DATABASE_URL

pool: asyncpg.Pool | None = None


async def init_pool():
    global pool
    pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10)


async def close_pool():
    global pool
    if pool:
        await pool.close()


async def fetch(sql: str, *args) -> list[dict]:
    """Execute a SELECT and return rows as list of dicts."""
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *args)
        return [dict(r) for r in rows]


async def fetchrow(sql: str, *args) -> dict | None:
    async with pool.acquire() as conn:
        row = await conn.fetchrow(sql, *args)
        return dict(row) if row else None


async def execute(sql: str, *args) -> str:
    async with pool.acquire() as conn:
        return await conn.execute(sql, *args)
