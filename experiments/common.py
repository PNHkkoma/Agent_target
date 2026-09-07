from __future__ import annotations

import json
import os
import asyncio
from pathlib import Path
from typing import Any

import httpx


API_URL = os.getenv("EXPERIMENT_API_URL", "http://localhost:8000/api").rstrip("/")
RESULTS_DIR = Path("experiment-results")


async def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    # Retry lỗi mạng/5xx ngắn hạn để report eval không bị hỏng vì một lần gọi provider chập chờn.
    async with httpx.AsyncClient(timeout=120) as client:
        for attempt in range(3):
            try:
                response = await client.post(f"{API_URL}{path}", json=payload)
                response.raise_for_status()
                return response.json()
            except (httpx.TransportError, httpx.HTTPStatusError) as error:
                retryable = not isinstance(error, httpx.HTTPStatusError) or error.response.status_code >= 500
                if not retryable or attempt == 2:
                    raise
                await asyncio.sleep(0.5 * (attempt + 1))
    raise RuntimeError("Unreachable retry state")


def save(name: str, data: Any) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    destination = RESULTS_DIR / name
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
