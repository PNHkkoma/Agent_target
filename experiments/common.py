from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx


API_URL = "http://localhost:8000/api"
RESULTS_DIR = Path("experiment-results")


async def post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(f"{API_URL}{path}", json=payload)
        response.raise_for_status()
        return response.json()


def save(name: str, data: Any) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    destination = RESULTS_DIR / name
    destination.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination

