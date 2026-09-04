from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any


# Chuyển log ứng dụng thành từng dòng JSON để dễ theo dõi.
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in (
            "requestId",
            "provider",
            "model",
            "latencyMs",
            "inputTokens",
            "outputTokens",
            "status",
            "attempt",
            "errorCode",
            "event",
            "step",
            "tool",
            "toolCallId",
            "arguments",
            "result",
            "totalSteps",
            "totalToolCalls",
        ):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
