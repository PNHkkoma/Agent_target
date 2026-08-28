from __future__ import annotations

import asyncio

from experiments.common import post, save


def make_history(count: int) -> list[dict[str, str]]:
    history = []
    for index in range(1, count + 1):
        role = "user" if index % 2 else "assistant"
        history.append(
            {
                "role": role,
                "content": f"Message {index}: " + ("context payload " * 10),
            }
        )
    return history


async def main() -> None:
    history = make_history(100)
    results = []
    for label, selected_history in (
        ("100_messages", history),
        ("last_20_messages", history[-20:]),
    ):
        response = await post(
            "/chat",
            {
                "message": "Reply with only: received",
                "history": selected_history,
                "options": {"temperature": 0, "max_tokens": 20},
            },
        )
        results.append({"experiment": label, **response})
    destination = save("context-window.json", results)
    print(f"Saved comparison to {destination}")


if __name__ == "__main__":
    asyncio.run(main())

