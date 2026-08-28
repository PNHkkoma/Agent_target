import json
from pathlib import Path

import pytest


PROMPTS = json.loads((Path(__file__).parent / "prompt_cases.json").read_text(encoding="utf-8"))


def test_prompt_gate_contains_20_to_50_cases() -> None:
    assert 20 <= len(PROMPTS) <= 50
    assert len({case["id"] for case in PROMPTS}) == len(PROMPTS)


@pytest.mark.parametrize("case", PROMPTS, ids=lambda case: case["id"])
def test_every_automated_prompt_has_a_valid_contract(case) -> None:
    assert case["kind"] in {"raw", "structured"}
    assert isinstance(case["prompt"], str) and case["prompt"].strip()

