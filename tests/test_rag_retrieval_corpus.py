import json
from pathlib import Path

import pytest


CASES = json.loads(
    (Path(__file__).parent / "rag_retrieval_cases.json").read_text(encoding="utf-8")
) + json.loads(
    (Path(__file__).parent / "rag_retrieval_cases_extended.json").read_text(
        encoding="utf-8"
    )
)


# Không nhận đầu vào; xác nhận eval có tối thiểu 100 query và bao phủ mọi nhóm bắt buộc.
def test_retrieval_corpus_covers_all_sample_documents() -> None:
    assert len(CASES) >= 100
    assert {
        case.get("category", "regression") for case in CASES
    } >= {
        "regression",
        "paraphrase",
        "typo",
        "very_short",
        "exact_sku",
        "hard_negative",
        "no_answer",
        "metadata_filter",
        "multi_condition",
    }


# Nhận từng retrieval case; xác nhận query, expected source và metadata filter đúng kiểu.
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_retrieval_case_contract(case) -> None:
    assert case["query"].strip()
    expected = case["expected_document"]
    assert expected is None or expected.strip()
    assert isinstance(case.get("filters", {}), dict)
