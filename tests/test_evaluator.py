"""Tests for the evaluation metric and the end-to-end recall computation.

The evaluator imports the generator module (which imports PyTorch), so this
module is skipped when the LLM dependencies are not installed (fast CI).
"""

import json
from pathlib import Path

import pytest

pytest.importorskip(
    "torch",
    exc_type=ImportError,
    reason="evaluator imports the torch-based generator",
)

from src.evaluator import _calculate_iou, run_evaluation  # noqa: E402
from src.models import MinimalSource  # noqa: E402


def _source(file_path: str, start: int, end: int) -> MinimalSource:
    return MinimalSource(
        file_path=file_path,
        first_character_index=start,
        last_character_index=end,
    )


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (_source("f.py", 0, 100), _source("f.py", 0, 100), 1.0),
        (_source("f.py", 0, 100), _source("f.py", 200, 300), 0.0),
        (_source("f.py", 0, 100), _source("f.py", 50, 150), 1 / 3),
        (_source("f.py", 0, 0), _source("f.py", 0, 0), 0.0),
    ],
)
def test_calculate_iou(
    first: MinimalSource, second: MinimalSource, expected: float
) -> None:
    assert _calculate_iou(first, second) == pytest.approx(expected)


def test_run_evaluation_computes_recall(tmp_path: Path) -> None:
    answered_path = tmp_path / "answered.json"
    answered_path.write_text(
        json.dumps(
            {
                "rag_questions": [
                    {
                        "question_id": "hit",
                        "question": "A?",
                        "answer": "a",
                        "sources": [
                            {
                                "file_path": "f.py",
                                "first_character_index": 0,
                                "last_character_index": 100,
                            }
                        ],
                    },
                    {
                        "question_id": "miss",
                        "question": "B?",
                        "answer": "b",
                        "sources": [
                            {
                                "file_path": "f.py",
                                "first_character_index": 0,
                                "last_character_index": 50,
                            }
                        ],
                    },
                ]
            }
        )
    )
    search_results_path = tmp_path / "search_results.json"
    search_results_path.write_text(
        json.dumps(
            {
                "search_results": [
                    {
                        "question_id": "hit",
                        "question": "A?",
                        "retrieved_sources": [
                            {
                                "file_path": "f.py",
                                "first_character_index": 0,
                                "last_character_index": 100,
                            }
                        ],
                    },
                    {
                        "question_id": "miss",
                        "question": "B?",
                        "retrieved_sources": [
                            {
                                "file_path": "f.py",
                                "first_character_index": 200,
                                "last_character_index": 300,
                            }
                        ],
                    },
                ],
                "k": 1,
            }
        )
    )

    recall = run_evaluation(search_results_path, answered_path)

    assert recall == pytest.approx(0.5)
