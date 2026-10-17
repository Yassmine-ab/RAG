"""End-to-end tests of the indexing / retrieval stages on a tiny local corpus.

Only the lexical part of the pipeline is exercised here, so these tests run
without PyTorch or the Qwen model (useful for fast CI).
"""

import json
from pathlib import Path

import pytest

from src.indexer import MIN_CHUNK_SIZE, _filter_short_chunks, run_indexing
from src.models import MinimalSource, RagDataset, StudentSearchResults
from src.retriever import run_search_dataset, run_searching

REPO_ROOT = Path(__file__).resolve().parents[1]

PY_LORA = '''@app.get("/v1/load_lora_adapter")
def load_lora_adapter(adapter_name: str):
    """Dynamically load a LoRA adapter into the running vLLM server."""
    return {"status": "ok", "adapter": adapter_name}
'''

PY_QUANTIZE = '''def quantize_model(model, dtype: str):
    """Quantize the model weights to the requested dtype for faster inference."""
    return model.to(dtype=dtype)
'''

MD_INSTALL = """# Installation

Install vLLM with pip: `pip install vllm`. The package requires CUDA 12.
"""


@pytest.fixture()
def tiny_corpus(tmp_path: Path) -> Path:
    """A two-file corpus with predictable, non-overlapping vocabulary."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "lora.py").write_text(PY_LORA)
    (raw_dir / "quantize.py").write_text(PY_QUANTIZE)
    (raw_dir / "install.md").write_text(MD_INSTALL)
    return raw_dir


@pytest.fixture()
def dataset_file(tmp_path: Path) -> Path:
    dataset = {
        "rag_questions": [
            {"question_id": "q1", "question": "How to dynamically load a LoRA adapter?"},
            {"question_id": "q2", "question": "How do I install vLLM?"},
        ]
    }
    path = tmp_path / "unanswered.json"
    path.write_text(json.dumps(dataset))
    return path


@pytest.mark.parametrize("method", ["tfidf", "bm25"])
def test_search_finds_the_expected_source(
    method: str, tiny_corpus: Path, tmp_path: Path
) -> None:
    index_path = tmp_path / "index.pkl"
    run_indexing(method, 2000, tiny_corpus, index_path)

    results = run_searching(
        "How to dynamically load a LoRA adapter?", 2, index_path
    )

    assert len(results) == 2
    assert results[0].file_path == str(tiny_corpus / "lora.py")
    assert 0 <= results[0].first_character_index < results[0].last_character_index


PY_IDENTIFIER_LOADER = '''def load_lora_adapter(adapter_name: str) -> str:
    """Alpha bravo charlie delta echo foxtrot golf hotel india."""
    return adapter_name
'''

PY_IDENTIFIER_QUANTIZE = '''def quantize_model_weights(model, dtype: str):
    """Juliet kilo lima mike november oscar papa quebec romeo."""
    return model.to(dtype=dtype)
'''


def test_bm25_tokenizer_matches_compound_identifiers(tmp_path: Path) -> None:
    """A plain-English question must match a snake_case identifier."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "loader.py").write_text(PY_IDENTIFIER_LOADER)
    (raw_dir / "quantize.py").write_text(PY_IDENTIFIER_QUANTIZE)
    index_path = tmp_path / "index.pkl"
    run_indexing("bm25", 2000, raw_dir, index_path)

    results = run_searching("How do I load a LoRA adapter?", 1, index_path)

    assert results[0].file_path == str(raw_dir / "loader.py")


@pytest.mark.parametrize("method", ["tfidf", "bm25"])
def test_search_dataset_writes_valid_json(
    method: str, tiny_corpus: Path, dataset_file: Path, tmp_path: Path
) -> None:
    index_path = tmp_path / "index.pkl"
    run_indexing(method, 2000, tiny_corpus, index_path)

    output_path = run_search_dataset(
        dataset_file, 2, index_path, tmp_path / "out"
    )

    assert output_path.exists()
    parsed = StudentSearchResults.model_validate_json(output_path.read_text())
    assert parsed.k == 2
    assert len(parsed.search_results) == 2
    assert all(len(r.retrieved_sources) == 2 for r in parsed.search_results)


def test_index_command_rejects_unknown_method(tiny_corpus: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        run_indexing("embeddings", 2000, tiny_corpus, tmp_path / "index.pkl")


def test_search_requires_a_built_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Index not found"):
        run_searching("anything", 3, tmp_path / "missing.pkl")


def test_index_requires_the_corpus(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Corpus directory not found"):
        run_indexing("tfidf", 2000, tmp_path / "missing_raw", tmp_path / "index.pkl")


@pytest.mark.parametrize("k", [0, -1])
def test_search_rejects_non_positive_k(
    k: int, tiny_corpus: Path, tmp_path: Path
) -> None:
    index_path = tmp_path / "index.pkl"
    run_indexing("tfidf", 2000, tiny_corpus, index_path)

    with pytest.raises(ValueError, match="greater than 0"):
        run_searching("anything", k, index_path)


def test_tfidf_filter_drops_short_chunks() -> None:
    texts = ["short", "x" * MIN_CHUNK_SIZE, "y" * (MIN_CHUNK_SIZE + 10)]
    sources = [
        MinimalSource(
            file_path="f.py", first_character_index=0, last_character_index=1
        )
        for _ in texts
    ]

    filtered_texts, filtered_sources = _filter_short_chunks(texts, sources)

    assert filtered_texts == texts[1:]
    assert len(filtered_sources) == len(filtered_texts)


@pytest.mark.parametrize(
    "relative_path",
    [
        "data/datasets/AnsweredQuestions/dataset_code_public.json",
        "data/datasets/AnsweredQuestions/dataset_docs_public.json",
        "data/datasets/UnansweredQuestions/dataset_code_public.json",
        "data/datasets/UnansweredQuestions/dataset_docs_public.json",
    ],
)
def test_public_datasets_are_valid(relative_path: str) -> None:
    path = REPO_ROOT / relative_path
    if not path.exists():
        pytest.skip(f"dataset not present: {relative_path}")

    dataset = RagDataset.model_validate_json(path.read_text())

    assert len(dataset.rag_questions) > 0
