import pickle
import json
from pathlib import Path
from tqdm import tqdm
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity
from src.indexer import tokenize_for_bm25
from src.models import (
    MinimalSource,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
)


def _load_index(index_path: Path) -> dict[str, Any]:
    """Load a previously persisted index from disk."""
    if not index_path.exists():
        raise FileNotFoundError(
            f"Index not found: {index_path}. "
            "Run the 'index' command first to build it."
        )
    data: dict[str, Any] = pickle.loads(index_path.read_bytes())
    return data


def _search_in_index(
    question: str, k: int, data: dict[str, Any]
) -> list[MinimalSource]:
    """Search for the top-k sources in the index for a single question."""
    if data["method"] not in ("tfidf", "bm25"):
        raise ValueError(f"Unknown method in index: {data['method']!r}.")
    if data["method"] == "tfidf":
        question_vector = data["vectorizer"].transform([question])
        scores = cosine_similarity(question_vector, data["matrix"]).ravel()
    else:
        tokenized_question = tokenize_for_bm25(question)
        scores = data["bm25"].get_scores(tokenized_question)
    top_k = scores.argsort()[::-1][:k]
    return [data["corpus_sources"][i] for i in top_k]


def run_searching(
    question: str, k: int, index_path: Path
) -> list[MinimalSource]:
    """Search for the top-k sources in the index for a single question."""
    if k <= 0:
        raise ValueError("k must be greater than 0")
    return _search_in_index(question, k, _load_index(index_path))


def load_dataset(dataset_path: Path) -> RagDataset:
    """Load and validate a RAG question dataset from JSON."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    raw = json.loads(dataset_path.read_text())
    return RagDataset.model_validate(raw)


def run_search_dataset(
    dataset_path: Path, k: int, index_path: Path, save_directory: Path
) -> Path:
    """Run search for every question in a dataset and save the results."""
    if k <= 0:
        raise ValueError("k must be greater than 0")
    dataset = load_dataset(dataset_path)
    data = _load_index(index_path)
    results: list[MinimalSearchResults] = []
    for item in tqdm(dataset.rag_questions, desc="Searching"):
        sources = _search_in_index(item.question, k, data)
        results.append(
            MinimalSearchResults(
                question_id=item.question_id,
                question=item.question,
                retrieved_sources=sources,
            )
        )
    output = StudentSearchResults(search_results=results, k=k)
    save_directory.mkdir(parents=True, exist_ok=True)
    output_path = (
        save_directory / dataset_path.name
    )  # .name retourne uniquement le dernier segment du chemin
    output_path.write_text(output.model_dump_json(indent=2))
    return output_path
