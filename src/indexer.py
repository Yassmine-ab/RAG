import pickle
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from src.chunking import chunk_markdown, chunk_python
from src.models import MinimalSource

MIN_CHUNK_SIZE = 50


def _list_source_files(raw_dir: Path) -> list[Path]:
    py_files = list(raw_dir.rglob("*.py"))
    md_files = list(raw_dir.rglob("*.md"))
    return py_files + md_files


def _build_corpus(
    files: list[Path], max_chunk_size: int
) -> tuple[list[str], list[MinimalSource]]:
    corpus_texts: list[str] = []
    corpus_sources: list[MinimalSource] = []
    for file_path in tqdm(files, desc="Chunking"):
        content = file_path.read_text()
        chunks = (
            chunk_python(content, max_chunk_size)
            if file_path.suffix == ".py"
            else chunk_markdown(content, max_chunk_size)
        )
        for chunk in chunks:
            corpus_texts.append(chunk.text)
            corpus_sources.append(
                MinimalSource(
                    file_path=str(file_path),
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index_exclusive,
                )
            )
    return corpus_texts, corpus_sources


def _build_index(
    corpus_texts: list[str], corpus_sources: list[MinimalSource], method: str
) -> dict[str, object]:
    if method not in ("tfidf", "bm25"):
        raise ValueError(f"Unknown method: {method!r}. Use 'tfidf' or 'bm25'.")
    if method == "tfidf":
        corpus_texts, corpus_sources = _filter_short_chunks(
            corpus_texts, corpus_sources
        )
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus_texts)
        return {
            "method": "tfidf",
            "vectorizer": vectorizer,
            "matrix": matrix,
            "corpus_sources": corpus_sources,
        }
    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    return {"method": "bm25", "bm25": bm25, "corpus_sources": corpus_sources}


def _filter_short_chunks(
    corpus_texts: list[str], corpus_sources: list[MinimalSource]
) -> tuple[list[str], list[MinimalSource]]:
    filtered_texts: list[str] = []
    filtered_sources: list[MinimalSource] = []
    for text, source in zip(corpus_texts, corpus_sources):
        if len(text) >= MIN_CHUNK_SIZE:
            filtered_texts.append(text)
            filtered_sources.append(source)
    return filtered_texts, filtered_sources


def _save_index(index_path: Path, index_data: dict[str, object]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_bytes(pickle.dumps(index_data))


def run_indexing(
    method: str, max_chunk_size: int, raw_dir: Path, index_path: Path
) -> None:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {raw_dir}. "
            "Make sure the vLLM repository is present under data/raw/."
        )
    files = _list_source_files(raw_dir)
    corpus_texts, corpus_sources = _build_corpus(files, max_chunk_size)
    index_data = _build_index(corpus_texts, corpus_sources, method=method)
    _save_index(index_path, index_data)
