import fire
import time
from pathlib import Path
from src.indexer import run_indexing
from src.retriever import run_searching, run_search_dataset
from src.generator import run_answering, run_answer_dataset
from src.evaluator import run_evaluation

MAX_CHUNK_SIZE = 2000
K_DEFAULT = 5
DEFAULT_RAW_DIR = "data/raw"
DEFAULT_INDEX_PATH = "data/processed/index.pkl"
DEFAULT_SEARCH_RESULTS_DIR = "data/output/search_results"
DEFAULT_ANSWER_DIR = "data/output/search_results_and_answer"


class Cli:
    """Entry point exposing all pipeline commands."""

    def index(
        self, max_chunk_size: int = MAX_CHUNK_SIZE, method: str = "tfidf"
    ) -> None:
        """Ingest data/raw/ and build the index under data/processed/"""
        try:
            if not 1 <= max_chunk_size <= 2000:
                raise ValueError("max_chunk_size must be between 1 and 2000")
            start = time.time()
            run_indexing(
                method,
                max_chunk_size,
                Path(DEFAULT_RAW_DIR),
                Path(DEFAULT_INDEX_PATH),
            )
            elapsed = time.time() - start
            print(
                f"Ingestion complete! Indices saved under data/processed/ "
                f"in {elapsed:.1f}s (method={method})"
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")

    def search(self, question: str, k: int = K_DEFAULT) -> None:
        """Return the top-k sources for a single query."""
        try:
            results = run_searching(question, k, Path(DEFAULT_INDEX_PATH))
            for result in results:
                print(
                    f"{result.file_path} "
                    f"[{result.first_character_index}:"
                    f"{result.last_character_index}]"
                )
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")

    def search_dataset(
        self,
        dataset_path: str,
        k: int = K_DEFAULT,
        save_directory: str = DEFAULT_SEARCH_RESULTS_DIR,
    ) -> None:
        """Run search over a whole dataset and write a
        StudentSearchResults JSON file."""
        try:
            output_path = run_search_dataset(
                Path(dataset_path),
                k,
                Path(DEFAULT_INDEX_PATH),
                Path(save_directory),
            )
            print(f"Saved student_search_results to {output_path}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")

    def answer(self, question: str, k: int = K_DEFAULT) -> None:
        """Answer a single query using the retrieved context."""
        try:
            result = run_answering(question, k, Path(DEFAULT_INDEX_PATH))
            print(result.answer)
        except (ValueError, FileNotFoundError) as e:
            print(f"Error: {e}")

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = DEFAULT_ANSWER_DIR,
    ) -> None:
        """Generate answers for a dataset, producing a
        StudentSearchResultsAndAnswer JSON file."""
        try:
            output_path = run_answer_dataset(
                Path(student_search_results_path), Path(save_directory)
            )
            print(f"Saved student_search_results_and_answer to {output_path}")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> None:
        """Evaluate the recall of the retrieved sources
        against the reference sources."""
        try:
            recall = run_evaluation(
                Path(student_search_results_path), Path(dataset_path)
            )
            print(f"Recall: {recall:.3f} ({recall * 100:.1f}%)")
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}")


def main() -> None:
    """Run the Fire CLI."""
    fire.Fire(Cli)
