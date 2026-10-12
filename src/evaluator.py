from pathlib import Path
from src.models import MinimalSource, AnsweredQuestion
from src.retriever import load_dataset
from src.generator import load_student_search_results

IOU_THRESHOLD = 0.05


def _calculate_iou(a: MinimalSource, b: MinimalSource) -> float:
    """Calculate the Intersection over Union (IoU) of two sources."""
    intersection = max(
        0,
        min(a.last_character_index, b.last_character_index)
        - max(a.first_character_index, b.first_character_index),
    )
    union = (
        (a.last_character_index - a.first_character_index)
        + (b.last_character_index - b.first_character_index)
        - intersection
    )
    return intersection / union if union > 0 else 0.0


def run_evaluation(
    student_search_results_path: Path, answered_questions_path: Path
) -> float:
    """Evaluate the recall of the retrieved sources
    against the reference sources."""
    answered_questions = load_dataset(answered_questions_path)
    search_results = load_student_search_results(student_search_results_path)
    reference_by_id = {
        q.question_id: q.sources
        for q in answered_questions.rag_questions
        if isinstance(q, AnsweredQuestion)
    }
    results_by_id = {
        r.question_id: r.retrieved_sources
        for r in search_results.search_results
    }

    recall: list[float] = []
    for question_id, reference_sources in reference_by_id.items():
        retrieved_sources = results_by_id[question_id]
        total_reference_sources = len(reference_sources)
        reference_sources_found = 0
        for reference_source in reference_sources:
            for retrieved_source in retrieved_sources:
                if retrieved_source.file_path == reference_source.file_path:
                    IoU = _calculate_iou(retrieved_source, reference_source)
                    if IoU >= IOU_THRESHOLD:
                        reference_sources_found += 1
                        break
        recall.append(reference_sources_found / total_reference_sources)
    return sum(recall) / len(recall) if recall else 0.0
