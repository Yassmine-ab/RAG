import torch
import json
import uuid
from pathlib import Path
from tqdm import tqdm
from typing import Any
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    logging
)
from src.models import (
    MinimalSource,
    MinimalAnswer,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)
from src.retriever import run_searching

logging.set_verbosity_error()


MAX_NEW_TOKENS = 512


def _get_chunk_text(source: MinimalSource) -> str:
    """Read the original text of a chunk from its source file."""
    file_path = Path(source.file_path)
    content = file_path.read_text()
    return content[source.first_character_index: source.last_character_index]


def _build_prompt(question: str, sources: list[MinimalSource]) -> str:
    """Build a grounded prompt from a question and its retrieved sources."""
    context_blocks = []
    for i, source in enumerate(sources):
        text = _get_chunk_text(source)
        context_blocks.append(f"[Source {i + 1}: {source.file_path}]\n{text}")
    context = "\n\n".join(context_blocks)
    return (
        "You are a helpful assistant answering questions about the vLLM "
        "codebase. Use ONLY the context below to answer the question. "
        "If the context does not contain the answer, say so clearly. "
        "Be concise and factual.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
    )


class Small_LLM_Model:
    """Loads Qwen once and generates answers from retrieved context."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name

        # Auto-select device with priority: mps > cuda > cpu
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        if dtype is None:
            dtype = (
                torch.float16
                if self.device in ["cuda", "mps"]
                else torch.float32
            )
        self.dtype = dtype

        # --- load tokenizer & model ---------------------------------------
        self.tokenizer: Any = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            # ensure we have a pad token to keep batch helpers happy
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_class: Any = AutoModelForCausalLM
        self.model: Any = model_class.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

        # switch to inference-only mode
        for p in self.model.parameters():
            p.requires_grad = False

    def generate(self, question: str, sources: list[MinimalSource]) -> str:
        """Generate a grounded answer for a question from its sources."""
        prompt = _build_prompt(question, sources)
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        ).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        input_length = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_length:]
        text: str = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text.strip()


def run_answering(question: str, k: int, index_path: Path) -> MinimalAnswer:
    """Answer a single query using the retrieved context."""
    sources = run_searching(question, k, index_path)
    generator = Small_LLM_Model()
    answer_text = generator.generate(question, sources)
    return MinimalAnswer(
        question_id=str(uuid.uuid4()),
        question=question,
        retrieved_sources=sources,
        answer=answer_text,
    )


def load_student_search_results(
    student_search_results_path: Path,
) -> StudentSearchResults:
    """Load and validate a StudentSearchResults file from JSON."""
    if not student_search_results_path.exists():
        raise FileNotFoundError(
            f"Search results not found: {student_search_results_path}"
        )
    raw = json.loads(student_search_results_path.read_text())
    return StudentSearchResults.model_validate(raw)


def run_answer_dataset(
    student_search_results_path: Path, save_directory: Path
) -> Path:
    """Generate answers for every question in a StudentSearchResults file."""
    search_results = load_student_search_results(student_search_results_path)
    generator = Small_LLM_Model()
    answers: list[MinimalAnswer] = []
    for search_result in tqdm(
        search_results.search_results, desc="Generating answers"
    ):
        answer_text = generator.generate(
            search_result.question, search_result.retrieved_sources
        )
        answers.append(
            MinimalAnswer(
                question_id=search_result.question_id,
                question=search_result.question,
                retrieved_sources=search_result.retrieved_sources,
                answer=answer_text,
            )
        )
    output = StudentSearchResultsAndAnswer(
        search_results=answers, k=search_results.k
    )
    save_directory.mkdir(parents=True, exist_ok=True)
    output_path = save_directory / student_search_results_path.name
    output_path.write_text(output.model_dump_json(indent=2))
    return output_path
