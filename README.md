*This activity has been created as part of the 42 curriculum by yassabda.*

# RAG against the machine

## Description

This project implements a Retrieval-Augmented Generation (RAG) system that answers natural-language
questions about the [vLLM](https://github.com/vllm-project/vllm) codebase. The system ingests the
provided vLLM repository into a searchable lexical index, retrieves the most relevant source snippets
for a given question, and generates a grounded natural-language answer using `Qwen/Qwen3-0.6B`.

The pipeline follows the four classic RAG stages:

1. **Indexing** — chunk the corpus (Python code and Markdown docs) and build a lexical search index.
2. **Retrieval** — match a question against the index and return the top-k most relevant source locations.
3. **Augmenting** — read the exact text of each retrieved source and assemble it into a grounded prompt.
4. **Generating** — pass that context to Qwen3-0.6B to produce a concise, grounded answer.

Retrieval quality is measured with a recall@k metric (IoU ≥ 0.05, same-file match), reported here and
by the moulinette.

## Instructions

### Requirements

- Python 3.10+
- [`uv`](https://docs.astral.sh/uv/) as package/project manager
- The vLLM repository placed under `data/raw/` (not versioned in this repo)

### Installation

```bash
make install
# equivalent to: uv sync
```

### Running the pipeline

All commands follow `uv run python -m src <command> [options]` (or `make run ARGS="<command> [options]"`).

```bash
# 1. Build the index
uv run python -m src index --max_chunk_size 2000 --method tfidf

# 2. Search a single query
uv run python -m src search "How to configure the OpenAI server?" --k 5

# 3. Search a whole dataset
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions

# 4. Answer a single query
uv run python -m src answer "How to configure the OpenAI server?" --k 5

# 5. Generate answers for a dataset
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions

# 6. Evaluate retrieval quality (own implementation, for iteration)
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

### Makefile targets

| Target | Description |
|---|---|
| `make install` | Install project dependencies (`uv sync`) |
| `make run ARGS="..."` | Run the CLI with the given arguments |
| `make debug ARGS="..."` | Run the CLI under `pdb` |
| `make clean` | Remove caches (`__pycache__`, `.mypy_cache`, `.pytest_cache`) |
| `make lint` | Run `flake8` and `mypy` with the required flags |
| `make lint-strict` | Run `flake8` and `mypy --strict` |
| `make test` | Run the (non-graded) `pytest` suite |

## Resources

- [vLLM documentation](https://docs.vllm.ai/)
- [scikit-learn `TfidfVectorizer` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [`rank-bm25` package](https://github.com/dorianbrown/rank_bm25)
- [Hugging Face `transformers` documentation](https://huggingface.co/docs/transformers)
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python `ast` module documentation](https://docs.python.org/3/library/ast.html)
- [Python Fire documentation](https://github.com/google/python-fire)

### AI usage disclosure

An AI assistant (Claude) was used throughout this project as a learning aid:

- **Architecture and design discussions**: deciding how to split responsibilities between chunking,
  indexing, retrieval, and generation modules; evaluating trade-offs (e.g. why to re-read chunk text
  from the raw corpus files rather than from the persisted index, to keep `answer_dataset` decoupled
  from a potentially-rebuilt index).
- **Debugging**: diagnosing bugs such as Pydantic models being called with positional instead of
  keyword arguments, decorator-aware Python chunking offsets, and a TF-IDF/cosine-similarity bias
  toward very short chunks.

All AI-suggested code was read, tested, and adapted by hand before being kept; the final implementation
choices (chunking boundaries, filtering of short chunks, prompt wording, stop-word handling) were made
and validated by running the pipeline against the real vLLM corpus.

## System architecture

```
data/raw/ (vLLM repo)
      │
      ▼
┌─────────────┐   chunk_python() / chunk_markdown()
│  chunking.py │──────────────────────────────────────┐
└─────────────┘                                        │
                                                         ▼
                                              ┌──────────────────┐
                                              │    indexer.py     │  TF-IDF or BM25
                                              │  (index command)  │──────────────┐
                                              └──────────────────┘               │
                                                                                  ▼
                                                                       data/processed/index.pkl
                                                                                  │
                           ┌──────────────────────────────────────────────────────┘
                           ▼
                 ┌───────────────────┐
                 │   retriever.py     │  search / search_dataset
                 │ (search commands)  │─────────────────────┐
                 └───────────────────┘                       │
                                                               ▼
                                                   StudentSearchResults (JSON)
                                                               │
                           ┌───────────────────────────────────┘
                           ▼
                 ┌───────────────────┐
                 │   generator.py     │  answer / answer_dataset
                 │ (Qwen/Qwen3-0.6B)  │──────────────────────┐
                 └───────────────────┘                        │
                                                                ▼
                                            StudentSearchResultsAndAnswer (JSON)

                 ┌───────────────────┐
                 │   evaluator.py     │  evaluate (recall@k, IoU-based)
                 └───────────────────┘
```

Each stage communicates through typed Pydantic models (`src/models.py`) and/or JSON files, so stages
can be run and tested independently. In particular, `answer`/`answer_dataset` never depend on the
index: they re-read chunk text directly from `data/raw/` using the `file_path`/offsets carried by each
`MinimalSource`, so a `StudentSearchResults` file stays valid even if the index is later rebuilt with
different parameters.

## Chunking strategy

Two dedicated strategies are implemented in `src/chunking.py`, as required:

- **Python (`chunk_python`)**: the file is parsed with the standard `ast` module. Top-level
  `FunctionDef`/`AsyncFunctionDef`/`ClassDef` nodes each become one chunk, including their decorators
  (the AST node for a decorated function starts at the `def` line, not the `@decorator` line, so the
  chunk boundary is explicitly backed up to include the decorator). Code between two definitions
  (imports, module-level constants) is grouped into its own chunk. Files with a syntax error, or with
  no top-level function/class, fall back to a fixed-size split.
- **Markdown (`chunk_markdown`)**: the file is split at every heading line (`#`, `##`, ...), so each
  chunk corresponds to one section of the document.

In both cases, any chunk exceeding `--max_chunk_size` (default 2000 characters, matching the
moulinette's `max_context_length`) is further split into fixed-size slices, guaranteeing that no chunk
ever exceeds the limit.

Chunks shorter than 50 characters (e.g. a lone heading immediately followed by another heading) are
dropped for the TF-IDF index: with cosine similarity, a very short chunk's TF-IDF vector is dominated
by very few words, which was observed to give it an artificially high similarity score and crowd out
genuinely relevant, longer chunks. BM25 is not filtered this way, since it already normalizes by
document length.

## Retrieval method

Both classic lexical methods mentioned in the subject are implemented, selectable with `--method`:

- **TF-IDF** (`scikit-learn`'s `TfidfVectorizer`, with English stop words removed), scored with
  cosine similarity.
- **BM25** (`rank-bm25`'s `BM25Okapi`), scored with its native `get_scores`.

The chosen method is stored inside the persisted index (`data/processed/index.pkl`), so `search`/
`search_dataset` automatically use whichever method the index was built with — no extra flag needed at
search time.

English stop-word removal was added after observing that, in a corpus dominated by code and technical
documentation, common question words (e.g. "how", "to") can have an unexpectedly *high* IDF (they are
comparatively rare in code), which let them dominate the similarity score alongside — or instead of —
the actually discriminative terms of the question.

## Performance analysis

Indexing the full vLLM corpus (~1900 files, ~18–19k chunks) consistently completes in well under the
5-minute budget (observed around 6–8 seconds on the development machine).

During development, TF-IDF without stop-word removal and without short-chunk filtering performed
poorly: the top results for "How to configure the OpenAI server?" were dominated by small, barely
relevant chunks (e.g. a single heading line), and the genuinely relevant file
(`docs/serving/openai_compatible_server.md`) did not appear in the top 5. Adding `stop_words="english"`
and dropping chunks under 50 characters moved that file into the top results.

## Design decisions

- **Two-list corpus representation** (`corpus_texts: list[str]`, `corpus_sources: list[MinimalSource]`)
  rather than a combined object: the vectorizer/BM25 index only ever needs raw text, and the output
  models only ever need `MinimalSource` — keeping them separate avoids carrying unused text through the
  whole pipeline and keeps the persisted index closer to what scikit-learn/rank-bm25 expect natively.
- **Re-reading chunk text from `data/raw/` for generation**, instead of persisting chunk text inside
  the index: this keeps `answer`/`answer_dataset` correct even if the index is rebuilt (e.g. with a
  different `max_chunk_size` or method) between producing a `StudentSearchResults` file and generating
  answers from it — a scenario explicitly encouraged by the subject when comparing chunk sizes.
  `MinimalSource`'s `file_path` + offsets always point at a stable, unchanging source file.
  `answer_dataset`'s CLI signature therefore needs no `--index_path` argument, matching the subject's
  example usage exactly.
  **Key size trade-off**: the raw corpus must remain untouched for this to stay correct.
- **`enable_thinking=False`** when calling Qwen3's chat template: Qwen3 is a "thinking" model by
  default and otherwise emits a `<think>...</think>` reasoning block before the actual answer. Since
  the `answer` field is expected to be the answer itself, generation is run without the reasoning
  trace rather than generating and then stripping it (also cheaper, since fewer tokens are generated).
- **Mypy-friendly node typing in chunking**: `_node_start_position`/`_node_end_position_exclusive` are
  typed to accept a `Union[FunctionDef, AsyncFunctionDef, ClassDef]` rather than the generic `ast.AST`,
  since only these three node types are guaranteed to expose `decorator_list`/`end_lineno`/
  `end_col_offset` — this lets `mypy --disallow-untyped-defs` pass without `getattr`/`# type: ignore`.

## Challenges faced

- **Decorator-aware Python chunking**: `ast.FunctionDef.lineno` points at the `def` line, not at any
  decorator above it, so a naive implementation split `@router.get(...)` from the function it decorates
  into two separate chunks — losing, for a FastAPI-style codebase like vLLM, exactly the HTTP route
  information most relevant to many questions. Fixed by walking back to the first decorator's position
  (and one character further, to include the `@` itself, since the decorator's own AST node excludes it).
- **TF-IDF favoring very short chunks**: cosine similarity between a question and a two-word chunk
  (e.g. a lone Markdown heading) can be artificially high, since nearly all of that chunk's weight goes
  into few, possibly-matching words. Diagnosed by inspecting the IDF weights of the query's own tokens
  and the retrieved chunks' text/length directly, then fixed with a minimum chunk-size filter.
- **Pydantic models reject positional arguments**: several early bugs came from calling
  `SomeModel(a, b, c)` instead of `SomeModel(field_a=a, field_b=b, field_c=c)` — Pydantic's `BaseModel`
  requires keyword arguments. Fixed by auditing every model instantiation in the codebase.
- **`uv`'s default "installable package" layout**: `uv init` defaults to a publishable-package layout
  (a `[project.scripts]` entry point, a package directory matching the project name) that does not
  match this project's required `src/` + `uv run python -m src <command>` convention. Fixed with
  `[tool.uv] package = false` in `pyproject.toml`, which tells `uv` to only manage the virtual
  environment, not build an installable package.

## Example usage

```bash
$ uv run python -m src index --max_chunk_size 2000 --method tfidf
Chunking: 100%|██████████| 1937/1937 [00:05<00:00, 360.00it/s]
Ingestion complete! Indices saved under data/processed/ in 6.9s (method=tfidf)

$ uv run python -m src search "How to configure the OpenAI server?" --k 5
data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md [0:1953]
data/raw/vllm-0.10.1/docs/configuration/serve_args.md [0:94]
data/raw/vllm-0.10.1/docs/design/arch_overview.md [1474:2157]
data/raw/vllm-0.10.1/docs/getting_started/quickstart.md [6294:8079]
data/raw/vllm-0.10.1/docs/getting_started/quickstart.md [8079:9263]

$ uv run python -m src answer "How to configure the OpenAI server?" --k 3
The OpenAI server can be configured by setting the `openai_api_key` and
`openai_api_base` variables in the code.
```
