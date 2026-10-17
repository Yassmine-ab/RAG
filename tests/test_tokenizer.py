"""Unit tests for the identifier-aware BM25 tokenizer."""

from src.indexer import tokenize_for_bm25


def test_snake_case_identifiers_are_split() -> None:
    tokens = tokenize_for_bm25(
        "def load_lora_adapter(adapter_name: str) -> str: ..."
    )

    assert "load" in tokens
    assert "lora" in tokens
    assert "adapter" in tokens
    assert "load_lora_adapter" not in tokens


def test_camel_case_identifiers_are_split() -> None:
    tokens = tokenize_for_bm25("class LoRAAdapter: pass")

    assert "adapter" in tokens
    assert "LoRAAdapter".lower() not in tokens


def test_acronym_is_split_into_letter_groups() -> None:
    tokens = tokenize_for_bm25("LoRA")

    assert tokens == ["lo", "ra"]


def test_stop_words_are_dropped() -> None:
    tokens = tokenize_for_bm25("how to load the adapter")

    assert "how" not in tokens
    assert "to" not in tokens
    assert "the" not in tokens
    assert "adapter" in tokens


def test_single_character_tokens_are_dropped() -> None:
    assert tokenize_for_bm25("a b c ab") == ["ab"]


def test_file_path_is_prepended_for_documents() -> None:
    tokens = tokenize_for_bm25("body", file_path="docs/features/lora.md")

    assert tokens[:3] == ["docs", "features", "lora"]


def test_file_path_is_not_used_for_queries() -> None:
    tokens = tokenize_for_bm25("How to load a LoRA adapter?")

    assert "docs" not in tokens
    assert "features" not in tokens
    assert "load" in tokens
    assert "adapter" in tokens
