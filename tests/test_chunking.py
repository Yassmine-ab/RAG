"""Unit tests for the chunking strategies (no model / no corpus required)."""

from src.chunking import chunk_markdown, chunk_python


DECORATED_SOURCE = '''import os


@app.get("/v1/load_lora_adapter")
def load_lora_adapter(adapter_name: str):
    """Dynamically load a LoRA adapter."""
    return adapter_name
'''


def test_python_chunk_offsets_point_back_into_the_source() -> None:
    chunks = chunk_python(DECORATED_SOURCE, 2000)

    assert chunks
    for chunk in chunks:
        assert (
            DECORATED_SOURCE[
                chunk.first_character_index : chunk.last_character_index_exclusive
            ]
            == chunk.text
        )


def test_python_function_chunk_includes_its_decorator() -> None:
    chunks = chunk_python(DECORATED_SOURCE, 2000)
    function_chunks = [chunk for chunk in chunks if "def load_lora_adapter" in chunk.text]

    assert len(function_chunks) == 1
    assert function_chunks[0].text.lstrip().startswith('@app.get("/v1/load_lora_adapter")')


def test_python_module_level_code_is_its_own_chunk() -> None:
    chunks = chunk_python(DECORATED_SOURCE, 2000)
    module_chunks = [chunk for chunk in chunks if "import os" in chunk.text]

    assert len(module_chunks) == 1
    assert "def load_lora_adapter" not in module_chunks[0].text


def test_python_chunks_never_exceed_max_chunk_size() -> None:
    source = "def long_function():\n" + "    x = 1\n" * 200
    chunks = chunk_python(source, 120)

    assert chunks
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    for chunk in chunks:
        assert (
            source[chunk.first_character_index : chunk.last_character_index_exclusive]
            == chunk.text
        )


def test_python_syntax_error_falls_back_to_fixed_size_splitting() -> None:
    broken = "def broken(:\n    pass\n" * 10
    chunks = chunk_python(broken, 50)

    assert chunks
    assert "".join(chunk.text for chunk in chunks) == broken
    assert all(len(chunk.text) <= 50 for chunk in chunks)


def test_markdown_splits_on_headings() -> None:
    text = "# Title\n\nIntro paragraph.\n\n## Section\n\nSection body.\n"
    chunks = chunk_markdown(text, 2000)

    assert len(chunks) == 2
    assert chunks[0].text.startswith("# Title")
    assert chunks[1].text.startswith("## Section")
    for chunk in chunks:
        assert (
            text[chunk.first_character_index : chunk.last_character_index_exclusive]
            == chunk.text
        )


def test_markdown_without_heading_is_a_single_chunk() -> None:
    text = "Just a paragraph without any heading.\n"
    chunks = chunk_markdown(text, 2000)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].first_character_index == 0


def test_empty_input_produces_no_chunk() -> None:
    assert chunk_python("", 2000) == []
    assert chunk_markdown("", 2000) == []
