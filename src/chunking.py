import ast
from dataclasses import dataclass
from typing import Union

DefNode = Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]


@dataclass
class TextChunk:
    first_character_index: int
    last_character_index_exclusive: int
    text: str


def _find_heading_positions(text: str) -> list[int]:
    positions: list[int] = []
    position = 0
    for line in text.splitlines(keepends=True):
        if line.startswith("#"):
            positions.append(position)
        position += len(line)
    return positions


def _split_by_positions(text: str, positions: list[int]) -> list[TextChunk]:
    if not positions or positions[0] != 0:
        positions.insert(0, 0)
    chunks: list[TextChunk] = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        chunks.append(TextChunk(start, end, text[start:end]))
    return chunks


def _split_by_max_size(
    text: str, start_pos: int, max_chunk_size: int
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    position = 0
    length = len(text)
    while position < length:
        end = min(position + max_chunk_size, length)
        chunks.append(
            TextChunk(
                start_pos + position, start_pos + end, text[position:end]
            )
        )
        position = end
    return chunks


def chunk_markdown(text: str, max_chunk_size: int) -> list[TextChunk]:
    positions = _find_heading_positions(text)
    sections = _split_by_positions(text, positions)
    chunks: list[TextChunk] = []
    for section in sections:
        if section.text.strip():
            chunks.extend(
                _split_by_max_size(
                    section.text, section.first_character_index, max_chunk_size
                )
            )
    return chunks


def _find_line_positions(text: str) -> list[int]:
    positions: list[int] = []
    position = 0
    for line in text.split("\n"):
        positions.append(position)
        position += len(line) + 1
    return positions


def _node_start_position(node: DefNode, line_positions: list[int]) -> int:
    if node.decorator_list:
        first_decorator = node.decorator_list[0]
        return (
            line_positions[first_decorator.lineno - 1]
            + first_decorator.col_offset
            - 1
        )  # -1 pour le '@' du décorateur
    return (
        line_positions[node.lineno - 1] + node.col_offset
    )  # lineno est indexé à 1, col_offset est indexé à 0


def _node_end_position_exclusive(
    node: DefNode, line_positions: list[int]
) -> int:
    assert node.end_lineno is not None and node.end_col_offset is not None
    return line_positions[node.end_lineno - 1] + node.end_col_offset


def chunk_python(text: str, max_chunk_size: int) -> list[TextChunk]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return _split_by_max_size(text, 0, max_chunk_size)
    root_level_nodes: list[DefNode] = [
        node
        for node in tree.body
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        )
    ]
    if not root_level_nodes:
        return _split_by_max_size(text, 0, max_chunk_size)
    line_positions = _find_line_positions(text)
    chunks: list[TextChunk] = []
    previous_end = 0
    for node in root_level_nodes:
        node_start = _node_start_position(node, line_positions)
        node_end = _node_end_position_exclusive(node, line_positions)
        if node_start > previous_end:
            leading_text = text[previous_end:node_start]
            if leading_text.strip():
                chunks.extend(
                    _split_by_max_size(
                        leading_text, previous_end, max_chunk_size
                    )
                )
        chunks.extend(
            _split_by_max_size(
                text[node_start:node_end], node_start, max_chunk_size
            )
        )
        previous_end = node_end
    if previous_end < len(text):
        trailing_text = text[previous_end:]
        if trailing_text.strip():
            chunks.extend(
                _split_by_max_size(trailing_text, previous_end, max_chunk_size)
            )
    return chunks
