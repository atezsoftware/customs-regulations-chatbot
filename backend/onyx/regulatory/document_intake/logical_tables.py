"""Lossless logical pipe rows for verified multiline intake tables."""

import re
from dataclasses import dataclass, replace

from onyx.regulatory.chunker import SourceBlock, _markdown_to_blocks

_SEPARATOR = re.compile(r"\s*:?-{3,}:?\s*")


@dataclass(frozen=True)
class SourceCell:
    column_index: int
    start_char: int
    end_char: int


@dataclass(frozen=True)
class LogicalRow:
    row_index: int
    start_char: int
    end_char: int
    separator: bool
    cells: tuple[SourceCell, ...]


@dataclass(frozen=True)
class LogicalTable:
    start_char: int
    end_char: int
    rows: tuple[LogicalRow, ...]


@dataclass(frozen=True)
class LogicalTableEvidence:
    version: int
    coordinate_system: str
    table_start_char: int
    table_end_char: int
    table_sha256: str
    rows: tuple[LogicalRow, ...]
    context_rows: tuple[LogicalRow, ...]


def pipe_positions(text: str) -> list[int]:
    positions: list[int] = []
    escaped = False
    for index, character in enumerate(text):
        if character == "|" and not escaped:
            positions.append(index)
        escaped = not escaped if character == "\\" else False
    return positions


def _complete(line: str) -> bool:
    stripped = line.strip()
    pipes = pipe_positions(stripped)
    return bool(len(pipes) >= 2 and pipes[0] == 0 and pipes[-1] == len(stripped) - 1)


def _separator(line: str) -> bool:
    if not _complete(line):
        return False
    text = line.strip()
    pipes = pipe_positions(text)
    return all(
        _SEPARATOR.fullmatch(text[left + 1 : right])
        for left, right in zip(pipes, pipes[1:])
    )


def _row(text: str, start: int, end: int, index: int) -> LogicalRow:
    raw = text[start:end]
    leading = len(raw) - len(raw.lstrip())
    raw = raw.strip()
    start += leading
    end = start + len(raw)
    pipes = pipe_positions(raw)
    if not _complete(raw):
        raise ValueError("A multiline intake table has an incomplete logical row")
    return LogicalRow(
        row_index=index,
        start_char=start,
        end_char=end,
        separator=_separator(raw),
        cells=tuple(
            SourceCell(column, start + left + 1, start + right)
            for column, (left, right) in enumerate(zip(pipes, pipes[1:]))
        ),
    )


def scan_logical_tables(text: str) -> list[LogicalTable]:
    """Only a literal header/separator proves a table; preserve ragged group headers."""
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    offset = 0
    for line in lines:
        starts.append(offset)
        offset += len(line)
    tables: list[LogicalTable] = []
    index = 0
    fence: tuple[str, int] | None = None
    while index + 1 < len(lines):
        marker = re.match(r"^\s*(`{3,}|~{3,})", lines[index])
        if fence is not None:
            if (
                marker
                and marker[1][0] == fence[0]
                and len(marker[1]) >= fence[1]
                and not lines[index][marker.end() :].strip()
            ):
                fence = None
            index += 1
            continue
        if marker:
            fence = marker[1][0], len(marker[1])
            index += 1
            continue
        if not (_complete(lines[index]) and _separator(lines[index + 1])):
            index += 1
            continue
        first = index
        rows = [
            _row(text, starts[index], starts[index] + len(lines[index]), 0),
            _row(text, starts[index + 1], starts[index + 1] + len(lines[index + 1]), 1),
        ]
        index += 2
        row_index = 2
        while index < len(lines) and lines[index].strip().startswith("|"):
            row_start = index
            if _complete(lines[index]):
                row = _row(
                    text, starts[index], starts[index] + len(lines[index]), row_index
                )
                rows.append(row)
                row_index += 1
                index += 1
                continue
            # Keep cell continuations literal; ragged/merged cells are not padded.
            index += 1
            while (
                index < len(lines)
                and lines[index].strip()
                and not lines[index].strip().startswith("|")
            ):
                index += 1
            end = starts[index] if index < len(lines) else len(text)
            if not _complete(text[starts[row_start] : end]):
                raise ValueError(
                    "A multiline intake table row is incomplete or ambiguous"
                )
            row = _row(text, starts[row_start], end, row_index)
            rows.append(row)
            row_index += 1
        tables.append(LogicalTable(starts[first], rows[-1].end_char, tuple(rows)))
    return tables


def logical_markdown_blocks(text: str) -> tuple[list[SourceBlock], list[LogicalTable]]:
    """Replace only proven multiline spans; all bytes and ordinary blocks stay exact."""
    tables = scan_logical_tables(text)
    multiline = [
        table
        for table in tables
        if any("\n" in text[row.start_char : row.end_char] for row in table.rows)
    ]
    if not multiline:
        return _markdown_to_blocks(text), tables
    baseline = _markdown_to_blocks(text)
    blocks: list[SourceBlock] = []
    for block in baseline:
        cursor = block.start_char
        for table in multiline:
            if table.start_char >= block.end_char or table.end_char <= cursor:
                continue
            if cursor < table.start_char:
                raw = text[cursor : table.start_char]
                if raw.strip():
                    leading = len(raw) - len(raw.lstrip())
                    blocks.append(
                        replace(
                            block,
                            text=raw.strip(),
                            start_char=cursor + leading,
                            end_char=cursor + len(raw.rstrip()),
                        )
                    )
            cursor = max(cursor, table.end_char)
        if cursor < block.end_char:
            raw = text[cursor : block.end_char]
            if raw.strip():
                leading = len(raw) - len(raw.lstrip())
                blocks.append(
                    replace(
                        block,
                        text=raw.strip(),
                        start_char=cursor + leading,
                        end_char=cursor + len(raw.rstrip()),
                    )
                )
    blocks.extend(
        SourceBlock(
            block_id=0,
            kind="table",
            text=text[table.start_char : table.end_char],
            start_char=table.start_char,
            end_char=table.end_char,
        )
        for table in multiline
    )
    ordered = sorted(blocks, key=lambda block: block.start_char)
    return [
        replace(block, block_id=index) for index, block in enumerate(ordered)
    ], tables
