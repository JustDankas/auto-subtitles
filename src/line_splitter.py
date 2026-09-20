"""Pure text wrapping helpers for the subtitle overlay."""

from typing import Callable


def _split_long_word(word: str, measure: Callable[[str], int], max_width_px: int) -> list[str]:
    parts: list[str] = []
    current = ""
    for character in word:
        candidate = current + character
        if current and measure(candidate) > max_width_px:
            parts.append(current)
            current = character
        else:
            current = candidate
    if current:
        parts.append(current)
    return parts or [word]


def _wrap_lines(text: str, measure: Callable[[str], int], max_width_px: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    lines: list[str] = []
    current = ""
    for word in words:
        if measure(word) > max_width_px:
            pieces = _split_long_word(word, measure, max_width_px)
            if current:
                lines.append(current)
                current = ""
            lines.extend(pieces[:-1])
            current = pieces[-1]
            continue

        candidate = word if not current else f"{current} {word}"
        if current and measure(candidate) > max_width_px:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def split_into_chunks(
    text: str,
    measure: Callable[[str], int],
    max_width_px: int,
    max_lines: int = 2,
) -> list[str]:
    """Return chunks that each render in at most ``max_lines`` lines."""
    if not text.strip():
        return []
    if max_width_px < 1:
        raise ValueError("max_width_px must be positive")
    if max_lines < 1:
        raise ValueError("max_lines must be positive")

    wrapped_lines = _wrap_lines(text, measure, max_width_px)
    return [
        "\n".join(wrapped_lines[index:index + max_lines])
        for index in range(0, len(wrapped_lines), max_lines)
    ]