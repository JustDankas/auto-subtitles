import re

from text_to_num import alpha2digit

_STANDALONE_I = re.compile(r"\bi\b")
_YEAR_PAIR_AFTER_CUE = re.compile(
    r"\b(in|since|by|year)\s+(\d{2})\s+(\d{2})\b"
)

# Matches exactly two 2-digit numbers separated by a space,
# ensuring no adjacent digits before or after the pair.
_TWO_ISOLATED_2DIGIT_NUMS = re.compile(
    r"(?<!\d)\b(\d{2})\s+(\d{2})\b(?!\s*\d)"
)


def _combine_cued_years(text: str) -> str:
    return _YEAR_PAIR_AFTER_CUE.sub(
        lambda match: f"{match.group(1)} {match.group(2)}{match.group(3)}",
        text,
    )

def _combine_isolated_2digit_pairs(text: str) -> str:
    return _TWO_ISOLATED_2DIGIT_NUMS.sub(r"\1\2", text)


def format_line(
    raw_text: str,
    numbers: bool = True,
    number_threshold: float = 3.0,
) -> str:
    if not raw_text:
        return raw_text
    # text = raw_text.lower()
    text = raw_text
    if numbers:
        text = alpha2digit(text, "en", threshold=number_threshold)
        # text = _combine_cued_years(text)
        text = _combine_isolated_2digit_pairs(text)
    text = text[0].upper() + text[1:]
    text = _STANDALONE_I.sub("I", text)
    return text
