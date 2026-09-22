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

# Matches any digit followed by groups of 3 digits
_THOUSANDS_SEPARATOR = re.compile(r"(\d)(?=(\d{3})+(?!\d))")

# Matches a digit sequence, the word "point", and subsequent space-separated single/multiple digits
_DECIMAL_PATTERN = re.compile(r"\b(\d+)\s+point\s+((?:\d+\s*)+)\b", re.IGNORECASE)

def _combine_cued_years(text: str) -> str:
    return _YEAR_PAIR_AFTER_CUE.sub(
        lambda match: f"{match.group(1)} {match.group(2)}{match.group(3)}",
        text,
    )

def _combine_isolated_2digit_pairs(text: str) -> str:
    return _TWO_ISOLATED_2DIGIT_NUMS.sub(r"\1\2", text)

def _add_thousands_separators(text: str) -> str:
    return _THOUSANDS_SEPARATOR.sub(r"\1,", text)

def _format_decimals(text: str) -> str:
    def replace_decimal(match):
        integer_part = match.group(1)
        # Remove spaces between fractional digits (e.g., "7 8" -> "78")
        fractional_part = re.sub(r"\s+", "", match.group(2))
        return f"{integer_part}.{fractional_part}"

    return _DECIMAL_PATTERN.sub(replace_decimal, text)

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
        text = _format_decimals(text)
        # text = _combine_cued_years(text)
        text = _combine_isolated_2digit_pairs(text)
        text = _add_thousands_separators(text)
    text = text[0].upper() + text[1:]
    text = _STANDALONE_I.sub("I", text)
    return text
