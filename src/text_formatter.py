"""
Lightweight capitalization heuristic for ASR output.

IMPORTANT LIMITATION: the streaming Zipformer model outputs ALL-CAPS,
unpunctuated text by training convention (standard for icefall/Kaldi
LibriSpeech-style recipes) - it was never trained to predict casing,
punctuation, or sentence boundaries, so there is no signal from the model
itself to read this from. Every line break you see is OUR OWN endpoint
detector's decision (rule2/rule3 in asr_engine.py), not the model
recognizing a sentence boundary.

This module is a cheap heuristic that gets most of the visual improvement
for zero new dependencies: lowercase everything, capitalize the first
letter of each line (valid because every line IS one of our own endpoint
boundaries), and capitalize the standalone pronoun "I". It will NOT
capitalize proper nouns, add real punctuation, or handle mid-sentence
capitalization correctly - that needs a separate punctuation/truecasing
restoration model as a future enhancement.
"""

import re

from text_to_num import alpha2digit

_STANDALONE_I = re.compile(r"\bi\b")
_YEAR_PAIR_AFTER_CUE = re.compile(
    r"\b(in|since|by|year)\s+(\d{2})\s+(\d{2})\b"
)


def _combine_cued_years(text: str) -> str:
    return _YEAR_PAIR_AFTER_CUE.sub(
        lambda match: f"{match.group(1)} {match.group(2)}{match.group(3)}",
        text,
    )


def format_line(
    raw_text: str,
    numbers: bool = True,
    number_threshold: float = 3.0,
) -> str:
    if not raw_text:
        return raw_text
    text = raw_text.lower()
    if numbers:
        text = alpha2digit(text, "en", threshold=number_threshold)
        text = _combine_cued_years(text)
    text = text[0].upper() + text[1:]
    text = _STANDALONE_I.sub("I", text)
    return text
