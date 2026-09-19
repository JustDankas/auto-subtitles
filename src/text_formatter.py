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

_STANDALONE_I = re.compile(r"\bi\b")


def format_line(raw_text: str) -> str:
    if not raw_text:
        return raw_text
    text = raw_text.lower()
    text = text[0].upper() + text[1:]
    text = _STANDALONE_I.sub("I", text)
    return text
