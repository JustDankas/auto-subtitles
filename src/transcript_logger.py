"""
Structured logging for transcribed segments: pretty-printed to console AND
appended to a JSONL file, both carrying timestamp, text, confidence, and
duration - per the earlier decision to log all four fields in both places.

Flushes after every line so an ungraceful exit (crash, kill, power loss)
doesn't lose already-transcribed text.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class TranscriptLogger:
    def __init__(self, jsonl_path: str):
        self.path = Path(jsonl_path)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(
        self,
        text: str,
        confidence: Optional[float],
        duration: float,
        start_seconds: Optional[float] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "text": text,
            "confidence": confidence,  # may be null - see asr_engine.py note
            "duration_seconds": round(duration, 3),
        }
        if start_seconds is not None:
            entry["segment_start_seconds"] = round(start_seconds, 3)

        self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._fh.flush()

        conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
        print(f"[{entry['timestamp']}] (conf={conf_str}, dur={duration:.2f}s) {text}")

    def close(self) -> None:
        self._fh.close()
