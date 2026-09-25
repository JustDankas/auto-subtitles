"""
Phase 7: split-window overlay.

Two top-level windows moved together:
  - DragHandle: small always-on-top, always-interactive bar with a grip and
    the close button. This is how you drag and how you exit, always.
  - SubtitleOverlay: the caption box below it, showing per-line fading/
    shrinking boxes. Optionally click-through via --click-through (a real
    toggle button in the GUI itself is a natural next step, out of scope
    for this pass).

Usage:
    python 06_subtitle_overlay_app.py --asr-model-dir sherpa-onnx-streaming-zipformer-en-2023-06-21 --vad-model silero_vad.onnx --provider cpu --log-file transcript.jsonl
    python 06_subtitle_overlay_app.py --asr-model-dir ... --click-through   (test click-through mode)
"""

import argparse
import sys

from PyQt6.QtWidgets import QApplication

from drag_handle import DragHandle
from overlay_window import SubtitleOverlay
from pipeline_worker import PipelineWorker

HANDLE_OVERLAP = 12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    # ASR options
    parser.add_argument("--vad-model", type=str, default="silero_vad.onnx")
    parser.add_argument("--asr-model-dir", type=str, required=True)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--provider", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-threads", type=int, default=3)
    parser.add_argument("--log-file", type=str, default="transcript.jsonl")
    parser.add_argument("--vad-threshold", type=float, default=0.2)
    parser.add_argument("--min-silence", type=float, default=1.2, help="Seconds of silence to consider a line ended (default: 0.5)")
    parser.add_argument("--rule2-silence", type=float, default=1.2, help="Seconds of trailing silence before finalizing (default: 1.2)")
    parser.add_argument("--rule3-utterance", type=float, default=12.0, help="Seconds of continuous speech before force-finalizing (default: 20.0)")
    parser.add_argument("--overlap-seconds", type=float, default=1.0, help="Seconds of audio overlap to feed into the next line when rule3 is triggered (default: 1.0)")
    # Speaker detection options
    parser.add_argument("--speaker-model", type=str, default=None, help="Path to wespeaker/CAM++ ONNX model. Leave this blank to disable speaker detection.")
    parser.add_argument("--speaker-k", type=float, default=2.5, help="Z-score change detection threshold")
    parser.add_argument("--std-floor", type=float, default=0.05, help="Minimum standard deviation floor")
    parser.add_argument("--speaker-min-window", type=int, default=3, help="Min window before detection begins")
    parser.add_argument("--speaker-max-window", type=int, default=20, help="Max history window size")
    parser.add_argument("--speaker-split-backdate-seconds", type=float, default=0.7, help="Backdate for speaker change detection (default: 0.4)")
    # Number formatting options
    parser.add_argument(
        "--numbers",
        choices=["on", "off"],
        default="on",
        help="Convert spoken numbers to digits (default: on)",
    )
    parser.add_argument(
        "--number-threshold",
        type=float,
        default=3.0,
        help="Minimum number value to convert (default: 3)",
    )
    # GUI options
    parser.add_argument("--new-text-color", type=str, default="#FFFF00")
    parser.add_argument("--old-text-color", type=str, default="#E5E5E5")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument(
        "--click-through",
        action="store_true",
        help="Make the caption box ignore mouse clicks (passes through to whatever's behind it). "
        "The drag handle above it always stays interactive regardless.",
    )
    parser.add_argument("--x", type=int, default=None, help="Initial X position")
    parser.add_argument("--y", type=int, default=None, help="Initial Y position")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)

    # Calculate default positions if not specified
    screen = app.primaryScreen().geometry()

    if args.x is None:
        # Middle of the X axis
        start_x = (screen.width() - args.width) // 2
    else:
        start_x = args.x

    if args.y is None:
        # 1/3 from the bottom of the screen (2/3 down from the top)
        start_y = int(screen.height() * (2 / 3)) - (args.height // 2)
    else:
        start_y = args.y

    handle = DragHandle(width=args.width)
    overlay = SubtitleOverlay(
        new_text_color=args.new_text_color,
        old_text_color=args.old_text_color,
        width=args.width,
        height=args.height,
        click_through=args.click_through,
    )

    overlay.move(start_x, start_y)
    handle.move(
        start_x,
        start_y - handle.height() // 2 + HANDLE_OVERLAP,
    )
    overlay.set_drag_handle(handle)

    def on_handle_moved(dx: int, dy: int) -> None:
        overlay.move(overlay.x() + dx, overlay.y() + dy)

    handle.moved.connect(on_handle_moved)

    worker = PipelineWorker(
        vad_model_path=args.vad_model,
        asr_model_dir=args.asr_model_dir,
        log_file=args.log_file,
        speaker_model_path=args.speaker_model,
        speaker_k=args.speaker_k,
        speaker_min_window=args.speaker_min_window,
        std_floor=args.std_floor,
        max_window=args.speaker_max_window,
        provider=args.provider,
        int8=args.int8,
        vad_threshold=args.vad_threshold,
        min_silence=args.min_silence,
        rule2_min_trailing_silence=args.rule2_silence,
        rule3_min_utterance_length=args.rule3_utterance,
        numbers=args.numbers == "on",
        number_threshold=args.number_threshold,
        num_threads=args.num_threads,
        overlap_seconds=args.overlap_seconds,
    )

    worker.partial_updated.connect(overlay.update_partial)
    worker.line_finalized.connect(overlay.commit_line)
    # No visible status label in this layout yet - errors still land in the
    # console via traceback.print_exc() in pipeline_worker.py.
    worker.status_changed.connect(lambda msg: print(f"[status] {msg}"))
    worker.error_occurred.connect(lambda msg: print(f"[error] {msg}"))

    def shutdown() -> None:
        worker.stop()
        worker.wait(3000)
        handle.close()
        overlay.close()
        QApplication.instance().quit()

    handle.close_requested.connect(shutdown)

    overlay.show()
    handle.show()
    worker.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
