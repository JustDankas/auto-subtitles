"""
Usage:
    python app.py --asr-model-dir sherpa-onnx-streaming-zipformer-en-2023-06-21 --vad-model silero_vad.onnx --provider cpu --log-file transcript.jsonl
    python app.py --scd-checkpoint models\\scd.pt          (enable speaker-change line splits)
    python app.py --asr-model-dir ... --click-through       (test click-through mode)
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
    parser.add_argument("--vad-model", type=str, default="models\\silero_vad.onnx")
    parser.add_argument("--asr-model-dir", type=str, default="models\\nemotron-en-0.6b-560ms-int8-2026-04-25")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--provider", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--num-threads", type=int, default=3)
    parser.add_argument("--log-file", type=str, default="logs\\transcript.jsonl")
    parser.add_argument("--vad-threshold", type=float, default=0.2)
    parser.add_argument("--min-silence", type=float, default=0.5, help="Seconds of silence to consider a line ended (default: 0.5)")
    parser.add_argument("--rule2-silence", type=float, default=1.2, help="Seconds of trailing silence before finalizing (default: 1.2)")
    parser.add_argument("--rule3-utterance", type=float, default=20.0, help="Seconds of continuous speech before force-finalizing (default: 12.0)")
    parser.add_argument("--overlap-seconds", type=float, default=1.0, help="Seconds of audio overlap to feed into the next line when rule3 is triggered (default: 1.0)")
    # Speaker change detection (SCDNet) options
    parser.add_argument("--scd-checkpoint", type=str, default=None, help="Path to the SCDNet checkpoint (.pt). Leave empty to disable speaker-change line splitting.")
    parser.add_argument("--scd-threshold", type=float, default=0.5, help="Smoothed change probability needed to fire (default: 0.6; tune at stream level, expect it to need to be higher than the training-optimal value)")
    parser.add_argument("--scd-hop", type=float, default=0.25, help="Seconds of new audio between SCD inferences (default: 0.25)")
    parser.add_argument("--scd-left-guard", type=float, default=0.5, help="Ignore peaks in the first N seconds of each 3 s window. The model never saw changes closer than 0.5 s to an edge (default: 0.5)")
    parser.add_argument("--scd-right-guard", type=float, default=0.5, help="Ignore peaks in the last N seconds of the window; also ~the detection latency (default: 0.5)")
    parser.add_argument("--scd-min-gap", type=float, default=1.0, help="Minimum seconds between two reported changes (default: 1.0)")
    parser.add_argument("--scd-reset-gap", type=float, default=1.5, help="Seconds of non-speech after which the SCD audio window is cleared (default: 1.5)")
    parser.add_argument("--scd-settle", type=float, default=0.1, help="Extra seconds to wait after asr-decode-lag before splitting a line (default: 0.1)")
    parser.add_argument("--scd-device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device for SCD inference, independent of --provider (default: cpu)")
    parser.add_argument("--scd-threads", type=int, default=1, help="Torch CPU threads for SCD inference (default: 1)")
    parser.add_argument("--scd-debug-dump", type=str, default=None, help="Append per-window smoothed probabilities to this .jsonl file for offline retuning")
    parser.add_argument("--asr-decode-lag", type=float, default=0.8, help="Seconds between a word being spoken and appearing in the ASR partial result (default: 0.8; calibrate per plan Phase 1b)")
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
    parser.add_argument("--new-text-color", type=str, default="FFFF00")
    parser.add_argument("--old-text-color", type=str, default="E5E5E5")
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
        new_text_color=f"#{args.new_text_color}",
        old_text_color=f"#{args.old_text_color}",
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
        scd_checkpoint=args.scd_checkpoint,
        scd_device=args.scd_device,
        scd_threads=args.scd_threads,
        scd_threshold=args.scd_threshold,
        scd_hop=args.scd_hop,
        scd_left_guard=args.scd_left_guard,
        scd_right_guard=args.scd_right_guard,
        scd_min_gap=args.scd_min_gap,
        scd_reset_gap=args.scd_reset_gap,
        scd_settle=args.scd_settle,
        scd_debug_dump=args.scd_debug_dump,
        asr_decode_lag=args.asr_decode_lag,
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
