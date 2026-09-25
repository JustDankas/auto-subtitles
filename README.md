# Stream ASR

Real-time subtitles for any audio playing on your Windows PC. Runs offline on the CPU.

Stream ASR captures the sound your computer sends to its speakers or headphones, transcribes it as it plays, and shows the text in a floating caption window that stays on top of your video player, browser, or meeting app. It writes each finished line to a timestamped JSONL transcript.

All recognition happens on your machine. No audio leaves it.

## Use cases

- **Accessibility.** Add captions to livestreams, screen-shared meetings, and videos that ship without subtitles.
- **Low-volume viewing.** Follow a conference recording or a movie at low volume in a shared space.
- **Language practice.** Read English captions alongside spoken English.
- **Searchable transcripts.** Turn a lecture or a livestream into a JSONL file you can grep, parse, or feed into other tools.

## Features

- Captures system audio through WASAPI loopback, so it works with any player or app, with no plugins.
- Streaming recognition with a Nemotron transducer from [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Text appears while people are still speaking.
- Silero voice activity detection keeps silence and most music away from the recognizer.
- Optional speaker-change detection uses WeSpeaker CAM++ embeddings and z-score scoring to split caption lines when the speaker changes. It detects changes, but does not identify speakers by name.
- Translucent, draggable, always-on-top caption window. Each line fades out after a few seconds.
- Optional click-through mode (experimental) so clicks reach the app behind the captions.
- Per-line JSONL log with timestamp, text, confidence, and duration. The logger flushes after each line, so a crash does not lose finished lines.
- CPU inference. In a Phase 0 test the recognizer ran at a real-time factor of 0.158, which means it decodes audio about six times faster than it plays.

## Requirements

- Windows 10 or 11. Loopback capture uses WASAPI, so Linux and macOS do not work.
- Python 3.10 to 3.12 recommended. The code uses `X | None` type hints, which need 3.10. Python 3.13 removed the `audioop` module the capture code imports, so `requirements.txt` installs the `audioop-lts` backport there.
- About 1.5 GB of free disk space for the model download and its extracted files.
- No GPU.

## Installation

Open PowerShell and run these commands from the folder where you want the project.

**1. Clone the repository and create a virtual environment**

```powershell
git clone https://github.com/JustDankas/auto-subtitles.git
cd auto-subtitles
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow it for the current session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**2. Install the dependencies**

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `python src/app.py` later fails with `ImportError: DLL load failed while importing QtWidgets`, the PyQt6 wheel from PyPI is not compatible with your system. Use the conda route below instead.

**Alternative: conda environment**

Conda-forge ships Qt with its own compatible runtime libraries, which fixes the DLL error. Run these in Anaconda Prompt or Miniconda Prompt instead of steps 1 and 2 (clone the repository first):

```powershell
conda create -n stream-asr -c conda-forge python=3.12 numpy pyqt6
conda activate stream-asr
pip install sherpa-onnx PyAudioWPatch
```

Then continue with step 3. Activate the environment with `conda activate stream-asr` each time you open a new terminal.

**3. Download the models**

The app needs a streaming Nemotron speech recognition model, the Silero VAD model (under 1 MB), and the WeSpeaker CAM++ speaker recognition model. All models come from the sherpa-onnx release page.

```powershell
mkdir models
curl.exe -L -o models\silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
curl.exe -L -o models\wespeaker_en_voxceleb_CAM++.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM%2B%2B.onnx
curl.exe -L -o models\nemotron.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2
tar -xjf models\nemotron.tar.bz2 -C models
Remove-Item models\nemotron.tar.bz2
```

After extraction, the extracted `models\nemotron-en-0.6b-560ms-int8-2026-04-25` directory contains `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, and `tokens.txt`. The speaker model is saved as `models\wespeaker_en_voxceleb_CAM++.onnx`.

## Run

Using recommended default hyperparameters

```powershell
python src/app.py --asr-model-dir models\nemotron-en-0.6b-560ms-int8-2026-04-25
```

Start any audio on your default output device (a YouTube video works well). Captions appear in the overlay within a few seconds. Drag the bar labeled **Drag to Move** to reposition the captions, and click the **✕** on that bar to quit.

More examples:

```powershell
# Speaker-change detection with suggested hyperparameters
python src/app.py --asr-model-dir models\nemotron-en-0.6b-560ms-int8-2026-04-25 --vad-model models\silero_vad.onnx --int8 --speaker-model models\wespeaker_en_voxceleb_CAM++.onnx --speaker-k 1.5 --std-floor 0.125 --speaker-min-window 2 --speaker-max-window 15 --num-threads 4 --speaker-split-backdate-seconds 0.7 --speaker-split-mode token

# Raise the VAD threshold to reject background music, and write to a custom log file
python src/app.py --asr-model-dir models\nemotron-en-0.6b-560ms-int8-2026-04-25 --vad-model models\silero_vad.onnx --vad-threshold 0.5 --log-file lecture.jsonl

# With click-through and custom new-utterance text color
python src/app.py --asr-model-dir models\nemotron-en-0.6b-560ms-int8-2026-04-25 --vad-model models\silero_vad.onnx --click-through --new-text-color 00C3FF
```

### Command-line options

| Option                             | Default            | Description                                                                          |
| ---------------------------------- | ------------------ | ------------------------------------------------------------------------------------ |
| `--asr-model-dir`                  | required           | Folder containing the encoder, decoder, joiner, and `tokens.txt` files.              |
| `--vad-model`                      | `silero_vad.onnx`  | Path to the Silero VAD model.                                                        |
| `--provider`                       | `cpu`              | `cpu` or `cuda`. The CPU path is the tested one.                                     |
| `--int8`                           | off                | Use the int8-quantized model files when the folder has them.                         |
| `--log-file`                       | `transcript.jsonl` | Where finished lines are appended.                                                   |
| `--vad-threshold`                  | `0.15`             | Speech probability cutoff. Raise it to reject music, lower it to catch quiet speech. |
| `--min-silence`                    | `0.5`              | Seconds of silence before the VAD closes a speech region.                            |
| `--rule2-silence`                  | `0.5`              | Seconds of trailing silence that end a caption line.                                 |
| `--rule3-utterance`                | `12.0`             | Seconds of continuous speech after which a line break is forced.                     |
| `--speaker-model`                  | off                | Path to the WeSpeaker/CAM++ ONNX model. Enables speaker-change detection.            |
| `--speaker-k`                      | `1.5`              | Z-score threshold for declaring an embedding change.                                 |
| `--std-floor`                      | `0.125`            | Minimum similarity standard deviation used by the detector.                          |
| `--speaker-min-window`             | `2`                | Embeddings collected before change detection begins.                                 |
| `--speaker-max-window`             | `15`               | Maximum number of recent embeddings kept for comparison.                             |
| `--speaker-split-backdate-seconds` | `0.7`              | Backdate used to keep unsettled boundary words with the next line.                   |
| `--speaker-split-mode`             | `token`            | `token` uses ASR timestamps; `wallclock` uses a wall-clock approximation.            |
| `--new-text-color`                 | `#FFFF00`          | Color of the line currently being recognized.                                        |
| `--old-text-color`                 | `#E5E5E5`          | Color of finished lines.                                                             |
| `--width`, `--height`              | `900`, `160`       | Caption window size in pixels.                                                       |
| `--x`, `--y`                       | `100`, `100`       | Initial window position.                                                             |
| `--click-through`                  | off                | Let mouse clicks pass through the caption box. Experimental.                         |

## Output

Each finished line becomes one JSON object in the log file, and the same fields print to the console:

```json
{
  "timestamp": "2026-09-18T14:03:21.482913+00:00",
  "text": "We need to ship this by Friday",
  "confidence": 0.93,
  "duration_seconds": 3.84
}
```

Timestamps are UTC. `confidence` is derived from the recognizer's token probabilities and can be `null` when the installed sherpa-onnx version does not expose token probabilities. Follow the log live with `Get-Content transcript.jsonl -Wait`.

## Architecture

```
System audio (default output device)
        |
        v
LoopbackAudioCapture    PortAudio callback: downmix to mono, resample to 16 kHz float32
        |
        v
AudioRingBuffer         10 s circular buffer, overwrites the oldest audio if the consumer stalls
        |               drained every 100 ms by the pipeline thread
        v
SpeechGate              Silero VAD (sherpa-onnx, CPU), 512-sample windows
        |
        |-- no speech --> 0.3 s pre-roll buffer
        |
        |-- speech -----> StreamingAsrEngine    Nemotron transducer, greedy search, endpoint detection
                                |
                                |-- same speech audio --> SpeakerEmbeddingService (optional CAM++)
                                |                              |
                                |                              v
                                |                      InstantChangeDetector
                                |                      z-score over recent embeddings
                                |                              |
                                |                  speaker change --> UI/log line split
                                |
                                |-- partial text, throttled to 4 updates/s --> overlay (current line)
                                |-- finalized line --> text_formatter --> overlay + TranscriptLogger (JSONL)
```

Three threads run at once. PortAudio's callback thread captures audio. A `QThread` (`PipelineWorker`) runs the VAD, the recognizer, and the logger. The Qt main thread draws the overlay. The worker talks to the GUI through Qt signals and does not touch widgets directly.

### Modules

| File                   | Role                                                                                                  |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| `audio_capture.py`     | WASAPI loopback capture through PyAudioWPatch, plus downmix and resampling.                           |
| `ring_buffer.py`       | Thread-safe circular buffer with dropped-audio counters.                                              |
| `vad_gate.py`          | Wraps sherpa-onnx's Silero VAD and buffers arbitrary-length input into exact 512-sample windows.      |
| `asr_engine.py`        | Persistent `OnlineStream` with endpoint detection and a confidence estimate.                          |
| `pipeline_worker.py`   | The `QThread` that connects capture, VAD, ASR, and logging, and emits GUI signals.                    |
| `speaker_detector.py`  | Accumulates speech for CAM++ embeddings and detects changes with a rolling cosine-similarity z-score. |
| `text_formatter.py`    | Number formatter that handles years, fractions, thousands.                                            |
| `transcript_logger.py` | Console and JSONL logging, flushed per line.                                                          |
| `overlay_window.py`    | Caption window with per-line boxes that fade and shrink.                                              |
| `drag_handle.py`       | Separate always-interactive window for dragging and closing.                                          |
| `app.py`               | Entry point: parses arguments, wires signals, starts the worker.                                      |

### Design decisions

**The recognizer decides where lines end.** My first version used VAD segments as line boundaries, so latency depended on pauses. A speaker who never paused produced no text for a long time. The recognizer's own endpoint detector now decides, and `--rule3-utterance` forces a break after 12 seconds of continuous speech. That bounds latency at a few seconds even for an unbroken lecture.

**The VAD gate saves CPU and reduces phantom words.** The recognizer receives audio while the VAD reports speech, so silence and most music stay out of it.

**A pre-roll buffer protects the first word.** The VAD needs a short stretch of confirmed speech before it reports activity. The worker keeps the last 0.3 seconds of audio and replays it into the recognizer when speech starts, so the first word survives.

**Partial results are throttled.** The model revises its in-progress text many times per second. The worker emits a partial only when the text changed and at least 250 ms have passed, which keeps the overlay from flickering.

**The ring buffer drops old audio instead of blocking.** Blocking inside the audio callback causes glitches. When a consumer falls more than 10 seconds behind, the buffer overwrites the oldest samples and counts them in `stats`.

**The overlay uses two windows.** On Windows, click-through applies to a whole native window, so one window cannot be half click-through. The caption box can be click-through while the small drag handle stays interactive, which lets you move or close the app at any time.

**Speaker changes split the display without resetting ASR.** When `--speaker-model` is enabled, the worker feeds the same VAD-approved speech audio to the CAM++ embedding extractor. `InstantChangeDetector` compares each normalized embedding with a rolling window of recent embeddings. When the similarity z-score exceeds `--speaker-k`, the current caption line is finalized for display and logging, but the recognizer stream, endpoint detector, and audio overlap history continue uninterrupted. This avoids dropping words at a speaker boundary.

The split point is backdated by `--speaker-split-backdate-seconds` so unsettled decoder output and audio from the incoming speaker are not attached to the previous line. In `token` mode, the worker uses the recognizer's per-token timestamps and per-word confidence when available. If the installed sherpa-onnx build does not expose usable token data, it automatically falls back to the `wallclock` approximation. Speaker changes do not add speaker labels to the JSONL output.

## Tuning

| Symptom                                       | Try                                                                                                              |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| Captions appear for music or background noise | Raise `--vad-threshold` to 0.4 or 0.6.                                                                           |
| Quiet speech is missed                        | Lower `--vad-threshold` to 0.1 or 0.3.                                                                           |
| Captions lag behind the speaker               | Lower `--rule2-silence` and `--rule3-utterance`.                                                                 |
| Sentences split in the middle                 | Raise `--rule2-silence` and `--min-silence`.                                                                     |
| CPU usage is high                             | Add `--int8`.                                                                                                    |
| Speaker changes are missed                    | Lower `--speaker-k`, lower `--std-floor`, or increase `--speaker-max-window`.                                    |
| Speaker lines split too often                 | Raise `--speaker-k` or `--std-floor`; increase `--speaker-min-window`.                                           |
| Speaker split boundaries feel early or late   | Adjust `--speaker-split-backdate-seconds`; use `--speaker-split-mode token` when token timestamps are available. |

## Troubleshooting

**`ModuleNotFoundError: No module named 'sherpa_onnx'`**
Activate the virtual environment and run `pip install -r requirements.txt`.

**`ModuleNotFoundError: No module named 'audioop'`**
You are on Python 3.13 or later. Run `pip install audioop-lts` or use Python 3.12.

**`FileNotFoundError: Could not find model files`**
Point `--asr-model-dir` at the extracted folder that holds `tokens.txt`.

**The overlay shows nothing**
Loopback records the default output device. Set the device you are listening on as the Windows default, then restart the app. Watch the console: `[status] Listening: <device name>` confirms which device it opened.

**`ImportError: DLL load failed while importing QtWidgets`**
The PyQt6 wheel from PyPI cannot load its Qt libraries on some Windows setups. Switch to the conda environment described under Installation. Installing PyQt6 from conda-forge avoids the problem because conda supplies matching Qt and runtime DLLs. Installing the latest Microsoft Visual C++ Redistributable is the other fix to try first if you prefer to stay on venv.

**`--click-through` does nothing, or the caption window misbehaves**
This mode uses `Qt.WindowType.WindowTransparentForInput`, and I have not verified it across Windows and PyQt6 versions. Open an issue with your Windows and PyQt6 versions.

**`--provider cuda` fails**
The CUDA path is untested. I could not get it working on a GTX 1070, and CPU inference was fast enough that I stopped pursuing it.

**Speaker detection falls back from token mode**

The `token` mode requires a sherpa-onnx build that exposes per-token timestamps in the recognizer result, and a tokenizer whose word-boundary markers can be reconstructed reliably. When those conditions are not met, the app prints a notice and uses the wall-clock backdate automatically. This affects split precision, not ASR recognition.

## Limitations

- Windows only.
- English only. The bundled model is an English streaming Nemotron.
- No punctuation, and proper nouns come out in lowercase.
- Status and error messages print to the console and do not appear in the overlay.
- Changing the default audio device while the app runs requires a restart.

## Roadmap

- Punctuation and truecasing restoration with a small text model.
- Multilingual support through a multilingual streaming model or hot-swappable models.
- Config file for model paths, VAD sensitivity, and log location.
- In-app click-through toggle and a status indicator in the drag handle.
- Automatic recovery when the output device disappears.
- Packaged installer.

## Built with

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) for streaming ASR and VAD inference
- [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection
- [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) for WASAPI loopback capture
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the overlay
