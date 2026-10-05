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
- Optional speaker-change detection uses a PyTorch SCDNet checkpoint to split caption lines when the speaker changes. It detects changes, but does not identify speakers by name.
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
pip install sherpa-onnx PyAudioWPatch torch torchaudio
```

Then continue with step 3. Activate the environment with `conda activate stream-asr` each time you open a new terminal.

**3. Download the models**

The app needs a streaming Nemotron speech recognition model and the Silero VAD model (under 1 MB). If you want speaker-change line splitting, also provide a trained SCDNet PyTorch checkpoint (`.pt` file).

```powershell
mkdir models
curl.exe -L -o models\silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
# 1. Download the archive
curl.exe -L -o models\nemotron.tar.bz2 https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25.tar.bz2

# 2. Create your custom target directory inside models
New-Item -ItemType Directory -Force -Path models\nemotron-en-0.6b-560ms-int8-2026-04-25

# 3. Extract contents directly into your custom folder
tar -xjf models\nemotron.tar.bz2 -C models\nemotron-en-0.6b-560ms-int8-2026-04-25 --strip-components=1

# 4. Remove the archive
Remove-Item models\nemotron.tar.bz2
```

After extraction, the `models\nemotron-en-0.6b-560ms-int8-2026-04-25` directory contains `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, and `tokens.txt`. SCD is disabled unless you pass a trained checkpoint with `--scd-checkpoint`.

## Run

Using recommended default hyperparameters

```powershell
python src/app.py
```

Start any audio on your default output device (a YouTube video works well). Captions appear in the overlay within a few seconds. Drag the bar labeled **Drag to Move** to reposition the captions, and click the **✕** on that bar to quit.

More examples:

```powershell
# Speaker-change detection with suggested hyperparameters
python src/app.py --scd-checkpoint models\scdnet_tcn_20261005_best.pt --scd-device cpu --scd-threshold 0.3

# Raise the VAD threshold to reject background music, and write to a custom log file
python src/app.py --vad-threshold 0.5 --log-file lecture.jsonl

# With click-through and custom new-utterance text color
python src/app.py --click-through --new-text-color 00C3FF
```

### Command-line options

| Option                | Default            | Description                                                                          |
| --------------------- | ------------------ | ------------------------------------------------------------------------------------ |
| `--asr-model-dir`     | required           | Folder containing the encoder, decoder, joiner, and `tokens.txt` files.              |
| `--vad-model`         | `silero_vad.onnx`  | Path to the Silero VAD model.                                                        |
| `--provider`          | `cpu`              | `cpu` or `cuda`. The CPU path is the tested one.                                     |
| `--int8`              | off                | Use the int8-quantized model files when the folder has them.                         |
| `--log-file`          | `transcript.jsonl` | Where finished lines are appended.                                                   |
| `--vad-threshold`     | `0.15`             | Speech probability cutoff. Raise it to reject music, lower it to catch quiet speech. |
| `--min-silence`       | `0.5`              | Seconds of silence before the VAD closes a speech region.                            |
| `--rule2-silence`     | `0.5`              | Seconds of trailing silence that end a caption line.                                 |
| `--rule3-utterance`   | `12.0`             | Seconds of continuous speech after which a line break is forced.                     |
| `--scd-checkpoint`    | off                | Path to a trained SCDNet PyTorch checkpoint (`.pt`). Enables speaker-change splits.  |
| `--scd-threshold`     | `0.5`              | Smoothed change probability required to report a change.                             |
| `--scd-hop`           | `0.25`             | Seconds of new speech audio between SCD inference windows.                           |
| `--scd-left-guard`    | `0.5`              | Ignore candidate changes this close to the start of each 3-second window.            |
| `--scd-right-guard`   | `1.0`              | Ignore candidate changes this close to the end of each window.                       |
| `--scd-min-gap`       | `1.0`              | Minimum seconds between reported changes.                                            |
| `--scd-reset-gap`     | `1.5`              | Clear the SCD audio window after this much non-speech.                               |
| `--scd-settle`        | `0.1`              | Extra wait after the ASR decode lag before splitting a line.                         |
| `--scd-device`        | `cpu`              | PyTorch device for SCD inference (`cpu` or `cuda`), independent of `--provider`.     |
| `--scd-threads`       | `1`                | Number of CPU threads used by SCD inference.                                         |
| `--scd-debug-dump`    | off                | Append per-window probabilities to this JSONL path for offline tuning.               |
| `--asr-decode-lag`    | `0.8`              | Estimated seconds from spoken word to ASR partial; used to align SCD splits.         |
| `--new-text-color`    | `#FFFF00`          | Color of the line currently being recognized.                                        |
| `--old-text-color`    | `#E5E5E5`          | Color of finished lines.                                                             |
| `--width`, `--height` | `900`, `160`       | Caption window size in pixels.                                                       |
| `--x`, `--y`          | `100`, `100`       | Initial window position.                                                             |
| `--click-through`     | off                | Let mouse clicks pass through the caption box. Experimental.                         |

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
        |                      |
        |                      |-- partial text, throttled to 4 updates/s --> overlay (current line)
        |                      |-- finalized line --> text_formatter --> overlay + TranscriptLogger (JSONL)
        |
        |-- same speech audio --> StreamingSpeakerChangeDetector (optional PyTorch SCDNet)
                                       |
                                       | 3 s windows, evaluated every 0.25 s
                                       | smoothed change probabilities + peak selection
                                       v
                                speaker change --> UI/log line split (ASR stream unchanged)
```

Three threads run at once. PortAudio's callback thread captures audio. A `QThread` (`PipelineWorker`) runs the VAD, the recognizer, and the logger. The Qt main thread draws the overlay. The worker talks to the GUI through Qt signals and does not touch widgets directly.

### Modules

| File                   | Role                                                                                               |
| ---------------------- | -------------------------------------------------------------------------------------------------- |
| `audio_capture.py`     | WASAPI loopback capture through PyAudioWPatch, plus downmix and resampling.                        |
| `ring_buffer.py`       | Thread-safe circular buffer with dropped-audio counters.                                           |
| `vad_gate.py`          | Wraps sherpa-onnx's Silero VAD and buffers arbitrary-length input into exact 512-sample windows.   |
| `asr_engine.py`        | Persistent `OnlineStream` with endpoint detection and a confidence estimate.                       |
| `pipeline_worker.py`   | The `QThread` that connects capture, VAD, ASR, optional SCD, and logging, and emits GUI signals.   |
| `scd_detector.py`      | Runs sliding-window SCD inference and converts model probabilities into audio-clock change events. |
| `scd_model.py`         | PyTorch SCDNet model and log-mel feature extraction used by the detector.                          |
| `text_formatter.py`    | Number formatter that handles years, fractions, thousands.                                         |
| `transcript_logger.py` | Console and JSONL logging, flushed per line.                                                       |
| `overlay_window.py`    | Caption window with per-line boxes that fade and shrink.                                           |
| `drag_handle.py`       | Separate always-interactive window for dragging and closing.                                       |
| `app.py`               | Entry point: parses arguments, wires signals, starts the worker.                                   |

### Design decisions

**The recognizer decides where lines end.** My first version used VAD segments as line boundaries, so latency depended on pauses. A speaker who never paused produced no text for a long time. The recognizer's own endpoint detector now decides, and `--rule3-utterance` forces a break after 12 seconds of continuous speech. That bounds latency at a few seconds even for an unbroken lecture.

**The VAD gate saves CPU and reduces phantom words.** The recognizer receives audio while the VAD reports speech, so silence and most music stay out of it.

**A pre-roll buffer protects the first word.** The VAD needs a short stretch of confirmed speech before it reports activity. The worker keeps the last 0.3 seconds of audio and replays it into the recognizer when speech starts, so the first word survives.

**Partial results are throttled.** The model revises its in-progress text many times per second. The worker emits a partial only when the text changed and at least 250 ms have passed, which keeps the overlay from flickering.

**The ring buffer drops old audio instead of blocking.** Blocking inside the audio callback causes glitches. When a consumer falls more than 10 seconds behind, the buffer overwrites the oldest samples and counts them in `stats`.

**The overlay uses two windows.** On Windows, click-through applies to a whole native window, so one window cannot be half click-through. The caption box can be click-through while the small drag handle stays interactive, which lets you move or close the app at any time.

**SCDNet splits the display without resetting ASR.** When `--scd-checkpoint` is set, the worker feeds the same VAD-approved speech audio to a PyTorch SCDNet model in overlapping 3-second windows. Smoothed per-frame change probabilities are thresholded and peak-selected; detected changes are mapped onto the audio clock and aligned with ASR partial word counts using `--asr-decode-lag`. A split finalizes the current display/log line, but leaves the recognizer stream and endpoint detector untouched. SCD detects changes, not speaker identities, so no speaker labels are added to the JSONL output.

## Tuning

| Symptom                                       | Try                                                                                          |
| --------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Captions appear for music or background noise | Raise `--vad-threshold` to 0.4 or 0.6.                                                       |
| Quiet speech is missed                        | Lower `--vad-threshold` to 0.1 or 0.3.                                                       |
| Captions lag behind the speaker               | Lower `--rule2-silence` and `--rule3-utterance`.                                             |
| Sentences split in the middle                 | Raise `--rule2-silence` and `--min-silence`.                                                 |
| CPU usage is high                             | Add `--int8`.                                                                                |
| Speaker changes are missed                    | Lower `--scd-threshold`; check that a checkpoint is supplied with `--scd-checkpoint`.        |
| Speaker lines split too often                 | Raise `--scd-threshold` or `--scd-min-gap`.                                                  |
| Speaker split boundaries feel early or late   | Tune `--scd-right-guard` and `--asr-decode-lag`; both affect the alignment/latency tradeoff. |

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

**Speaker-change detection does not start**

SCD is disabled by default. Pass `--scd-checkpoint` with a compatible trained SCDNet `.pt` checkpoint. PyTorch loads the checkpoint on the selected `--scd-device` (`cpu` by default).

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
- [PyTorch](https://pytorch.org/) and torchaudio for SCDNet speaker-change inference and audio features
- [Silero VAD](https://github.com/snakers4/silero-vad) for voice activity detection
- [PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) for WASAPI loopback capture
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) for the overlay
