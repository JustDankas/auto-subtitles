import argparse
from pathlib import Path

import soundfile as sf

MIN_CLIP_S = 1.2  # drop utterances too short to serve as a meaningful "turn"

def prepare(librispeech_subset_dir: str, out_dir: str, max_speakers: int = None,
            max_clips_per_speaker: int = None):
    src = Path(librispeech_subset_dir)
    out = Path(out_dir)
    speaker_dirs = sorted(p for p in src.iterdir() if p.is_dir())
    if max_speakers:
        speaker_dirs = speaker_dirs[:max_speakers]

    for spk_dir in speaker_dirs:
        flacs = sorted(spk_dir.rglob("*.flac"))
        if max_clips_per_speaker:
            flacs = flacs[:max_clips_per_speaker]
        spk_out = out / spk_dir.name
        spk_out.mkdir(parents=True, exist_ok=True)
        kept = 0
        for flac_path in flacs:
            audio, sr = sf.read(str(flac_path), dtype="float32")
            if len(audio) / sr < MIN_CLIP_S:
                continue
            sf.write(str(spk_out / f"{flac_path.stem}.wav"), audio, sr)  # already 16kHz
            kept += 1
        print(f"{spk_dir.name}: kept {kept}/{len(flacs)} clips")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--max-clips-per-speaker", type=int, default=None)
    args = ap.parse_args()
    prepare(args.librispeech_dir, args.out_dir, args.max_speakers, args.max_clips_per_speaker)
