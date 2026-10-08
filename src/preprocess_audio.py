"""Convert curated recordings into fixed-duration, normalized WAV clips."""

import argparse
import hashlib
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from config import (
    CLIP_DURATION_SECONDS,
    CLIP_SAMPLE_COUNT,
    FILTERED_DIR,
    METADATA_FILENAME,
    PREPROCESSED_DIR,
    PREPROCESSED_METADATA,
    PROJECT_ROOT,
    SAMPLE_RATE,
)


REQUIRED_COLUMNS = {"file_path", "source_dataset", "original_class", "binary_label", "scenario"}


def recording_identifier(source_dataset: str, file_path: str) -> str:
    identity = f"{source_dataset}:{file_path}".encode("utf-8")
    return hashlib.sha256(identity).hexdigest()[:16]


def load_standardized_audio(file_path: Path) -> np.ndarray:
    audio, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if audio.size > CLIP_SAMPLE_COUNT:
        start = (audio.size - CLIP_SAMPLE_COUNT) // 2
        audio = audio[start : start + CLIP_SAMPLE_COUNT]
    elif audio.size < CLIP_SAMPLE_COUNT:
        audio = np.pad(audio, (0, CLIP_SAMPLE_COUNT - audio.size))

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 0.0:
        audio = audio / peak
    return audio.astype(np.float32, copy=False)


def relative_to_project(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def preprocess_dataset(
    metadata_path: Path,
    output_dir: Path,
    output_metadata: Path,
    limit: int | None = None,
) -> Path:
    metadata = pd.read_csv(metadata_path)
    missing = REQUIRED_COLUMNS.difference(metadata.columns)
    if missing:
        raise ValueError(f"Input metadata is missing columns: {sorted(missing)}")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer")
        metadata = metadata.head(limit)
    if metadata.empty:
        raise ValueError("Input metadata contains no recordings")

    output_records: list[dict[str, object]] = []
    for _, row in metadata.iterrows():
        original_path = str(row["file_path"])
        source_path = Path(original_path)
        if not source_path.is_absolute():
            source_path = PROJECT_ROOT / source_path
        if not source_path.is_file():
            raise FileNotFoundError(f"Recording listed in metadata not found: {source_path}")

        dataset_name = str(row["source_dataset"])
        class_name = str(row["original_class"])
        identifier = recording_identifier(dataset_name, original_path)
        target_path = output_dir / dataset_name / class_name / f"{identifier}.wav"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        audio = load_standardized_audio(source_path)
        sf.write(target_path, audio, SAMPLE_RATE, subtype="PCM_16")

        record = row.to_dict()
        record.update(
            {
                "original_file_path": original_path,
                "file_path": relative_to_project(target_path),
                "recording_id": identifier,
                "sample_rate": SAMPLE_RATE,
                "duration_seconds": CLIP_DURATION_SECONDS,
            }
        )
        output_records.append(record)

    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(output_records).to_csv(output_metadata, index=False)
    return output_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=FILTERED_DIR / METADATA_FILENAME)
    parser.add_argument("--output-dir", type=Path, default=PREPROCESSED_DIR)
    parser.add_argument("--output-metadata", type=Path, default=PREPROCESSED_METADATA)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_path = preprocess_dataset(
        args.metadata, args.output_dir, args.output_metadata, args.limit
    )
    result = pd.read_csv(result_path)
    print(f"Wrote {len(result)} standardized recordings to {result_path}")


if __name__ == "__main__":
    main()