"""Standardize curated audio and extract compact candidate features."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from config import (
    CLIP_SAMPLE_COUNT,
    FEATURES_DIR,
    FEATURES_FILENAME,
    FILTERED_DIR,
    METADATA_FILENAME,
    PROJECT_ROOT,
    SAMPLE_RATE,
)


FEATURE_COLUMNS = [
    "mfcc_1_mean",
    "mfcc_2_mean",
    "mfcc_3_mean",
    "spectral_centroid_mean",
    "spectral_bandwidth_mean",
    "zero_crossing_rate_mean",
    "rms_energy_mean",
    "spectral_rolloff_mean",
    "spectral_flatness_mean",
]


def standardize_clip(file_path: Path) -> np.ndarray:
    """Load mono audio, resample, center-crop/pad to 3 seconds, and peak-normalize."""
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


def split_windows(audio: np.ndarray) -> list[np.ndarray]:
    """Cut audio into non-overlapping, peak-normalized windows without zero-padding.

    A trailing remainder shorter than one window is dropped. A recording shorter
    than one window is kept whole (unpadded) so it still contributes one example.
    Training (prepare_dataset.py) and the DETECT page both use this function.
    """
    if not audio.size:
        return []
    starts = range(0, audio.size - CLIP_SAMPLE_COUNT + 1, CLIP_SAMPLE_COUNT)
    windows = [audio[start : start + CLIP_SAMPLE_COUNT] for start in starts] or [audio]
    normalized = []
    for window in windows:
        peak = float(np.max(np.abs(window)))
        if peak > 0.0:
            window = window / peak
        normalized.append(window.astype(np.float32, copy=False))
    return normalized


def extract_features(audio: np.ndarray) -> dict[str, float]:
    """Calculate interpretable clip-level means for ranking and later selection."""
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=3)
    features = {
        "mfcc_1_mean": float(np.mean(mfcc[0])),
        "mfcc_2_mean": float(np.mean(mfcc[1])),
        "mfcc_3_mean": float(np.mean(mfcc[2])),
        "spectral_centroid_mean": float(
            np.mean(librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE))
        ),
        "spectral_bandwidth_mean": float(
            np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE))
        ),
        "zero_crossing_rate_mean": float(
            np.mean(librosa.feature.zero_crossing_rate(y=audio))
        ),
        "rms_energy_mean": float(np.mean(librosa.feature.rms(y=audio))),
        "spectral_rolloff_mean": float(
            np.mean(librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE))
        ),
        "spectral_flatness_mean": float(
            np.mean(librosa.feature.spectral_flatness(y=audio))
        ),
    }
    return features


def extract_dataset(
    metadata_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> Path:
    metadata = pd.read_csv(metadata_path)
    required = {"file_path", "source_dataset", "original_class", "binary_label", "scenario"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Input metadata is missing columns: {sorted(missing)}")
    if limit is not None:
        metadata = metadata.head(limit)

    feature_rows: list[dict[str, object]] = []
    for row in metadata.itertuples(index=False):
        recording_path = (PROJECT_ROOT / str(row.file_path)).resolve()
        if not recording_path.is_file():
            raise FileNotFoundError(f"Recording listed in metadata not found: {recording_path}")
        features = extract_features(standardize_clip(recording_path))
        record = {
            "file_path": row.file_path,
            "source_dataset": row.source_dataset,
            "original_class": row.original_class,
            "binary_label": int(row.binary_label),
            "scenario": row.scenario,
        }
        for lineage_column in (
            "original_file_path",
            "recording_id",
            "noise_recording_id",
            "snr_db",
            "measured_snr_db",
            "noise_source_class",
        ):
            if lineage_column in metadata.columns:
                record[lineage_column] = getattr(row, lineage_column)
        feature_rows.append({**record, **features})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(
        feature_rows,
        columns=[
            "file_path",
            "source_dataset",
            "original_class",
            "binary_label",
            "scenario",
            *[
                column
                for column in (
                    "original_file_path",
                    "recording_id",
                    "noise_recording_id",
                    "snr_db",
                    "measured_snr_db",
                    "noise_source_class",
                )
                if column in metadata.columns
            ],
            *FEATURE_COLUMNS,
        ],
    ).to_csv(output_path, index=False)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=FILTERED_DIR / METADATA_FILENAME,
        help="Curated metadata CSV created by filter_datasets.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=FEATURES_DIR / FEATURES_FILENAME,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally process only the first N metadata rows for a smoke test.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_path = extract_dataset(args.metadata, args.output, args.limit)
    print(f"Wrote candidate features to {output_path}")
    print(f"Candidate feature count: {len(FEATURE_COLUMNS)}")


if __name__ == "__main__":
    main()