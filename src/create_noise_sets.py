"""Create clean and controlled-SNR copies using real confuser recordings."""

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from config import (
    NOISY_AUDIO_DIR,
    NOISY_METADATA,
    PREPROCESSED_METADATA,
    PROJECT_ROOT,
    RANDOM_SEED,
    SAMPLE_RATE,
    SNR_LEVELS_DB,
)


REQUIRED_COLUMNS = {
    "file_path",
    "source_dataset",
    "original_class",
    "binary_label",
    "scenario",
    "recording_id",
}


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def mix_at_snr(
    target: np.ndarray, background: np.ndarray, snr_db: float
) -> tuple[np.ndarray, np.ndarray]:
    target_rms = float(np.sqrt(np.mean(np.square(target, dtype=np.float64))))
    background_rms = float(np.sqrt(np.mean(np.square(background, dtype=np.float64))))
    if target_rms <= 1e-12:
        raise ValueError("Cannot impose an SNR on a silent target recording")
    if background_rms <= 1e-12:
        raise ValueError("Selected background recording is silent")

    background_scale = target_rms / (10.0 ** (snr_db / 20.0) * background_rms)
    target_component = target.astype(np.float64)
    background_component = background_scale * background.astype(np.float64)
    mixed = target_component + background_component
    peak = float(np.max(np.abs(mixed)))
    if peak > 0.99:
        attenuation = 0.99 / peak
        mixed *= attenuation
        target_component *= attenuation
    return mixed.astype(np.float32), target_component.astype(np.float32)


def measured_snr_db(target: np.ndarray, mixture: np.ndarray) -> float:
    residual = mixture.astype(np.float64) - target.astype(np.float64)
    target_rms = float(np.sqrt(np.mean(np.square(target, dtype=np.float64))))
    residual_rms = float(np.sqrt(np.mean(np.square(residual))))
    return float(20.0 * np.log10(target_rms / residual_rms))


def create_noise_sets(
    metadata_path: Path,
    output_dir: Path,
    output_metadata: Path,
    snr_levels: tuple[int, ...] = SNR_LEVELS_DB,
    limit: int | None = None,
    seed: int = RANDOM_SEED,
) -> Path:
    metadata = pd.read_csv(metadata_path)
    missing = REQUIRED_COLUMNS.difference(metadata.columns)
    if missing:
        raise ValueError(f"Input metadata is missing columns: {sorted(missing)}")
    if metadata.empty:
        raise ValueError("Input metadata contains no recordings")
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive integer")
        targets = metadata.head(limit)
    else:
        targets = metadata

    backgrounds = metadata[metadata["binary_label"] == 0]
    if backgrounds.empty:
        raise ValueError("At least one class-0 confuser recording is required as background")

    rng = np.random.default_rng(seed)
    background_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    conditions: tuple[int | None, ...] = (None, *snr_levels)

    for _, target_row in targets.iterrows():
        target_id = str(target_row["recording_id"])
        target_path = resolve_path(str(target_row["file_path"]))
        if not target_path.is_file():
            raise FileNotFoundError(f"Recording listed in metadata not found: {target_path}")
        target_audio, _ = librosa.load(target_path, sr=SAMPLE_RATE, mono=True)

        eligible = backgrounds[backgrounds["recording_id"].astype(str) != target_id]
        if eligible.empty:
            raise ValueError(
                f"No separate confuser background is available for recording {target_id}"
            )
        donor_index = int(rng.integers(0, len(eligible)))
        donor_row = eligible.iloc[donor_index]
        donor_id = str(donor_row["recording_id"])
        if donor_id not in background_cache:
            donor_path = resolve_path(str(donor_row["file_path"]))
            if not donor_path.is_file():
                raise FileNotFoundError(f"Background recording not found: {donor_path}")
            background_cache[donor_id], _ = librosa.load(
                donor_path, sr=SAMPLE_RATE, mono=True
            )
        background_audio = background_cache[donor_id]
        if len(background_audio) != len(target_audio):
            background_audio = librosa.util.fix_length(
                background_audio, size=len(target_audio)
            )

        for snr_db in conditions:
            condition_name = "clean" if snr_db is None else f"snr_{snr_db}dB"
            if snr_db is None:
                mixed = target_audio.copy()
                target_component = target_audio
            else:
                mixed, target_component = mix_at_snr(
                    target_audio, background_audio, float(snr_db)
                )
            target_path_out = (
                output_dir
                / condition_name
                / str(target_row["source_dataset"])
                / str(target_row["original_class"])
                / f"{target_id}.wav"
            )
            target_path_out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target_path_out, mixed, SAMPLE_RATE, subtype="PCM_16")

            record = target_row.to_dict()
            record.update(
                {
                    "file_path": target_path_out.resolve()
                    .relative_to(PROJECT_ROOT)
                    .as_posix(),
                    "noise_recording_id": "" if snr_db is None else donor_id,
                    "snr_db": "clean" if snr_db is None else snr_db,
                    "noise_source_class": "" if snr_db is None else donor_row["original_class"],
                    "measured_snr_db": ""
                    if snr_db is None
                    else measured_snr_db(target_component, mixed),
                }
            )
            rows.append(record)

    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(output_metadata, index=False)
    return output_metadata


def parse_snr_levels(value: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("SNR levels must be comma-separated integers") from error
    if not levels or any(level < 0 for level in levels):
        raise argparse.ArgumentTypeError("Provide one or more non-negative SNR levels")
    return levels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=PREPROCESSED_METADATA)
    parser.add_argument("--output-dir", type=Path, default=NOISY_AUDIO_DIR)
    parser.add_argument("--output-metadata", type=Path, default=NOISY_METADATA)
    parser.add_argument(
        "--snr-levels",
        type=parse_snr_levels,
        default=SNR_LEVELS_DB,
        help="Comma-separated SNR values in dB (default: 20,10,5,0).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result_path = create_noise_sets(
        args.metadata,
        args.output_dir,
        args.output_metadata,
        args.snr_levels,
        args.limit,
        args.seed,
    )
    result = pd.read_csv(result_path)
    print(f"Wrote {len(result)} clean/noisy clips to {result_path}")
    print(result.groupby("snr_db").size().to_string())


if __name__ == "__main__":
    main()