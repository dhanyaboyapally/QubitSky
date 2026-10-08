"""Copy selected drone and acoustic-confuser audio into a curated dataset."""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from config import (
    AUDIO_SUFFIXES,
    DRONE_AUDIO_DIR,
    ESC50_AUDIO_DIR,
    ESC50_CONFUSERS,
    ESC50_METADATA,
    FILTERED_DIR,
    SCENARIO_CLASSES,
    URBANSOUND8K_AUDIO_DIR,
    URBANSOUND8K_CONFUSERS,
    URBANSOUND8K_METADATA,
)


METADATA_COLUMNS = [
    "file_path",
    "source_dataset",
    "original_class",
    "binary_label",
    "scenario",
]


def scenarios_for(class_name: str) -> str:
    """Return all configured scenarios associated with a confuser class."""
    return ";".join(
        scenario
        for scenario, classes in SCENARIO_CLASSES.items()
        if class_name in classes
    )


def copy_recording(
    source_path: Path,
    output_dir: Path,
    source_dataset: str,
    original_class: str,
    binary_label: int,
    scenario: str,
    project_root: Path,
) -> dict[str, object]:
    """Copy one recording and construct its inventory row."""
    target_dir = output_dir / source_dataset / original_class
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / source_path.name
    shutil.copy2(source_path, target_path)
    try:
        metadata_path = target_path.resolve().relative_to(project_root).as_posix()
    except ValueError:
        metadata_path = target_path.resolve().as_posix()
    return {
        "file_path": metadata_path,
        "source_dataset": source_dataset,
        "original_class": original_class,
        "binary_label": binary_label,
        "scenario": scenario,
    }


def require_columns(frame: pd.DataFrame, columns: set[str], dataset: str) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"{dataset} metadata is missing columns: {sorted(missing)}")


def filter_datasets(args: argparse.Namespace) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    if args.drone_dir.exists():
        drone_files = sorted(
            path
            for path in args.drone_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        )
        for source_path in drone_files:
            records.append(
                copy_recording(
                    source_path,
                    output_dir,
                    "drone",
                    "drone",
                    1,
                    "",
                    project_root,
                )
            )

    if args.esc50_metadata.exists():
        esc50 = pd.read_csv(args.esc50_metadata)
        require_columns(esc50, {"filename", "category"}, "ESC-50")
        selected = esc50[esc50["category"].isin(ESC50_CONFUSERS)]
        for row in selected.itertuples(index=False):
            class_name = str(row.category)
            source_path = args.esc50_audio_dir / str(row.filename)
            if not source_path.is_file():
                raise FileNotFoundError(f"ESC-50 audio file not found: {source_path}")
            records.append(
                copy_recording(
                    source_path,
                    output_dir,
                    "esc50",
                    class_name,
                    0,
                    scenarios_for(class_name),
                    project_root,
                )
            )

    if args.urbansound8k_metadata.exists():
        urban = pd.read_csv(args.urbansound8k_metadata)
        require_columns(urban, {"slice_file_name", "fold", "class"}, "UrbanSound8K")
        selected = urban[urban["class"].isin(URBANSOUND8K_CONFUSERS)]
        for _, row in selected.iterrows():
            class_name = str(row["class"])
            source_path = (
                args.urbansound8k_audio_dir
                / f"fold{int(row['fold'])}"
                / str(row["slice_file_name"])
            )
            if not source_path.is_file():
                raise FileNotFoundError(f"UrbanSound8K audio file not found: {source_path}")
            records.append(
                copy_recording(
                    source_path,
                    output_dir,
                    "urbansound8k",
                    class_name,
                    0,
                    scenarios_for(class_name),
                    project_root,
                )
            )

    if not records:
        raise ValueError(
            "No recordings were found. Add drone audio and dataset metadata/audio, "
            "or pass their locations with the command-line options."
        )

    metadata_path = output_dir / "metadata.csv"
    pd.DataFrame.from_records(records, columns=METADATA_COLUMNS).to_csv(
        metadata_path, index=False
    )
    return metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drone-dir", type=Path, default=DRONE_AUDIO_DIR)
    parser.add_argument("--esc50-metadata", type=Path, default=ESC50_METADATA)
    parser.add_argument("--esc50-audio-dir", type=Path, default=ESC50_AUDIO_DIR)
    parser.add_argument(
        "--urbansound8k-metadata", type=Path, default=URBANSOUND8K_METADATA
    )
    parser.add_argument(
        "--urbansound8k-audio-dir", type=Path, default=URBANSOUND8K_AUDIO_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=FILTERED_DIR)
    return parser


def main() -> None:
    metadata_path = filter_datasets(build_parser().parse_args())
    metadata = pd.read_csv(metadata_path)
    print(f"Wrote {len(metadata)} recordings to {metadata_path}")
    print(metadata.groupby(["source_dataset", "binary_label"]).size().to_string())


if __name__ == "__main__":
    main()