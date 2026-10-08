"""Convert NASA external-test MAT recordings into compact mono WAV source audio."""

import argparse
import csv
import io
import zipfile
from pathlib import Path

import numpy as np
import scipy.io
import soundfile as sf

from config import (
    NASA_ARCHIVE,
    NASA_AUDIO_DIR,
    NASA_CONVERSION_METADATA,
    PROJECT_ROOT,
)


def convert_archive(
    archive_path: Path,
    output_dir: Path,
    metadata_path: Path,
    remove_archive: bool = False,
    limit: int | None = None,
) -> tuple[int, int]:
    if not archive_path.is_file():
        raise FileNotFoundError(f"NASA archive not found: {archive_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".mat") and not name.startswith("__MACOSX/")
        )
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be a positive integer")
            members = members[:limit]
        for member in members:
            mat = scipy.io.loadmat(
                io.BytesIO(archive.read(member)),
                squeeze_me=True,
                struct_as_record=False,
            )
            acoustics = mat["acoustics"]
            pressure = np.asarray(acoustics.incident_pascals, dtype=np.float32)
            if pressure.ndim == 2:
                channel_count = pressure.shape[1]
                audio = pressure[:, 0]
            elif pressure.ndim == 1:
                channel_count = 1
                audio = pressure
            else:
                raise ValueError(
                    f"Unexpected NASA acoustic matrix shape in {member}: {pressure.shape}"
                )

            timestamps = np.asarray(acoustics.utc_time, dtype=np.float64).reshape(-1)
            if timestamps.size != audio.size or timestamps.size < 2:
                raise ValueError(f"Invalid NASA acoustic timestamps in {member}")
            sample_rate = int(round(1.0 / float(np.median(np.diff(timestamps)))))
            peak_pressure = float(np.max(np.abs(audio)))
            if peak_pressure <= 0 or not np.isfinite(peak_pressure):
                raise ValueError(f"NASA recording is silent or invalid: {member}")
            normalized = (0.99 * audio / peak_pressure).astype(np.float32)
            target = output_dir / f"{Path(member).stem}.flac"
            sf.write(
                target,
                normalized,
                sample_rate,
                format="FLAC",
                subtype="PCM_16",
            )
            try:
                target_reference = target.resolve().relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                target_reference = target.resolve().as_posix()
            entries.append(
                {
                    "file_path": target_reference,
                    "source_dataset": "nasa_external",
                    "original_class": Path(member).stem.rsplit("_", 1)[0],
                    "binary_label": "",
                    "scenario": "",
                    "recording_id": "",
                    "external_test": True,
                    "source_archive_member": member,
                    "source_sample_rate": sample_rate,
                    "source_channel_count": channel_count,
                    "selected_channel": 0,
                    "peak_pressure_pascals": peak_pressure,
                }
            )

    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(entries[0]) if entries else [])
        if entries:
            writer.writeheader()
            writer.writerows(entries)

    if remove_archive:
        archive_path.unlink()
    return len(entries), sum(Path(str(entry["file_path"])).stat().st_size for entry in entries)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=NASA_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=NASA_AUDIO_DIR)
    parser.add_argument("--metadata", type=Path, default=NASA_CONVERSION_METADATA)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N archive members for a smoke test.",
    )
    parser.add_argument(
        "--remove-archive",
        action="store_true",
        help="Remove the downloaded source ZIP after every MAT file converts successfully.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    count, size_bytes = convert_archive(
        args.archive,
        args.output_dir,
        args.metadata,
        args.remove_archive,
        args.limit,
    )
    print(f"Converted {count} NASA recordings to mono WAV ({size_bytes / 1024**2:.1f} MiB)")
    print(f"Saved conversion provenance to {args.metadata}")


if __name__ == "__main__":
    main()