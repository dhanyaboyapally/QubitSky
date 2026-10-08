"""Build a unified, lineage-aware inventory of local QubitSky audio files."""

import argparse
import hashlib
import re
import zipfile
from pathlib import Path

import pandas as pd

from config import (
    AUDIO_SUFFIXES,
    DDL_ROOT,
    ESC50_AUDIO_DIR,
    ESC50_CONFUSERS,
    ESC50_METADATA,
    ITU_DISJOINT_SPLIT,
    ITU_GROUPS_METADATA,
    MASTER_METADATA,
    NASA_EXTERNAL_ROOT,
    NASA_ARCHIVE,
    PROJECT_ROOT,
    SCENARIO_CLASSES,
    SVANSTROM_ROOT,
    URBANSOUND8K_AUDIO_DIR,
    URBANSOUND8K_CONFUSERS,
    URBANSOUND8K_METADATA,
)


OUTPUT_COLUMNS = [
    "file_path",
    "source_dataset",
    "original_class",
    "binary_label",
    "scenario",
    "recording_id",
    "recommended_split",
    "external_test",
]
DRONE_LABELS = {"drone", "uas", "uav", "positive"}
BACKGROUND_LABELS = {
    "background",
    "no_drone",
    "nondrone",
    "negative",
    "no_uas",
    "no_uav",
}


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def make_recording_id(source_dataset: str, file_path: Path, dataset_root: Path) -> str:
    try:
        stable_path = file_path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        stable_path = file_path.resolve().as_posix()
    digest = hashlib.sha256(f"{source_dataset}:{stable_path}".encode()).hexdigest()
    return digest[:20]


def make_group_recording_id(source_dataset: str, original_id: object) -> str:
    digest = hashlib.sha256(f"{source_dataset}:{original_id}".encode()).hexdigest()
    return digest[:20]


def scenario_for(class_name: str) -> str:
    class_key = normalized_name(class_name)
    scenarios = [
        scenario
        for scenario, classes in SCENARIO_CLASSES.items()
        if class_key in {normalized_name(item) for item in classes}
    ]
    return ";".join(scenarios)


def resolve_manifest_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.is_file():
        return project_candidate
    return (root / path).resolve()


def make_row(
    path: Path,
    source_dataset: str,
    original_class: str,
    binary_label: int | None,
    dataset_root: Path,
    external_test: bool = False,
    recording_id: str | None = None,
) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Audio file listed by dataset metadata was not found: {path}")
    return {
        "file_path": relative_path(path),
        "source_dataset": source_dataset,
        "original_class": original_class,
        "binary_label": binary_label,
        "scenario": scenario_for(original_class),
        "recording_id": recording_id
        or make_recording_id(source_dataset, path, dataset_root),
        "recommended_split": "",
        "external_test": external_test,
    }


def read_manifest(
    manifest_path: Path,
    source_dataset: str,
    dataset_root: Path,
    external_test: bool = False,
) -> list[dict[str, object]]:
    manifest = pd.read_csv(manifest_path)
    required = {"file_path", "binary_label"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(f"{manifest_path} is missing manifest columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for _, item in manifest.iterrows():
        raw_label = item["binary_label"]
        label = None if pd.isna(raw_label) else int(raw_label)
        if label not in (None, 0, 1):
            raise ValueError(f"binary_label must be 0, 1, or blank in {manifest_path}")
        if external_test and label not in (None, 0, 1):
            raise ValueError("External test labels must be 0, 1, or blank")
        file_path = resolve_manifest_path(str(item["file_path"]), dataset_root)
        raw_class_name = item.get("original_class", "unlabeled" if label is None else label)
        class_name = (
            "unlabeled" if pd.isna(raw_class_name) else str(raw_class_name)
        )
        supplied_id = item.get("recording_id")
        recording_id = (
            str(supplied_id)
            if supplied_id is not None and not pd.isna(supplied_id) and str(supplied_id)
            else None
        )
        row = make_row(
            file_path,
            source_dataset,
            class_name,
            label,
            dataset_root,
            external_test,
            recording_id,
        )
        if "scenario" in item and not pd.isna(item["scenario"]):
            row["scenario"] = str(item["scenario"])
        rows.append(row)
    return rows


def infer_ddl_label(path: Path, root: Path) -> tuple[int, str] | None:
    relative_parts = path.resolve().relative_to(root.resolve()).parts[:-1]
    for part in reversed(relative_parts):
        label = normalized_name(part)
        if label in DRONE_LABELS:
            return 1, part
        if label in BACKGROUND_LABELS:
            return 0, part
    sample_class = path.stem[14:18].upper() if len(path.stem) >= 18 else ""
    if path.stem[:14].isdigit():
        if sample_class in {"MINI", "PRO4"}:
            return 1, sample_class
        if sample_class == "XXXX":
            return 0, "background"
    return None


def ddl_recording_id(path: Path, root: Path) -> str | None:
    stem = path.stem
    if len(stem) >= 43 and stem[:14].isdigit() and stem[30:31].upper() in {"R", "S"}:
        return make_group_recording_id("ddl", stem[:43])
    return None


def ingest_ddl(root: Path, manifest_path: Path | None) -> list[dict[str, object]]:
    if manifest_path is not None:
        return read_manifest(manifest_path, "ddl", root)
    if not root.exists():
        return []
    groups_path = root / "groups.json"
    official_split_path = root / "splits" / "split_interval_disjoint.json"
    if groups_path.is_file() and official_split_path.is_file():
        import json

        groups_data = json.loads(groups_path.read_text(encoding="utf-8"))
        split_data = json.loads(official_split_path.read_text(encoding="utf-8"))
        assignment = split_data.get("assignment", {})
        rows = []
        for segment in groups_data.get("segments", []):
            relative_audio_path = str(segment["file"])
            audio_path = root / relative_audio_path
            label_name = normalized_name(str(segment["label"]))
            binary_label = 1 if label_name == "drone" else 0
            group_name = str(segment["group"])
            row = make_row(
                audio_path,
                "ddl",
                label_name,
                binary_label,
                root,
                recording_id=make_group_recording_id("ddl", group_name),
            )
            assigned_split = assignment.get(relative_audio_path)
            if assigned_split is None:
                raise ValueError(
                    f"ITU split metadata has no assignment for {relative_audio_path}"
                )
            row["recommended_split"] = str(assigned_split)
            rows.append(row)
        return rows

    audio_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    rows: list[dict[str, object]] = []
    unmapped: list[Path] = []
    for path in audio_paths:
        inferred = infer_ddl_label(path, root)
        if inferred is None:
            unmapped.append(path)
            continue
        label, class_name = inferred
        rows.append(
            make_row(
                path,
                "ddl",
                class_name,
                label,
                root,
                recording_id=ddl_recording_id(path, root),
            )
        )
    if unmapped:
        examples = ", ".join(str(path.relative_to(root)) for path in unmapped[:5])
        raise ValueError(
            f"Could not infer DDL labels for {len(unmapped)} files from class folders. "
            "Use folders named drone and background/no_drone, or provide --ddl-metadata. "
            f"Examples: {examples}"
        )
    return rows


def infer_svanstrom_class(path: Path) -> tuple[int, str] | None:
    """Svanström audio labels are encoded in filenames such as DRONE_001.wav."""
    stem = normalized_name(path.stem)
    for label_name, binary_label in (
        ("drone", 1),
        ("helicopter", 0),
        ("background", 0),
    ):
        if stem == label_name or stem.startswith(f"{label_name}_"):
            return binary_label, label_name
    return None


def ingest_svanstrom(root: Path) -> list[dict[str, object]]:
    if not root.exists():
        return []
    audio_paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    rows: list[dict[str, object]] = []
    unmapped: list[Path] = []
    for path in audio_paths:
        inferred = infer_svanstrom_class(path)
        if inferred is None:
            unmapped.append(path)
            continue
        binary_label, class_name = inferred
        rows.append(make_row(path, "svanstrom", class_name, binary_label, root))
    if unmapped:
        examples = ", ".join(path.name for path in unmapped[:5])
        raise ValueError(
            f"Could not infer Svanström labels for {len(unmapped)} audio files from "
            f"their filenames. Expected DRONE_, HELICOPTER_, or BACKGROUND_ prefixes. "
            f"Examples: {examples}"
        )
    return rows


def ingest_esc50(metadata_path: Path, audio_dir: Path) -> list[dict[str, object]]:
    if not metadata_path.exists():
        return []
    metadata = pd.read_csv(metadata_path)
    required = {"filename", "category"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"ESC-50 metadata is missing columns: {sorted(missing)}")
    selected = metadata[
        metadata["category"].astype(str).map(normalized_name).isin(
            {normalized_name(item) for item in ESC50_CONFUSERS}
        )
    ]
    rows = []
    for item in selected.itertuples(index=False):
        path = audio_dir / str(item.filename)
        original_id = getattr(item, "src_file", Path(str(item.filename)).stem)
        rows.append(
            make_row(
                path,
                "esc50",
                str(item.category),
                0,
                audio_dir.parent,
                recording_id=make_group_recording_id("esc50", original_id),
            )
        )
    return rows


def ingest_urbansound8k(metadata_path: Path, audio_dir: Path) -> list[dict[str, object]]:
    if not metadata_path.exists():
        return []
    metadata = pd.read_csv(metadata_path)
    required = {"slice_file_name", "fold", "class"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"UrbanSound8K metadata is missing columns: {sorted(missing)}")
    selected = metadata[
        metadata["class"].astype(str).map(normalized_name).isin(
            {normalized_name(item) for item in URBANSOUND8K_CONFUSERS}
        )
    ]
    rows = []
    for _, item in selected.iterrows():
        fold_dir = audio_dir / f"fold{int(item['fold'])}"
        path = fold_dir / str(item["slice_file_name"])
        if not path.is_file():
            path = path.with_suffix(".flac")
        if not path.is_file():
            continue
        rows.append(
            make_row(
                path,
                "urbansound8k",
                str(item["class"]),
                0,
                audio_dir.parent,
                recording_id=make_group_recording_id(
                    "urbansound8k", item.get("fsID", item["slice_file_name"])
                ),
            )
        )
    return rows


def nasa_archive_reference(archive_path: Path, member: str) -> str:
    return f"zip://{relative_path(archive_path)}::{member}"


def ingest_nasa(
    root: Path,
    manifest_path: Path | None,
    archive_path: Path | None = None,
) -> list[dict[str, object]]:
    if manifest_path is not None:
        return read_manifest(manifest_path, "nasa_external", root, external_test=True)
    if archive_path is not None and archive_path.is_file():
        with zipfile.ZipFile(archive_path) as archive:
            members = sorted(
                name
                for name in archive.namelist()
                if name.lower().endswith(".mat") and not name.startswith("__MACOSX/")
            )
        return [
            {
                "file_path": nasa_archive_reference(archive_path, member),
                "source_dataset": "nasa_external",
                "original_class": Path(member).stem.rsplit("_", 1)[0],
                "binary_label": None,
                "scenario": "",
                "recording_id": make_group_recording_id("nasa_external", member),
                "external_test": True,
            }
            for member in members
        ]
    if not root.exists():
        return []
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )
    return [
        make_row(path, "nasa_external", path.parent.name, None, root, external_test=True)
        for path in paths
    ]


def dataset_preflight(
    svanstrom_root: Path,
    esc50_metadata: Path,
    esc50_audio_dir: Path,
    nasa_root: Path,
    nasa_archive: Path,
    urban_metadata: Path,
    urban_audio: Path,
    ddl_root: Path,
) -> tuple[dict[str, int], list[str]]:
    svanstrom_files = [
        path
        for path in svanstrom_root.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    ] if svanstrom_root.exists() else []
    svanstrom_counts = {name: 0 for name in ("drone", "helicopter", "background")}
    for path in svanstrom_files:
        inferred = infer_svanstrom_class(path)
        if inferred:
            svanstrom_counts[inferred[1]] += 1

    esc_metadata = pd.read_csv(esc50_metadata) if esc50_metadata.is_file() else pd.DataFrame()
    esc_rows = []
    if not esc_metadata.empty and {"filename", "category"}.issubset(esc_metadata.columns):
        retained = esc_metadata[
            esc_metadata["category"].astype(str).map(normalized_name).isin(
                {normalized_name(item) for item in ESC50_CONFUSERS}
            )
        ]
        esc_rows = [
            row
            for row in retained.itertuples(index=False)
            if (esc50_audio_dir / str(row.filename)).is_file()
        ]
    esc_class_counts = pd.Series([str(row.category) for row in esc_rows]).value_counts().to_dict()

    if nasa_archive.is_file():
        with zipfile.ZipFile(nasa_archive) as archive:
            nasa_count = sum(
                1
                for name in archive.namelist()
                if name.lower().endswith(".mat") and not name.startswith("__MACOSX/")
            )
    else:
        nasa_count = sum(
            1
            for path in nasa_root.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        ) if nasa_root.exists() else 0

    urban_files = [
        path
        for path in urban_audio.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    ] if urban_audio.exists() else []
    urban_present = urban_metadata.is_file() and bool(urban_files)
    ddl_present = ddl_root.exists() and any(
        path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
        for path in ddl_root.rglob("*")
    )

    print("DATASET PREFLIGHT")
    print(
        "SVANSTROM: "
        f"total_audio_files={len(svanstrom_files)}, "
        f"drone={svanstrom_counts['drone']}, "
        f"helicopter={svanstrom_counts['helicopter']}, "
        f"background={svanstrom_counts['background']}"
    )
    print(
        "ESC-50: "
        f"discovered_audio_files={len(list(esc50_audio_dir.glob('*')) if esc50_audio_dir.exists() else [])}, "
        f"retained_classes={','.join(sorted(ESC50_CONFUSERS))}, "
        f"retained_files={len(esc_rows)}, class_counts={esc_class_counts}"
    )
    print(f"NASA: audio/MAT recordings={nasa_count}, external_test=True")
    print(
        f"OPTIONAL UrbanSound8K: {'present' if urban_present else 'not present'}, "
        f"local_audio_files={len(urban_files)}"
    )
    print(f"OPTIONAL DDL: {'present' if ddl_present else 'not present'}")

    missing: list[str] = []
    if not svanstrom_counts["drone"] or not svanstrom_counts["helicopter"] or not svanstrom_counts["background"]:
        missing.append("Svanström labeled Drone/Helicopter/Background audio")
    if not esc_rows:
        missing.append("ESC-50 metadata and retained-class audio")
    if not nasa_count:
        missing.append("NASA Small UAS Flyover Acoustics archive or audio")
    return {"svanstrom": len(svanstrom_files), "esc50": len(esc_rows), "nasa_external": nasa_count}, missing


def build_master_metadata(args: argparse.Namespace) -> Path:
    _, missing_required = dataset_preflight(
        args.svanstrom_root,
        args.esc50_metadata,
        args.esc50_audio_dir,
        args.nasa_root,
        args.nasa_archive,
        args.urbansound8k_metadata,
        args.urbansound8k_audio_dir,
        args.ddl_root,
    )
    if missing_required:
        raise ValueError("Required dataset(s) missing: " + "; ".join(missing_required))
    rows: list[dict[str, object]] = []
    rows.extend(ingest_svanstrom(args.svanstrom_root))
    rows.extend(ingest_ddl(args.ddl_root, args.ddl_metadata))
    rows.extend(ingest_esc50(args.esc50_metadata, args.esc50_audio_dir))
    rows.extend(
        ingest_urbansound8k(args.urbansound8k_metadata, args.urbansound8k_audio_dir)
    )
    rows.extend(ingest_nasa(args.nasa_root, args.nasa_metadata, args.nasa_archive))
    if not rows:
        raise ValueError(
            "No supported audio records found. Add datasets under data/raw/ or pass "
            "the dataset metadata/audio paths explicitly."
        )

    master = pd.DataFrame.from_records(rows, columns=OUTPUT_COLUMNS)
    duplicates = master[master["recording_id"].duplicated(keep=False)]
    if not duplicates.empty:
        labels_per_id = duplicates.groupby("recording_id")["binary_label"].nunique(dropna=True)
        if (labels_per_id > 1).any():
            bad_ids = labels_per_id[labels_per_id > 1].index.tolist()[:5]
            raise ValueError(f"A source recording has conflicting labels: {bad_ids}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(args.output, index=False)
    return args.output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svanstrom-root", type=Path, default=SVANSTROM_ROOT)
    parser.add_argument("--ddl-root", type=Path, default=DDL_ROOT)
    parser.add_argument("--ddl-metadata", type=Path, default=None)
    parser.add_argument("--esc50-metadata", type=Path, default=ESC50_METADATA)
    parser.add_argument("--esc50-audio-dir", type=Path, default=ESC50_AUDIO_DIR)
    parser.add_argument(
        "--urbansound8k-metadata", type=Path, default=URBANSOUND8K_METADATA
    )
    parser.add_argument(
        "--urbansound8k-audio-dir", type=Path, default=URBANSOUND8K_AUDIO_DIR
    )
    parser.add_argument("--nasa-root", type=Path, default=NASA_EXTERNAL_ROOT)
    parser.add_argument("--nasa-archive", type=Path, default=NASA_ARCHIVE)
    parser.add_argument("--nasa-metadata", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=MASTER_METADATA)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = build_master_metadata(args)
    master = pd.read_csv(output)
    print(f"Wrote {len(master)} source recordings to {output}")
    print(master.groupby(["source_dataset", "external_test"], dropna=False).size().to_string())
    print(f"Labeled training/evaluation candidates: {master['binary_label'].notna().sum()}")


if __name__ == "__main__":
    main()