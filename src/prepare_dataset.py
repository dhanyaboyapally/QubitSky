"""Prepare leakage-safe Stage 3 splits, features, scalers, and subset manifests."""

import argparse
import hashlib
import io
import json
import warnings
import zipfile
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from build_master_metadata import build_master_metadata, build_parser as master_parser
from config import (
    FEATURES_DIR,
    MASTER_METADATA,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    RESULTS_DIR,
    SAMPLE_RATE,
    SNR_LEVELS_DB,
)
from extract_features import FEATURE_COLUMNS, extract_features, split_windows
from select_features import rank_features
from create_noise_sets import measured_snr_db, mix_at_snr


META_COLUMNS = [
    "sample_id",
    "file_path",
    "source_dataset",
    "original_class",
    "binary_label",
    "scenario",
    "recording_id",
    "external_test",
    "split",
    "segment_index",
    "snr_db",
    "noise_recording_id",
    "measured_snr_db",
    "base_file_path",
    "condition_audio_path",
    "noise_source_file_path",
    "audio_persisted",
]


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sample_identifier(
    recording_id: str,
    source_path: str,
    segment_index: int,
    condition: str,
) -> str:
    source_piece = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:8]
    return f"{recording_id}__{source_piece}__{condition}__seg{segment_index:04d}"


def validate_master(master: pd.DataFrame) -> None:
    required = {
        "file_path",
        "source_dataset",
        "original_class",
        "binary_label",
        "scenario",
        "recording_id",
        "external_test",
    }
    missing = required.difference(master.columns)
    if missing:
        raise ValueError(f"Master metadata is missing columns: {sorted(missing)}")
    if master["recording_id"].isna().any() or master["recording_id"].astype(str).eq("").any():
        raise ValueError("Every source recording must have a non-empty recording_id")
    labeled = master[~master["external_test"].fillna(False).astype(bool)]
    if labeled["binary_label"].isna().any():
        raise ValueError("All non-external recordings need binary_label 0 or 1")
    if not labeled["binary_label"].isin([0, 1]).all():
        raise ValueError("Non-external binary_label values must be 0 or 1")
    label_counts = labeled.groupby("recording_id")["binary_label"].nunique()
    if (label_counts > 1).any():
        raise ValueError("A recording_id appears with conflicting binary labels")
    external = master[master["external_test"].fillna(False).astype(bool)]
    if not external.empty and external["source_dataset"].astype(str).ne("nasa_external").any():
        raise ValueError("Only NASA external recordings may have external_test=True")


def make_recording_split(
    master: pd.DataFrame,
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[dict[str, str], pd.DataFrame]:
    validate_master(master)
    internal = master[~master["external_test"].fillna(False).astype(bool)].copy()
    group_columns = ["recording_id", "binary_label", "source_dataset"]
    if "recommended_split" in internal.columns:
        group_columns.append("recommended_split")
    recording_table = internal[group_columns].drop_duplicates("recording_id")
    if "recommended_split" in recording_table.columns:
        recording_table["recommended_split"] = (
            recording_table["recommended_split"].fillna("").astype(str).str.lower()
        )
        recording_table["recommended_split"] = recording_table["recommended_split"].replace(
            {"val": "validation", "valid": "validation"}
        )
        invalid = set(recording_table["recommended_split"]) - {
            "",
            "train",
            "validation",
            "test",
        }
        if invalid:
            raise ValueError(f"Unsupported recommended split names: {sorted(invalid)}")
        recommended_counts = internal.groupby("recording_id")["recommended_split"].nunique()
        if (recommended_counts > 1).any():
            raise ValueError("Segments in one recording group have conflicting recommended splits")
    if recording_table["binary_label"].nunique() != 2:
        raise ValueError("The core dataset must include both drone and non-drone recordings")
    fixed_mask = (
        recording_table["recommended_split"].ne("")
        if "recommended_split" in recording_table.columns
        else pd.Series(False, index=recording_table.index)
    )
    fixed = recording_table[fixed_mask]
    random_groups = recording_table[~fixed_mask]
    split_map = {
        str(row.recording_id): str(row.recommended_split)
        for row in fixed.itertuples(index=False)
    }
    class_counts = recording_table["binary_label"].value_counts()
    if class_counts.min() < 2:
        raise ValueError("At least two unique recordings per class are required for a held-out test")

    random_class_counts = random_groups["binary_label"].value_counts()
    if random_class_counts.min() < 2:
        raise ValueError("At least two unfixed recordings per class are required for the random held-out split")
    train_groups, test_groups = train_test_split(
        random_groups["recording_id"].to_numpy(),
        test_size=test_size,
        random_state=seed,
        stratify=random_groups["binary_label"].to_numpy(),
    )
    split_map.update({str(recording_id): "test" for recording_id in test_groups})
    split_map.update({str(recording_id): "train" for recording_id in train_groups})

    train_table = random_groups[random_groups["recording_id"].isin(train_groups)]
    per_class_train = train_table["binary_label"].value_counts()
    if validation_size > 0 and len(train_table) >= 6 and per_class_train.min() >= 2:
        try:
            fit_groups, validation_groups = train_test_split(
                train_table["recording_id"].to_numpy(),
                test_size=validation_size,
                random_state=seed + 1,
                stratify=train_table["binary_label"].to_numpy(),
            )
        except ValueError:
            warnings.warn(
                "Could not stratify validation groups at this dataset size; "
                "keeping all non-test groups in training.",
                stacklevel=2,
            )
        else:
            for recording_id in validation_groups:
                split_map[str(recording_id)] = "validation"
            for recording_id in fit_groups:
                split_map[str(recording_id)] = "train"

    split_table = pd.DataFrame(
        {"recording_id": list(split_map), "split": list(split_map.values())}
    )
    return split_map, split_table


def load_audio_source(file_path: str) -> np.ndarray:
    if file_path.startswith("zip://"):
        archive_reference = file_path.removeprefix("zip://")
        archive_path_value, member = archive_reference.split("::", maxsplit=1)
        archive_path = resolve_path(archive_path_value)
        with zipfile.ZipFile(archive_path) as archive:
            mat_content = archive.read(member)
        mat_data = scipy.io.loadmat(
            io.BytesIO(mat_content), squeeze_me=True, struct_as_record=False
        )
        acoustics = mat_data["acoustics"]
        pressure = np.asarray(acoustics.incident_pascals, dtype=np.float32)
        timestamps = np.asarray(acoustics.utc_time, dtype=np.float64).reshape(-1)
        if pressure.ndim == 2:
            pressure = pressure[:, 0]
        elif pressure.ndim != 1:
            raise ValueError(f"Unexpected NASA acoustic matrix shape: {pressure.shape}")
        if timestamps.size != pressure.size or timestamps.size < 2:
            raise ValueError(f"NASA MAT member has invalid sample timestamps: {member}")
        source_rate = int(round(1.0 / float(np.median(np.diff(timestamps)))))
        audio = librosa.resample(
            y=pressure, orig_sr=source_rate, target_sr=SAMPLE_RATE
        )
        return audio.astype(np.float32, copy=False)

    audio, _ = librosa.load(resolve_path(file_path), sr=SAMPLE_RATE, mono=True)
    return audio.astype(np.float32, copy=False)


def segment_audio(file_path: str) -> list[np.ndarray]:
    audio = load_audio_source(file_path)
    if not audio.size:
        raise ValueError(f"Audio file is empty: {file_path}")
    return split_windows(audio)


def path_for_output(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def process_records(
    rows: pd.DataFrame,
    split: str,
    output_dir: Path,
) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        source_reference = str(row["file_path"])
        if source_reference.startswith("zip://"):
            archive_reference = source_reference.removeprefix("zip://")
            archive_path_value, member = archive_reference.split("::", maxsplit=1)
            archive_path = resolve_path(archive_path_value)
            if not archive_path.is_file():
                raise FileNotFoundError(f"NASA archive listed in metadata not found: {archive_path}")
            with zipfile.ZipFile(archive_path) as archive:
                if member not in archive.namelist():
                    raise FileNotFoundError(f"NASA MAT member not found: {member}")
        elif not resolve_path(source_reference).is_file():
            raise FileNotFoundError(
                f"Audio file listed in metadata was not found: {resolve_path(source_reference)}"
            )
        recording_id = str(row["recording_id"])
        condition = str(row.get("snr_db", "clean"))
        condition = "clean" if condition in {"nan", "None", ""} else condition
        clips = segment_audio(source_reference)
        for segment_index, clip in enumerate(clips):
            sample_id = sample_identifier(
                recording_id, source_reference, segment_index, condition
            )
            destination = (
                output_dir
                / split
                / str(row["source_dataset"])
                / str(row["original_class"])
                / f"{sample_id}.flac"
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, clip, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
            record = row.to_dict()
            record.update(
                {
                    "sample_id": sample_id,
                    "file_path": path_for_output(destination),
                    "recording_id": recording_id,
                    "external_test": bool(row.get("external_test", False)),
                    "split": split,
                    "segment_index": segment_index,
                    "snr_db": condition,
                    "base_file_path": path_for_output(resolve_path(source_reference))
                    if not source_reference.startswith("zip://")
                    else source_reference,
                    "condition_audio_path": path_for_output(destination),
                    "noise_source_file_path": "",
                    "noise_recording_id": row.get("noise_recording_id", ""),
                    "measured_snr_db": row.get("measured_snr_db", ""),
                    "audio_persisted": True,
                }
            )
            output_rows.append(record)
    return output_rows


def add_features(processed_rows: list[dict[str, object]]) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for row in processed_rows:
        features = row.get("_feature_values")
        if features is None:
            audio_path = str(row.get("condition_audio_path") or row["file_path"])
            audio, _ = librosa.load(resolve_path(audio_path), sr=SAMPLE_RATE, mono=True)
            features = extract_features(audio)
        record = {key: value for key, value in row.items() if not key.startswith("_")}
        output.append({**record, **features})
    columns = list(dict.fromkeys([*META_COLUMNS, *FEATURE_COLUMNS]))
    return pd.DataFrame.from_records(output).reindex(columns=columns)


def save_feature_tables(
    feature_tables: dict[str, pd.DataFrame],
    selected: dict[str, object],
    feature_dir: Path,
    models_dir: Path,
) -> None:
    for split, table in feature_tables.items():
        table.to_csv(feature_dir / f"{split}_features.csv", index=False)

    train = feature_tables["train"]
    if train["external_test"].fillna(False).astype(bool).any():
        raise ValueError("NASA external rows cannot be used to fit StandardScaler")
    for count in (4, 5, 6):
        feature_names = selected["qubit_mappings"][str(count)]["features"]
        scaler = StandardScaler()
        scaler.fit(train[feature_names].to_numpy(dtype=float))
        joblib.dump(scaler, models_dir / f"scaler_{count}.pkl")

        for split, table in feature_tables.items():
            standardized = table[META_COLUMNS].copy()
            if not table.empty:
                transformed = scaler.transform(table[feature_names].to_numpy(dtype=float))
                standardized[feature_names] = transformed
            else:
                for feature_name in feature_names:
                    standardized[feature_name] = pd.Series(dtype=float)
            standardized.to_csv(
                feature_dir / f"{split}_features_{count}.csv", index=False
            )


def create_split_noise_variants(
    clean_rows: list[dict[str, object]],
    split: str,
    output_dir: Path,
    levels: tuple[int, ...],
    seed: int,
    store_audio: bool = False,
) -> list[dict[str, object]]:
    clean_frame = pd.DataFrame(clean_rows)
    if clean_frame.empty:
        return []
    backgrounds = clean_frame[clean_frame["binary_label"] == 0]
    if backgrounds.empty:
        warnings.warn(f"No class-0 background recordings in {split}; skipping its noise variants.", stacklevel=2)
        return []

    rng = np.random.default_rng(seed)
    audio_cache: dict[str, np.ndarray] = {}
    variants: list[dict[str, object]] = []
    for row in clean_rows:
        target_id = str(row["recording_id"])
        target_path = resolve_path(str(row["file_path"]))
        target_audio, _ = librosa.load(target_path, sr=SAMPLE_RATE, mono=True)
        target_rms = float(np.sqrt(np.mean(np.square(target_audio, dtype=np.float64))))
        if target_rms <= 1e-12:
            warnings.warn(
                f"Target recording {target_id} is silent in {split}; keeping its clean "
                "sample and skipping its noisy derivatives.",
                stacklevel=2,
            )
            continue
        candidates = backgrounds[backgrounds["recording_id"].astype(str) != target_id]
        if candidates.empty:
            warnings.warn(
                f"No independent confuser is available for {target_id} in {split}; "
                "skipping its noisy derivatives.",
                stacklevel=2,
            )
            continue
        donor_order = rng.permutation(len(candidates))
        donor = None
        donor_audio = None
        for donor_index in donor_order:
            candidate = candidates.iloc[int(donor_index)]
            candidate_id = str(candidate["recording_id"])
            if candidate_id not in audio_cache:
                candidate_audio, _ = librosa.load(
                    resolve_path(str(candidate["file_path"])),
                    sr=SAMPLE_RATE,
                    mono=True,
                )
                audio_cache[candidate_id] = candidate_audio
            candidate_audio = audio_cache[candidate_id]
            candidate_rms = float(
                np.sqrt(np.mean(np.square(candidate_audio, dtype=np.float64)))
            )
            if candidate_rms > 1e-12:
                donor = candidate
                donor_audio = candidate_audio
                donor_id = candidate_id
                break
        if donor is None or donor_audio is None:
            warnings.warn(
                f"No non-silent confuser donor is available for {target_id} in {split}; "
                "skipping its noisy derivatives.",
                stacklevel=2,
            )
            continue
        if len(donor_audio) != len(target_audio):
            donor_audio = librosa.util.fix_length(donor_audio, size=len(target_audio))

        for snr_db in levels:
            mixed, target_component = mix_at_snr(target_audio, donor_audio, float(snr_db))
            sample_id = f"{row['sample_id']}__snr{snr_db}"
            destination = (
                output_dir
                / split
                / f"snr_{snr_db}dB"
                / str(row["source_dataset"])
                / str(row["original_class"])
                / f"{sample_id}.flac"
            )
            condition_audio_path = ""
            if store_audio:
                destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(destination, mixed, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
                condition_audio_path = path_for_output(destination)
            noisy_row = dict(row)
            noisy_row.update(
                {
                    "sample_id": sample_id,
                    "file_path": condition_audio_path,
                    "snr_db": snr_db,
                    "noise_recording_id": donor_id,
                    "measured_snr_db": measured_snr_db(target_component, mixed),
                    "base_file_path": str(row["file_path"]),
                    "condition_audio_path": condition_audio_path,
                    "noise_source_file_path": str(donor["file_path"]),
                    "audio_persisted": store_audio,
                    "_feature_values": extract_features(mixed),
                }
            )
            variants.append(noisy_row)
    return variants


def make_small_data_subsets(
    train_features: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> list[dict[str, object]]:
    recording_table = train_features[
        ["recording_id", "binary_label"]
    ].drop_duplicates("recording_id")
    class_to_ids = {
        int(label): group["recording_id"].astype(str).tolist()
        for label, group in recording_table.groupby("binary_label")
    }
    if set(class_to_ids) != {0, 1}:
        raise ValueError("Training split must have both classes for balanced subsets")
    rng = np.random.default_rng(seed)
    manifest_rows: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for requested in (25, 50, 100, 200):
        per_class = min(requested // 2, len(class_to_ids[0]), len(class_to_ids[1]))
        if per_class == 0:
            actual = 0
            chosen_ids: list[str] = []
        else:
            chosen_ids = []
            for label in (0, 1):
                chosen_ids.extend(
                    rng.choice(class_to_ids[label], size=per_class, replace=False).tolist()
                )
            actual = len(chosen_ids)
        chosen_set = set(chosen_ids)
        selected_rows = train_features[
            train_features["recording_id"].astype(str).isin(chosen_set)
        ]
        for _, row in selected_rows.iterrows():
            manifest_rows.append(
                {
                    "requested_recordings": requested,
                    "actual_recordings": actual,
                    "recording_id": row["recording_id"],
                    "sample_id": row["sample_id"],
                    "binary_label": row["binary_label"],
                }
            )
        pd.DataFrame(manifest_rows).query(
            "requested_recordings == @requested"
        ).to_csv(output_dir / f"train_subset_{requested}.csv", index=False)
    return manifest_rows


def build_split_summary(
    split_rows: dict[str, list[dict[str, object]]],
    output_path: Path,
    noisy_count: int,
) -> pd.DataFrame:
    summaries: list[dict[str, object]] = []
    for split, rows in split_rows.items():
        frame = pd.DataFrame(rows)
        if frame.empty:
            summaries.append(
                {
                    "split": split,
                    "files": 0,
                    "unique_recordings": 0,
                    "drone_recordings": 0,
                    "non_drone_recordings": 0,
                    "noisy_clips": 0,
                }
            )
            continue
        labeled = frame[frame["binary_label"].notna()]
        unique_labeled = labeled.drop_duplicates("recording_id")
        summary: dict[str, object] = {
            "split": split,
            "files": len(frame),
            "unique_recordings": frame["recording_id"].nunique(),
            "drone_recordings": int((unique_labeled["binary_label"] == 1).sum()),
            "non_drone_recordings": int((unique_labeled["binary_label"] == 0).sum()),
            "noisy_clips": int(
                frame.get("noise_recording_id", pd.Series(dtype=str))
                .fillna("")
                .astype(str)
                .ne("")
                .sum()
            ),
        }
        summary["counts_by_source"] = json.dumps(
            frame.groupby("source_dataset").size().to_dict(), sort_keys=True
        )
        scenarios = (
            frame["scenario"].fillna("").astype(str).str.split(";").explode()
        )
        scenarios = scenarios[scenarios.ne("")]
        summary["counts_by_scenario"] = json.dumps(
            scenarios.value_counts().to_dict(), sort_keys=True
        )
        summary["counts_by_snr"] = json.dumps(
            frame["snr_db"].fillna("clean").astype(str).value_counts().to_dict(),
            sort_keys=True,
        )
        summaries.append(summary)
    result = pd.DataFrame(summaries)
    result["generated_noisy_clips_total"] = noisy_count
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def run_stage3(args: argparse.Namespace) -> dict[str, object]:
    for directory in (
        args.processed_dir,
        args.features_dir,
        args.results_dir,
        args.models_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if args.build_master:
        builder_args = master_parser().parse_args(
            ["--output", str(args.master_metadata)]
        )
        build_master_metadata(builder_args)
    if not args.master_metadata.exists():
        raise FileNotFoundError(
            f"Master metadata not found: {args.master_metadata}. Run "
            "python src/build_master_metadata.py first."
        )

    master = pd.read_csv(args.master_metadata)
    validate_master(master)
    split_map, split_table = make_recording_split(
        master, args.test_size, args.validation_size, args.seed
    )
    split_table.to_csv(args.results_dir / "recording_split.csv", index=False)
    train_ids = {key for key, split in split_map.items() if split == "train"}
    validation_ids = {key for key, split in split_map.items() if split == "validation"}
    test_ids = {key for key, split in split_map.items() if split == "test"}
    if train_ids & test_ids or train_ids & validation_ids or test_ids & validation_ids:
        raise AssertionError("Recording-level split overlap detected")

    external_mask = master["external_test"].fillna(False).astype(bool)
    internal_master = master[~external_mask].copy()
    internal_master["split"] = internal_master["recording_id"].astype(str).map(split_map)
    if internal_master["split"].isna().any():
        raise AssertionError("A non-external recording was not assigned to a split")
    external_master = master[external_mask].copy()
    external_master["split"] = "nasa_external"

    raw_split_rows: dict[str, list[dict[str, object]]] = {}
    feature_tables: dict[str, pd.DataFrame] = {}
    for split in ("train", "validation", "test"):
        split_master = internal_master[internal_master["split"] == split]
        clean_rows = process_records(split_master, split, args.processed_dir)
        noisy_rows = (
            create_split_noise_variants(
                clean_rows,
                split,
                args.processed_dir,
                SNR_LEVELS_DB,
                args.seed + {"train": 0, "validation": 1, "test": 2}[split],
                args.store_noisy_audio,
            )
            if args.generate_noise
            else []
        )
        processed_rows = [*clean_rows, *noisy_rows]
        raw_split_rows[split] = processed_rows
        feature_tables[split] = add_features(processed_rows)

    noisy_count = sum(
        1
        for rows in raw_split_rows.values()
        for row in rows
        if row.get("noise_recording_id") not in (None, "", np.nan)
    )

    external_rows = process_records(external_master, "nasa_external", args.processed_dir)
    raw_split_rows["nasa_external"] = external_rows
    feature_tables["nasa_external"] = add_features(external_rows)

    # Empty validation splits still need valid output schemas for downstream tools.
    for split, table in feature_tables.items():
        table.to_csv(args.features_dir / f"{split}_features.csv", index=False)
    if feature_tables["train"].empty:
        raise ValueError("Training split produced no feature rows")

    ranking, selection = rank_features(
        args.features_dir / "train_features.csv",
        args.results_dir / "feature_selection.csv",
        args.results_dir / "selected_features.json",
    )
    save_feature_tables(feature_tables, selection, args.features_dir, args.models_dir)
    make_small_data_subsets(
        feature_tables["train"], args.results_dir / "small_data_subsets", args.seed
    )

    final_train_ids = set(feature_tables["train"]["recording_id"].astype(str))
    final_test_ids = set(feature_tables["test"]["recording_id"].astype(str))
    final_validation_ids = set(feature_tables["validation"]["recording_id"].astype(str))
    train_test_overlap = final_train_ids & final_test_ids
    if train_test_overlap:
        raise AssertionError(f"Train/test recording leakage: {sorted(train_test_overlap)[:5]}")
    if final_test_ids & final_validation_ids or final_train_ids & final_validation_ids:
        raise AssertionError("Validation split overlaps train or test recording IDs")
    nasa_in_train = int(feature_tables["train"]["external_test"].fillna(False).astype(bool).sum())
    nasa_in_validation = int(
        feature_tables["validation"]["external_test"].fillna(False).astype(bool).sum()
    )
    nasa_in_ranking = int(
        feature_tables["train"]["external_test"].fillna(False).astype(bool).sum()
    )
    nasa_in_scaler_fit = 0
    if nasa_in_train or nasa_in_validation or nasa_in_ranking or nasa_in_scaler_fit:
        raise AssertionError("NASA external rows entered an internal split or training fit")
    noisy_split_map: dict[str, set[str]] = {}
    for split in ("train", "validation", "test"):
        rows = [
            row
            for row in raw_split_rows[split]
            if row.get("noise_recording_id") not in (None, "", np.nan)
        ]
        for row in rows:
            noisy_split_map.setdefault(str(row["recording_id"]), set()).add(split)
    crossing_noise_ids = [key for key, splits in noisy_split_map.items() if len(splits) > 1]
    if crossing_noise_ids:
        raise AssertionError(f"Noisy derivatives cross splits: {crossing_noise_ids[:5]}")

    summary = build_split_summary(raw_split_rows, args.results_dir / "split_summary.csv", noisy_count)
    counts = master.groupby(["source_dataset", "external_test"], dropna=False).size()
    print(f"Total source files: {len(master)}")
    print(f"Drone recordings: {(internal_master['binary_label'] == 1).sum()}")
    print(f"Non-drone recordings: {(internal_master['binary_label'] == 0).sum()}")
    print(f"Unique source recordings: {master['recording_id'].nunique()}")
    print("Counts by source dataset:")
    print(counts.to_string())
    print("Split summary:")
    print(summary.to_string(index=False))
    print(f"Generated noisy clips included: {noisy_count}")
    print(f"Train/test recording_id intersection: {len(train_test_overlap)}")
    print(f"Train/validation recording_id intersection: {len(final_train_ids & final_validation_ids)}")
    print(f"Validation/test recording_id intersection: {len(final_validation_ids & final_test_ids)}")
    print(f"Noisy derivatives crossing splits: {len(crossing_noise_ids)}")
    print(f"NASA rows in train/validation/ranking/scaler fit: {nasa_in_train}/{nasa_in_validation}/{nasa_in_ranking}/{nasa_in_scaler_fit}")
    for count in (4, 5, 6):
        print(
            f"Top {count} features -> {count} qubits: "
            + ", ".join(selection["qubit_mappings"][str(count)]["features"])
        )
    return {
        "master_rows": len(master),
        "ranking": ranking,
        "selection": selection,
        "train_test_overlap": train_test_overlap,
        "crossing_noise_ids": crossing_noise_ids,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-metadata", type=Path, default=MASTER_METADATA)
    parser.add_argument("--build-master", action="store_true")
    parser.add_argument(
        "--no-noise",
        dest="generate_noise",
        action="store_false",
        help="Do not generate split-local 20/10/5/0 dB copies.",
    )
    parser.set_defaults(generate_noise=True)
    parser.add_argument(
        "--store-noisy-audio",
        action="store_true",
        help="Persist generated noisy conditions as FLAC; default stores their measured features and provenance only.",
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for directory in (
        args.processed_dir,
        args.features_dir,
        args.results_dir,
        args.models_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    run_stage3(args)


if __name__ == "__main__":
    main()