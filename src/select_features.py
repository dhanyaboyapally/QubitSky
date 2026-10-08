"""Rank candidate audio features using training data only."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif, mutual_info_classif

from config import (
    FEATURE_SELECTION_CSV,
    FEATURES_DIR,
    PROJECT_ROOT,
    RANDOM_SEED,
    SELECTED_FEATURES_JSON,
)
from extract_features import FEATURE_COLUMNS


def rank_features(
    train_features_path: Path,
    output_csv: Path = FEATURE_SELECTION_CSV,
    output_json: Path = SELECTED_FEATURES_JSON,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(train_features_path)
    if "binary_label" not in frame.columns or "recording_id" not in frame.columns:
        raise ValueError("Training feature table must include binary_label and recording_id")
    if "external_test" in frame.columns and frame["external_test"].fillna(False).astype(bool).any():
        raise ValueError("External-test records cannot be used for feature ranking")
    if "split" in frame.columns and not frame["split"].eq("train").all():
        raise ValueError("Feature ranking accepts train rows only")

    candidates = feature_columns or [column for column in FEATURE_COLUMNS if column in frame]
    missing = set(candidates).difference(frame.columns)
    if missing:
        raise ValueError(f"Feature columns are missing: {sorted(missing)}")
    if len(candidates) < 6:
        raise ValueError("At least six candidate feature columns are required for 4/5/6 selections")
    if frame.empty:
        raise ValueError("Training feature table contains no rows")
    if frame[candidates].isna().any().any():
        raise ValueError("Candidate features contain missing values")

    # Rank independent original recordings, not multiple segments as if they were new examples.
    recording_features = frame.groupby("recording_id", as_index=False)[candidates].mean()
    labels_per_recording = frame.groupby("recording_id")["binary_label"].nunique()
    if (labels_per_recording > 1).any():
        raise ValueError("A recording_id maps to multiple binary labels")
    labels = (
        frame.groupby("recording_id")["binary_label"].first()
        .reindex(recording_features["recording_id"])
        .astype(int)
        .to_numpy()
    )
    if np.unique(labels).size != 2:
        raise ValueError("Training feature table must contain both binary classes")
    matrix = recording_features[candidates].to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Candidate features contain non-finite values")

    mi_scores = mutual_info_classif(matrix, labels, random_state=RANDOM_SEED)
    anova_scores, anova_pvalues = f_classif(matrix, labels)
    anova_scores = np.nan_to_num(anova_scores, nan=0.0, posinf=0.0, neginf=0.0)
    anova_pvalues = np.nan_to_num(anova_pvalues, nan=1.0, posinf=1.0, neginf=1.0)
    ranking = pd.DataFrame(
        {
            "feature": candidates,
            "mutual_information": mi_scores,
            "anova_f_score": anova_scores,
            "anova_p_value": anova_pvalues,
        }
    )
    ranking["mutual_information_rank"] = ranking["mutual_information"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking["anova_rank"] = ranking["anova_f_score"].rank(
        ascending=False, method="min"
    ).astype(int)
    ranking["mean_rank"] = ranking[["mutual_information_rank", "anova_rank"]].mean(axis=1)
    ranking = ranking.sort_values(
        ["mean_rank", "mutual_information_rank", "anova_rank", "feature"],
        kind="stable",
    ).reset_index(drop=True)
    ranking["combined_rank"] = np.arange(1, len(ranking) + 1)

    ranked_features = ranking["feature"].tolist()
    selections: dict[str, object] = {
        "ranking_source": str(train_features_path),
        "ranking_unit": "recording_id mean across training segments",
        "recordings_used": int(recording_features.shape[0]),
        "qubit_mappings": {},
    }
    for count in (4, 5, 6):
        selections["qubit_mappings"][str(count)] = {
            "feature_count": count,
            "qubits": count,
            "features": ranked_features[:count],
        }

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_csv, index=False)
    output_json.write_text(json.dumps(selections, indent=2) + "\n", encoding="utf-8")
    return ranking, selections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-features",
        type=Path,
        default=FEATURES_DIR / "train_features.csv",
    )
    parser.add_argument("--output-csv", type=Path, default=FEATURE_SELECTION_CSV)
    parser.add_argument("--output-json", type=Path, default=SELECTED_FEATURES_JSON)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ranking, selections = rank_features(
        args.train_features, args.output_csv, args.output_json
    )
    print(f"Ranked {len(ranking)} candidate features from training recordings only")
    for count, mapping in selections["qubit_mappings"].items():
        print(f"{count} features -> {count} qubits: {', '.join(mapping['features'])}")


if __name__ == "__main__":
    main()