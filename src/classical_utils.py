"""Shared data, metric, selection, and plotting utilities for classical baselines."""

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import FEATURES_DIR, MODELS_DIR, PROJECT_ROOT, RESULTS_DIR, RANDOM_SEED


TRAIN_SIZES = (24, 50, 70)
SNR_CONDITIONS = ("clean", "20", "10", "5", "0")
FEATURE_COUNTS = (4, 5, 6)
SPLIT_NAMES = ("train", "validation", "test", "nasa_external")


def load_feature_sets() -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    selection = json.loads((RESULTS_DIR / "selected_features.json").read_text())
    tables = {
        split: pd.read_csv(FEATURES_DIR / f"{split}_features.csv")
        for split in SPLIT_NAMES
    }
    for split in ("train", "validation", "test"):
        if tables[split]["external_test"].fillna(False).astype(bool).any():
            raise ValueError(f"NASA rows unexpectedly appear in {split} features")
        if not tables[split]["split"].eq(split).all():
            raise ValueError(f"Unexpected split label in {split} feature table")
    if not tables["nasa_external"]["external_test"].fillna(False).astype(bool).all():
        raise ValueError("Every NASA feature row must be external_test=True")
    return selection, tables


def load_scaled_tables(feature_count: int) -> dict[str, pd.DataFrame]:
    return {
        split: pd.read_csv(FEATURES_DIR / f"{split}_features_{feature_count}.csv")
        for split in SPLIT_NAMES
    }


def load_subset_ids(train_size: int) -> tuple[list[str], int]:
    requested_manifest_size = {24: 25, 50: 50, 70: 100}.get(train_size)
    if requested_manifest_size is None:
        raise ValueError(f"Unsupported actual training recording count: {train_size}")
    path = (
        RESULTS_DIR
        / "small_data_subsets"
        / f"train_subset_{requested_manifest_size}.csv"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Training subset manifest not found: {path}")
    manifest = pd.read_csv(path)
    manifest = manifest[manifest["requested_recordings"] == requested_manifest_size]
    ids = sorted(manifest["recording_id"].astype(str).unique().tolist())
    label_counts = (
        manifest.drop_duplicates("recording_id")["binary_label"].value_counts().to_dict()
    )
    if set(label_counts) != {0, 1} or label_counts[0] != label_counts[1]:
        raise ValueError(f"Subset {train_size} is not balanced by recording: {label_counts}")
    return ids, len(ids)


def feature_names(selection: dict[str, Any], feature_count: int) -> list[str]:
    return list(selection["qubit_mappings"][str(feature_count)]["features"])


def training_rows(
    table: pd.DataFrame,
    recording_ids: list[str],
    snr: str,
) -> pd.DataFrame:
    condition = table["snr_db"].fillna("").astype(str)
    frame = table[
        table["recording_id"].astype(str).isin(recording_ids) & condition.eq(snr)
    ].copy()
    if set(frame["binary_label"].dropna().astype(int).unique()) != {0, 1}:
        raise ValueError(f"Training rows lack a class at SNR={snr}")
    return frame


def evaluation_rows(table: pd.DataFrame, snr: str) -> pd.DataFrame:
    condition = table["snr_db"].fillna("").astype(str)
    frame = table[condition.eq(snr)].copy()
    if frame.empty:
        raise ValueError(f"No evaluation rows available at SNR={snr}")
    return frame


def calculate_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    score: np.ndarray | None,
) -> dict[str, Any]:
    matrix = confusion_matrix(truth, prediction, labels=[0, 1])
    negative_recall = float(recall_score(truth, prediction, pos_label=0, zero_division=0))
    drone_recall = float(recall_score(truth, prediction, pos_label=1, zero_division=0))
    auc = (
        float(roc_auc_score(truth, score))
        if score is not None and np.unique(truth).size == 2
        else float("nan")
    )
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=0)),
        "recall": drone_recall,
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "roc_auc": auc,
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "drone_recall": drone_recall,
        "non_drone_recall": negative_recall,
        "confusion_matrix": json.dumps(matrix.tolist()),
    }


def score_frame(model: Any, frame: pd.DataFrame, features: list[str]) -> tuple[dict[str, Any], np.ndarray]:
    truth = frame["binary_label"].to_numpy(dtype=int)
    prediction = model.predict(frame[features].to_numpy(dtype=float))
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(frame[features].to_numpy(dtype=float))[:, 1]
    elif hasattr(model, "decision_function"):
        score = model.decision_function(frame[features].to_numpy(dtype=float))
    else:
        score = None
    return calculate_metrics(truth, prediction, score), prediction


def validation_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(metrics["f1"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["drone_recall"]),
    )


def save_exact_training_ids(
    train_size: int,
    recording_ids: list[str],
    snr: str,
    selected_rows: pd.DataFrame,
    output_path: Path,
) -> None:
    columns = ["sample_id", "recording_id", "binary_label", "snr_db"]
    exact = selected_rows[columns].drop_duplicates().copy()
    exact.insert(0, "requested_recordings", train_size)
    exact.insert(1, "actual_recordings", len(recording_ids))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = not output_path.exists()
    exact.to_csv(output_path, mode="a", header=header, index=False)


def append_results(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_nasa_external(
    model: Any,
    nasa_table: pd.DataFrame,
    features: list[str],
) -> dict[str, Any]:
    if not nasa_table["external_test"].fillna(False).astype(bool).all():
        raise ValueError("NASA external evaluator received non-external rows")
    if nasa_table["binary_label"].notna().any():
        labeled = nasa_table["binary_label"].notna()
        scores = (
            model.predict_proba(nasa_table.loc[labeled, features].to_numpy(dtype=float))[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(nasa_table.loc[labeled, features].to_numpy(dtype=float))
        )
        predictions = model.predict(nasa_table.loc[labeled, features].to_numpy(dtype=float))
        return {
            **calculate_metrics(
                nasa_table.loc[labeled, "binary_label"].to_numpy(dtype=int),
                predictions,
                scores,
            ),
            "nasa_labeled_rows": int(labeled.sum()),
            "nasa_recordings": int(nasa_table.loc[labeled, "recording_id"].nunique()),
            "mean_drone_probability": float(np.mean(scores)),
            "std_drone_probability": float(np.std(scores)),
        }

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(nasa_table[features].to_numpy(dtype=float))[:, 1]
    else:
        decision = model.decision_function(nasa_table[features].to_numpy(dtype=float))
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(decision, -40, 40)))
    predictions = model.predict(nasa_table[features].to_numpy(dtype=float))
    recording_scores = pd.DataFrame(
        {"recording_id": nasa_table["recording_id"].astype(str), "probability": probabilities}
    ).groupby("recording_id")["probability"].mean()
    return {
        "accuracy": float("nan"),
        "precision": float("nan"),
        "recall": float("nan"),
        "f1": float("nan"),
        "roc_auc": float("nan"),
        "balanced_accuracy": float("nan"),
        "drone_recall": float("nan"),
        "non_drone_recall": float("nan"),
        "confusion_matrix": "",
        "nasa_labeled_rows": 0,
        "nasa_recordings": int(recording_scores.size),
        "predicted_drone_fraction": float(np.mean(predictions == 1)),
        "mean_drone_probability": float(recording_scores.mean()),
        "std_drone_probability": float(recording_scores.std(ddof=0)),
        "drone_probability_q10": float(recording_scores.quantile(0.10)),
        "drone_probability_median": float(recording_scores.quantile(0.50)),
        "drone_probability_q90": float(recording_scores.quantile(0.90)),
    }


def generate_classical_plots(
    svm_path: Path,
    mlp_path: Path,
    comparison_path: Path,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    svm = pd.read_csv(svm_path)
    mlp = pd.read_csv(mlp_path)
    internal_svm = svm[(svm["evaluation_split"] == "test") & (svm["experiment"] == "matched_snr")]
    internal_mlp = mlp[(mlp["evaluation_split"] == "test") & (mlp["experiment"] == "matched_snr")]
    combined = pd.concat([internal_svm, internal_mlp], ignore_index=True, sort=False)
    combined.to_csv(comparison_path, index=False)

    clean = combined[combined["train_snr"].astype(str).eq("clean") & combined["evaluation_snr"].astype(str).eq("clean")]
    for model_name, frame in (("svm", clean[clean.model == "svm"]), ("mlp", clean[clean.model == "mlp"])):
        _plot_lines(
            frame,
            x="train_size_actual",
            y="accuracy",
            hue="feature_count",
            title=f"{model_name.upper()} clean accuracy vs training recordings",
            path=plot_dir / f"{model_name}_accuracy_vs_training_size.png",
        )
    _plot_lines(
        clean,
        x="train_size_actual",
        y="accuracy",
        hue="model",
        title="SVM vs MLP: clean accuracy by training size",
        path=plot_dir / "svm_vs_mlp_training_size.png",
    )

    clean_trained = combined[
        (combined["train_snr"].astype(str) == "clean")
        & (combined["experiment"] == "clean_model_noise_robustness")
    ]
    _plot_lines(
        clean_trained,
        x="evaluation_snr",
        y="accuracy",
        hue="model",
        title="Accuracy vs test SNR (clean-trained models)",
        path=plot_dir / "accuracy_vs_snr.png",
    )
    _plot_lines(
        clean_trained,
        x="evaluation_snr",
        y="f1",
        hue="model",
        title="F1 vs test SNR (clean-trained models)",
        path=plot_dir / "f1_vs_snr.png",
    )
    _plot_lines(
        clean,
        x="feature_count",
        y="accuracy",
        hue="model",
        title="Clean accuracy by selected feature count",
        path=plot_dir / "performance_vs_feature_count.png",
    )

    nasa = pd.concat(
        [svm[svm.evaluation_split == "nasa_external"], mlp[mlp.evaluation_split == "nasa_external"]],
        ignore_index=True,
        sort=False,
    )
    if not nasa.empty and "mean_drone_probability" in nasa:
        figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        valid = nasa.dropna(subset=["mean_drone_probability"])
        axis.bar(valid["model"], valid["mean_drone_probability"], color=["#2171b5", "#d95f0e"][:len(valid)])
        axis.set_ylim(0, 1)
        axis.set_ylabel("Mean drone probability across NASA recordings")
        axis.set_title("Unlabeled NASA external predictions (not an accuracy score)")
        figure.savefig(plot_dir / "nasa_external_comparison.png", dpi=180)
        plt.close(figure)


def _plot_lines(
    frame: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str,
    path: Path,
) -> None:
    usable = frame.dropna(subset=[x, y, hue]) if not frame.empty else frame
    figure, axis = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    if not usable.empty:
        for label, group in usable.groupby(hue):
            means = group.groupby(x, sort=True)[y].mean()
            axis.plot(means.index.astype(str), means.values, marker="o", label=str(label))
        axis.legend(title=hue.replace("_", " "))
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel(y.replace("_", " ").title())
    axis.set_title(title)
    axis.grid(True, alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)