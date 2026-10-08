"""CPU-only classical SVM/MLP training over the fixed Stage 3 experiment assets."""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from config import RANDOM_SEED
from classical_utils import (
    FEATURE_COUNTS,
    PROJECT_ROOT,
    RESULTS_DIR,
    SNR_CONDITIONS,
    TRAIN_SIZES,
    append_results,
    evaluation_rows,
    feature_names,
    generate_classical_plots,
    load_feature_sets,
    load_scaled_tables,
    load_subset_ids,
    run_nasa_external,
    save_exact_training_ids,
    score_frame,
    training_rows,
    validation_key,
)


SVM_C_VALUES = (0.1, 1.0, 10.0)
SVM_GAMMAS = ("scale", "auto")
# Early stopping may only stop after this many optimizer updates. With 24 training
# recordings an epoch is 2 updates, so a 15-epoch patience stopped the MLP at
# epoch 1 before it learned anything (it predicted "no drone" for every recording).
MLP_MIN_OPTIMIZER_STEPS = 500


def fit_model(
    model_name: str,
    train: pd.DataFrame,
    features: list[str],
    seed: int,
    validation: pd.DataFrame | None = None,
    c_value: float | None = None,
    gamma: str | None = None,
) -> Any:
    x_train = train[features].to_numpy(dtype=float)
    y_train = train["binary_label"].to_numpy(dtype=int)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    if model_name == "svm":
        model = SVC(
            kernel="rbf",
            C=float(c_value),
            gamma=str(gamma),
            class_weight="balanced",
            cache_size=512,
            random_state=seed,
        )
        model.fit(x_train, y_train)
        return model

    if model_name != "mlp":
        raise ValueError(f"Unsupported classical model: {model_name}")
    if validation is None:
        raise ValueError("MLP early stopping requires the fixed validation split")
    model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        batch_size=min(16, max(2, len(train))),
        learning_rate_init=0.001,
        alpha=0.0001,
        max_iter=1,
        early_stopping=False,
        random_state=seed,
        shuffle=True,
    )
    validation_x = validation[features].to_numpy(dtype=float)
    validation_y = validation["binary_label"].to_numpy(dtype=int)
    best_key = (-np.inf, -np.inf, -np.inf)
    best_weights: tuple[list[np.ndarray], list[np.ndarray]] | None = None
    best_metrics: dict[str, Any] | None = None
    patience = 15
    stale_epochs = 0
    steps_per_epoch = -(-len(train) // model.batch_size)
    min_epochs = -(-MLP_MIN_OPTIMIZER_STEPS // steps_per_epoch)
    max_epochs = max(250, min_epochs + 100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for epoch in range(max_epochs):
            model.partial_fit(
                x_train,
                y_train,
                classes=np.array([0, 1]),
                sample_weight=sample_weight,
            )
            validation_metrics, _ = score_frame(model, validation, features)
            key = validation_key(validation_metrics)
            if key > best_key:
                best_key = key
                best_weights = (
                    [weights.copy() for weights in model.coefs_],
                    [bias.copy() for bias in model.intercepts_],
                )
                best_metrics = validation_metrics
                model._qsky_n_iter = epoch + 1
                stale_epochs = 0
            else:
                stale_epochs += 1
            if stale_epochs >= patience and epoch + 1 >= min_epochs:
                break
    if best_weights is None or best_metrics is None:
        raise RuntimeError("MLP training did not produce a validation-selected checkpoint")
    model.coefs_, model.intercepts_ = best_weights
    model._qsky_validation_metrics = best_metrics
    return model


def select_svm(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    seed: int,
) -> tuple[SVC, float, str, dict[str, Any]]:
    best_model: SVC | None = None
    best_c: float | None = None
    best_gamma: str | None = None
    best_metrics: dict[str, Any] | None = None
    best_key = (-np.inf, -np.inf, -np.inf)
    for c_value in SVM_C_VALUES:
        for gamma in SVM_GAMMAS:
            candidate = fit_model(
                "svm", train, features, seed, c_value=c_value, gamma=gamma
            )
            metrics, _ = score_frame(candidate, validation, features)
            candidate_key = validation_key(metrics)
            if candidate_key > best_key:
                best_key = candidate_key
                best_model = candidate
                best_c = c_value
                best_gamma = gamma
                best_metrics = metrics
    assert best_model is not None and best_c is not None and best_gamma is not None
    assert best_metrics is not None
    return best_model, best_c, best_gamma, best_metrics


def run_classical(model_name: str, seed: int = 42) -> dict[str, Any]:
    if model_name not in {"svm", "mlp"}:
        raise ValueError("model_name must be 'svm' or 'mlp'")

    selection, _ = load_feature_sets()
    output_csv = RESULTS_DIR / f"{model_name}_results.csv"
    model_dir = PROJECT_ROOT / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    experiment_rows: list[dict[str, Any]] = []
    sample_id_rows: list[dict[str, Any]] = []
    clean_models: dict[tuple[int, int], tuple[Any, dict[str, Any]]] = {}
    validation_choices: list[dict[str, Any]] = []

    for feature_count in FEATURE_COUNTS:
        names = feature_names(selection, feature_count)
        split_tables = load_scaled_tables(feature_count)
        train_table = split_tables["train"]
        validation_table = split_tables["validation"]
        test_table = split_tables["test"]
        nasa_table = split_tables["nasa_external"]

        scaler_path = PROJECT_ROOT / "models" / f"scaler_{feature_count}.pkl"
        if not scaler_path.is_file():
            raise FileNotFoundError(f"Training-fitted scaler is missing: {scaler_path}")
        if train_table["external_test"].fillna(False).astype(bool).any():
            raise ValueError("NASA external rows are forbidden in training")

        for train_size in TRAIN_SIZES:
            recording_ids, actual_recordings = load_subset_ids(train_size)
            config_models: dict[str, tuple[Any, dict[str, Any]]] = {}

            for train_snr in SNR_CONDITIONS:
                train = training_rows(train_table, recording_ids, train_snr)
                validation = evaluation_rows(validation_table, train_snr)
                if set(train["recording_id"].astype(str)) & set(
                    validation["recording_id"].astype(str)
                ):
                    raise AssertionError("Training subset overlaps validation recordings")
                actual_condition_recordings = int(train["recording_id"].nunique())
                if model_name == "svm":
                    model, c_value, gamma, validation_metrics = select_svm(
                        train, validation, names, seed
                    )
                    hyperparameters = {"C": c_value, "gamma": gamma}
                else:
                    model = fit_model(
                        "mlp", train, names, seed, validation=validation
                    )
                    validation_metrics, _ = score_frame(model, validation, names)
                    hyperparameters = {
                        "hidden_layer_sizes": [16, 8],
                        "activation": "relu",
                        "solver": "adam",
                        "batch_size": 16,
                        "early_stopping": True,
                        "n_iter": int(model._qsky_n_iter),
                        "early_stopping_source": "fixed recording-disjoint validation split",
                    }

                config_models[train_snr] = (model, validation_metrics)
                model_path = (
                    model_dir
                    / f"{model_name}_f{feature_count}_n{train_size}_train_snr_{train_snr}.joblib"
                )
                joblib.dump(model, model_path)
                validation_choices.append(
                    {
                        "model": model_name,
                        "feature_count": feature_count,
                        "train_size_requested": train_size,
                        "train_size_actual": actual_recordings,
                        "train_snr": train_snr,
                        "validation_f1": validation_metrics["f1"],
                        "validation_balanced_accuracy": validation_metrics[
                            "balanced_accuracy"
                        ],
                        "validation_drone_recall": validation_metrics["drone_recall"],
                        "model_path": str(model_path),
                        "hyperparameters": hyperparameters,
                    }
                )
                for sample_id in train["sample_id"].astype(str):
                    sample_id_rows.append(
                        {
                            "model": model_name,
                            "feature_count": feature_count,
                            "requested_training_recordings": train_size,
                            "actual_manifest_recordings": actual_recordings,
                            "actual_condition_recordings": actual_condition_recordings,
                            "train_snr": train_snr,
                            "sample_id": sample_id,
                            "recording_id": str(
                                train.loc[train["sample_id"].astype(str) == sample_id, "recording_id"].iloc[0]
                            ),
                        }
                    )

                test = evaluation_rows(test_table, train_snr)
                test_metrics, _ = score_frame(model, test, names)
                experiment_rows.append(
                    make_result_row(
                        model_name,
                        "matched_snr",
                        feature_count,
                        train_size,
                        actual_recordings,
                        actual_condition_recordings,
                        train_snr,
                        train_snr,
                        validation_metrics,
                        test_metrics,
                        hyperparameters,
                        model_path,
                    )
                )
                if train_snr == "clean":
                    clean_models[(feature_count, train_size)] = (model, validation_metrics)

            clean_model, clean_validation_metrics = config_models["clean"]
            for evaluation_snr in SNR_CONDITIONS:
                test = evaluation_rows(test_table, evaluation_snr)
                test_metrics, _ = score_frame(clean_model, test, names)
                experiment_rows.append(
                    make_result_row(
                        model_name,
                        "clean_model_noise_robustness",
                        feature_count,
                        train_size,
                        actual_recordings,
                        int(
                            training_rows(train_table, recording_ids, "clean")[
                                "recording_id"
                            ].nunique()
                        ),
                        "clean",
                        evaluation_snr,
                        clean_validation_metrics,
                        test_metrics,
                        {"trained_snr": "clean"},
                        model_dir
                        / f"{model_name}_f{feature_count}_n{train_size}_train_snr_clean.joblib",
                    )
                )

    # Choose the final external model using clean validation results only.
    clean_choices = [choice for choice in validation_choices if choice["train_snr"] == "clean"]
    best_choice = max(
        clean_choices,
        key=lambda row: (
            row["validation_f1"],
            row["validation_balanced_accuracy"],
            row["validation_drone_recall"],
            row["train_size_actual"],
        ),
    )
    final_model = joblib.load(best_choice["model_path"])
    final_feature_names = feature_names(selection, int(best_choice["feature_count"]))
    nasa_metrics = run_nasa_external(
        final_model,
        load_scaled_tables(int(best_choice["feature_count"]))["nasa_external"],
        final_feature_names,
    )
    nasa_row = {
        "model": model_name,
        "experiment": "nasa_external_once",
        "feature_count": best_choice["feature_count"],
        "train_size_requested": best_choice["train_size_requested"],
        "train_size_actual": best_choice["train_size_actual"],
        "actual_condition_recordings": best_choice["train_size_actual"],
        "train_snr": "clean",
        "evaluation_snr": "clean",
        "evaluation_split": "nasa_external",
        "validation_f1": best_choice["validation_f1"],
        "validation_balanced_accuracy": best_choice["validation_balanced_accuracy"],
        "validation_drone_recall": best_choice["validation_drone_recall"],
        "hyperparameters": best_choice["hyperparameters"],
        "model_path": best_choice["model_path"],
        **nasa_metrics,
        "test_sample_count": int(len(load_scaled_tables(int(best_choice["feature_count"]))["nasa_external"])),
    }
    experiment_rows.append(nasa_row)

    # Select one validation-best model per feature-count/training-size, across training SNR.
    result_table = pd.DataFrame(experiment_rows)
    best_by_configuration: list[dict[str, Any]] = []
    for (feature_count, requested_size), group in result_table[
        (result_table["experiment"] == "matched_snr")
        & (result_table["evaluation_split"] == "test")
    ].groupby(["feature_count", "train_size_requested"]):
        candidates = group.sort_values(
            ["validation_f1", "validation_balanced_accuracy", "validation_drone_recall"],
            ascending=False,
            kind="stable",
        )
        chosen = candidates.iloc[0]
        best_by_configuration.append(
            {
                "model": model_name,
                "feature_count": int(feature_count),
                "train_size_requested": int(requested_size),
                "selected_train_snr": str(chosen["train_snr"]),
                "selected_validation_f1": float(chosen["validation_f1"]),
                "selected_model_path": str(chosen["model_path"]),
            }
        )
        source_model = joblib.load(str(chosen["model_path"]))
        joblib.dump(
            source_model,
            model_dir / f"best_f{int(feature_count)}_n{int(requested_size)}.joblib",
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_table.to_csv(output_csv, index=False)
    sample_ids_path = RESULTS_DIR / f"{model_name}_training_sample_ids.csv"
    pd.DataFrame(sample_id_rows).to_csv(sample_ids_path, index=False)
    per_model_id_paths = [
        RESULTS_DIR / f"{name}_training_sample_ids.csv" for name in ("svm", "mlp")
    ]
    available_manifests = [path for path in per_model_id_paths if path.is_file()]
    if available_manifests:
        pd.concat(
            [pd.read_csv(path) for path in available_manifests], ignore_index=True
        ).to_csv(RESULTS_DIR / "classical_training_sample_ids.csv", index=False)
    best_summary_path = RESULTS_DIR / f"best_{model_name}_configuration.json"
    best_summary_path.write_text(
        json.dumps(
            {
                "selected_on": "internal validation only; clean training condition",
                "best_external_model": best_choice,
                "best_per_feature_and_training_size": best_by_configuration,
                "nasa_metrics_or_confidence": nasa_metrics,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "results_path": output_csv,
        "best_summary_path": best_summary_path,
        "best_choice": best_choice,
        "nasa_metrics": nasa_metrics,
        "result_rows": len(result_table),
    }


def make_result_row(
    model_name: str,
    experiment: str,
    feature_count: int,
    requested_size: int,
    actual_manifest_recordings: int,
    actual_condition_recordings: int,
    train_snr: str,
    evaluation_snr: str,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    hyperparameters: dict[str, Any],
    model_path: Path,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "experiment": experiment,
        "feature_count": feature_count,
        "train_size_requested": requested_size,
        "train_size_actual": actual_manifest_recordings,
        "actual_condition_recordings": actual_condition_recordings,
        "train_snr": train_snr,
        "evaluation_snr": evaluation_snr,
        "evaluation_split": "test",
        "validation_f1": validation_metrics["f1"],
        "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        "validation_drone_recall": validation_metrics["drone_recall"],
        "hyperparameters": json.dumps(hyperparameters, sort_keys=True),
        "model_path": str(model_path),
        **test_metrics,
    }


def build_parser(model_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Train the CPU-only {model_name.upper()} baseline.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser


def main(model_name: str) -> None:
    args = build_parser(model_name).parse_args()
    result = run_classical(model_name, seed=args.seed)
    print(f"Wrote {result['result_rows']} {model_name.upper()} result rows to {result['results_path']}")
    print(f"Validation-selected model: {result['best_choice']}")
    print(f"NASA external output: {result['nasa_metrics']}")