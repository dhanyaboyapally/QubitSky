"""Stage 6: controlled local Aer noise sensitivity for the frozen Stage 5 model.

No IBM Runtime, account, backend, credential, or hardware-job APIs are imported.
Every circuit in this module executes only on a local AerSimulator.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit.primitives import BackendSamplerV2
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from sklearn.svm import SVC

from config import FEATURES_DIR, MODELS_DIR, PROJECT_ROOT, RANDOM_SEED, RESULTS_DIR
from quantum_kernel_simulator import (
    _aggregate_for_snr,
    _aggregate_recordings,
    _circuit_stats,
    _ids_hash,
    _load_stage_assets,
    _make_trainable_map,
    _record_map,
    _validate_scaler,
)
from classical_utils import calculate_metrics, feature_names


QUANTUM_RESULTS_DIR = RESULTS_DIR / "quantum"
NOISY_RESULTS_PATH = QUANTUM_RESULTS_DIR / "noisy_simulator_results.csv"
NOISY_KERNEL_DIR = QUANTUM_RESULTS_DIR / "noisy_kernels"
NOISE_METADATA_PATH = QUANTUM_RESULTS_DIR / "noise_model_metadata.json"
THRESHOLD_PATH = QUANTUM_RESULTS_DIR / "noise_threshold_summary.json"
FROZEN_CONFIG_PATH = MODELS_DIR / "quantum" / "stage7_frozen_config.json"
PLOT_DIR = RESULTS_DIR / "plots"

ERROR_LEVELS = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02)
SHOTS = 1024
INITIAL_NOISE_SEED = 7300
MAX_FIDELITY_CIRCUITS = 100_000
CLASSICAL_REFERENCE = {
    "svm_f1": 0.492,
    "svm_balanced_accuracy": 0.786,
    "mlp_f1": 0.625,
    "mlp_balanced_accuracy": 0.776,
}


def _read_frozen_inputs(seed: int) -> dict[str, Any]:
    data = _load_stage_assets(4, 24, seed)
    frozen_path = MODELS_DIR / "quantum_ideal" / "trainable_params_q4_n24_snr_clean_seed42.json"
    if not frozen_path.is_file():
        raise FileNotFoundError(f"Frozen Stage 5 trainable kernel parameters missing: {frozen_path}")
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    expected_hash = _ids_hash(data["train"]["recording_id"].astype(str).tolist())
    if frozen.get("training_recording_ids_sha256") != expected_hash:
        raise ValueError("Stage 5 trained kernel parameters do not match the current 24-recording subset")
    theta = np.asarray(frozen["optimal_parameters"], dtype=float)
    if theta.shape != (4,):
        raise ValueError(f"Expected four frozen trainable parameters, got shape {theta.shape}")
    if len(data["train"]) != 24 or data["train"].binary_label.value_counts().to_dict() != {0: 12, 1: 12}:
        raise ValueError("Stage 6 must use exactly 24 balanced training recording groups")

    manifest_path = QUANTUM_RESULTS_DIR / "stage5_sample_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_train_ids = sorted(data["train"].recording_id.astype(str).tolist())
    training_entries = manifest.get("training_recordings", manifest.get("training", []))
    validation_entries = manifest.get("validation_recordings", manifest.get("validation", []))
    test_entries = manifest.get("test_recordings", manifest.get("test", []))
    frozen_train_ids = sorted(str(record["recording_id"]) for record in training_entries)
    if expected_train_ids != frozen_train_ids:
        raise ValueError("Current Stage 3 training IDs differ from frozen Stage 5 manifest")
    intersections = (
        manifest.get("train_test_recording_intersection", manifest.get("train_test_overlap", [])),
        manifest.get("train_validation_recording_intersection", manifest.get("train_validation_overlap", [])),
        manifest.get("validation_test_recording_intersection", manifest.get("validation_test_overlap", [])),
    )
    if any(intersections):
        raise ValueError("Stage 5 manifest records split leakage")
    saved_training_sample_ids = sorted(
        sample_id
        for record in training_entries
        for sample_id in record.get("sample_ids", [])
    )
    current_training_sample_ids = sorted(
        sample_id
        for sample_ids in data["train"]["sample_ids"]
        for sample_id in json.loads(sample_ids)
    )
    if saved_training_sample_ids != current_training_sample_ids:
        raise ValueError("Current Stage 3 training sample IDs differ from frozen Stage 5 manifest")
    if validation_entries:
        saved_validation_ids = sorted(str(record["recording_id"]) for record in validation_entries)
        current_validation_ids = sorted(data["validation"]["recording_id"].astype(str).tolist())
        if saved_validation_ids != current_validation_ids:
            raise ValueError("Current validation recording IDs differ from frozen Stage 5 manifest")
    if isinstance(test_entries, dict):
        saved_test_entries = test_entries.get("clean", [])
    else:
        saved_test_entries = test_entries
    if saved_test_entries:
        saved_test_ids = sorted(str(record["recording_id"]) for record in saved_test_entries)
        current_test_ids = sorted(data["tests_by_snr"]["clean"]["recording_id"].astype(str).tolist())
        if saved_test_ids != current_test_ids:
            raise ValueError("Current test recording IDs differ from frozen Stage 5 manifest")

    stage5_results = pd.read_csv(QUANTUM_RESULTS_DIR / "ideal_simulator_results.csv")
    ideal = stage5_results[
        (stage5_results.model == "trainable_qsvc")
        & (stage5_results.feature_count == 4)
        & (stage5_results.training_size_requested == 24)
        & (stage5_results.training_snr.astype(str) == "clean")
        & (stage5_results.evaluation_snr.astype(str) == "clean")
        & (stage5_results.evaluation_split == "test")
    ]
    if len(ideal) != 1:
        raise ValueError("Expected exactly one Stage 5 trainable 4-qubit initial test row")
    return {
        **data,
        "theta": theta,
        "frozen_parameter_record": frozen,
        "ideal_metrics": ideal.iloc[0].to_dict(),
        "manifest": manifest,
    }


def _make_noise_model(
    error_level: float,
    error_type: str = "combined",
    seed: int = INITIAL_NOISE_SEED,
) -> tuple[NoiseModel, dict[str, Any]]:
    if error_type not in {"combined", "gate_only", "readout_only"}:
        raise ValueError(f"Unknown noise type: {error_type}")
    if not 0.0 <= error_level <= 0.02:
        raise ValueError("Synthetic error level must be in [0, 0.02]")

    if error_type == "combined":
        one_qubit_error = error_level / 10.0
        two_qubit_error = error_level
        readout_probability = error_level / 2.0
    elif error_type == "gate_only":
        one_qubit_error = error_level / 10.0
        two_qubit_error = error_level
        readout_probability = 0.0
    else:
        one_qubit_error = 0.0
        two_qubit_error = 0.0
        readout_probability = error_level / 2.0

    noise_model = NoiseModel()
    if one_qubit_error > 0:
        error = depolarizing_error(one_qubit_error, 1)
        noise_model.add_all_qubit_quantum_error(error, ["u"])
    if two_qubit_error > 0:
        error = depolarizing_error(two_qubit_error, 2)
        noise_model.add_all_qubit_quantum_error(error, ["cx"])
    if readout_probability > 0:
        readout = ReadoutError(
            [
                [1.0 - readout_probability, readout_probability],
                [readout_probability, 1.0 - readout_probability],
            ]
        )
        noise_model.add_all_qubit_readout_error(readout)

    metadata = {
        "noise_type": error_type,
        "noise_label": f"{error_level * 100:g}pct" if error_type == "combined" else f"{error_type}_{error_level * 100:g}pct",
        "error_level_interpretation": "target two-qubit depolarizing probability for combined/gate-only; target readout probability is half the label for readout-only",
        "single_qubit_error": one_qubit_error,
        "two_qubit_error": two_qubit_error,
        "readout_error": readout_probability,
        "shots": SHOTS,
        "seed": seed,
        "fidelity_definition": "global state fidelity; matched to Stage 5 ideal simulator",
        "method": "Aer depolarizing errors on transpiled u/cx gates; symmetric independent readout flips",
    }
    return noise_model, metadata


def _kernel_cache_path(
    matrix_name: str,
    noise_label: str,
    shots: int,
    seed: int,
    theta_hash: str,
) -> Path:
    return NOISY_KERNEL_DIR / (
        f"{matrix_name}_4q_24train_noise_{noise_label}_{shots}shots_"
        f"trainable_product_reps1_linear_seed{seed}_{theta_hash}.npy"
    )


def _cache_fingerprint(
    cache_path: Path,
    left: np.ndarray,
    right: np.ndarray | None,
    noise_metadata: dict[str, Any],
    theta: np.ndarray,
    sample_ids_hash: str,
) -> str:
    payload = {
        "left": hashlib.sha256(np.ascontiguousarray(left, dtype=np.float64).tobytes()).hexdigest(),
        "right": None
        if right is None
        else hashlib.sha256(np.ascontiguousarray(right, dtype=np.float64).tobytes()).hexdigest(),
        "noise": noise_metadata,
        "theta": theta.tolist(),
        "sample_ids_hash": sample_ids_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _evaluate_cached_matrix(
    kernel: FidelityQuantumKernel,
    cache_path: Path,
    left: np.ndarray,
    right: np.ndarray | None,
    noise_metadata: dict[str, Any],
    theta: np.ndarray,
    sample_ids_hash: str,
) -> tuple[np.ndarray, bool, float, int]:
    fingerprint = _cache_fingerprint(
        cache_path, left, right, noise_metadata, theta, sample_ids_hash
    )
    sidecar = cache_path.with_suffix(".json")
    if cache_path.is_file() and sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            return np.load(cache_path, allow_pickle=False), True, 0.0, 0

    before = time.perf_counter()
    matrix = kernel.evaluate(left, right)
    duration = time.perf_counter() - before
    if right is None:
        evaluations = len(left) * (len(left) - 1) // 2
    else:
        evaluations = len(left) * len(right)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix, allow_pickle=False)
    sidecar.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "cache_path": str(cache_path),
                "shape": list(matrix.shape),
                "noise_metadata": noise_metadata,
                "theta": theta.tolist(),
                "sample_ids_hash": sample_ids_hash,
                "kernel_evaluations": evaluations,
                "computed_seconds": duration,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return matrix, False, duration, evaluations


def _trainable_feature_map(theta: np.ndarray) -> QuantumCircuit:
    circuit, training_parameters = _make_trainable_map(4)
    parameter_values = dict(zip(training_parameters, theta.tolist()))
    circuit = circuit.assign_parameters(parameter_values, inplace=False)
    return circuit


def _noise_configuration(
    noise_label: str,
    error_level: float,
    noise_type: str,
    seed: int,
    data: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    features = data["features"]
    train = data["train"]
    validation = data["validation"]
    test = data["tests_by_snr"]["clean"]
    theta = data["theta"]
    n_train, n_validation, n_test = len(train), len(validation), len(test)
    estimated = n_train * (n_train - 1) // 2 + n_validation * n_train + n_test * n_train
    print(
        f"NOISE {noise_label}: estimated fidelity circuits={estimated:,}; "
        f"shots={SHOTS}; cache status checked per matrix; expected runtime depends on Aer CPU speed"
    )

    noise_model, metadata = _make_noise_model(error_level, noise_type, seed)
    backend = AerSimulator(method="statevector", noise_model=noise_model, seed_simulator=seed)
    sampler = BackendSamplerV2(
        backend=backend,
        options={"default_shots": SHOTS, "seed_simulator": seed},
    )
    fidelity = ComputeUncompute(sampler=sampler, local=False)
    circuit = _trainable_feature_map(theta)
    kernel = FidelityQuantumKernel(
        feature_map=circuit,
        fidelity=fidelity,
        enforce_psd=True,
        evaluate_duplicates="off_diagonal",
        max_circuits_per_job=250,
    )
    train_x = train[features].to_numpy(dtype=float)
    validation_x = validation[features].to_numpy(dtype=float)
    test_x = test[features].to_numpy(dtype=float)
    train_y = train.binary_label.to_numpy(dtype=int)
    matrices: dict[str, np.ndarray] = {}
    cache_hits: dict[str, bool] = {}
    timings: dict[str, float] = {}
    evaluations: dict[str, int] = {}
    arrays = {
        "training": (train_x, None, _ids_hash(train.recording_id.astype(str).tolist())),
        "validation": (
            validation_x,
            train_x,
            _ids_hash(validation.recording_id.astype(str).tolist() + train.recording_id.astype(str).tolist()),
        ),
        "test": (
            test_x,
            train_x,
            _ids_hash(test.recording_id.astype(str).tolist() + train.recording_id.astype(str).tolist()),
        ),
    }
    theta_hash = hashlib.sha256(np.ascontiguousarray(theta, dtype=np.float64).tobytes()).hexdigest()[:10]
    for matrix_name, (left, right, sample_hash) in arrays.items():
        cache_path = _kernel_cache_path(
            matrix_name, noise_label, SHOTS, seed, theta_hash
        )
        matrix, hit, duration, count = _evaluate_cached_matrix(
            kernel,
            cache_path,
            left,
            right,
            metadata,
            theta,
            sample_hash,
        )
        matrices[matrix_name] = matrix
        cache_hits[matrix_name] = hit
        timings[matrix_name] = duration
        evaluations[matrix_name] = count
        print(f"  {matrix_name} kernel: {'cached' if hit else 'computed'} ({duration:.2f}s)")

    metadata.update(
        {
            "noise_label": noise_label,
            "single_qubit_error": metadata["single_qubit_error"],
            "two_qubit_error": metadata["two_qubit_error"],
            "readout_error": metadata["readout_error"],
            "shots": SHOTS,
            "seed": seed,
            "cache_hits": cache_hits,
            "kernel_times_seconds": timings,
            "kernel_evaluations": evaluations,
            "estimated_fidelity_circuits": estimated,
            "training_recording_ids_sha256": _ids_hash(train.recording_id.astype(str).tolist()),
            "validation_recording_ids_sha256": _ids_hash(validation.recording_id.astype(str).tolist()),
            "test_recording_ids_sha256": _ids_hash(test.recording_id.astype(str).tolist()),
            "trainable_parameters": theta.tolist(),
            "feature_map": "Stage 5 frozen trainable product feature map; parameterized values bound from Stage 5",
        }
    )

    model = SVC(kernel="precomputed", C=0.1, class_weight="balanced")
    training_start = time.perf_counter()
    model.fit(matrices["training"], train_y)
    training_seconds = time.perf_counter() - training_start
    model_path = MODELS_DIR / "quantum_noise" / f"qsvc_4q24_{noise_label}_{SHOTS}shots_seed{seed}.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata["classifier_fit_seconds"] = training_seconds
    metadata["fitted_model_path"] = str(model_path)
    return metadata, matrices, arrays


def _evaluation_rows(
    label: str,
    error_level: float,
    noise_metadata: dict[str, Any],
    matrices: dict[str, np.ndarray],
    arrays: dict[str, tuple[np.ndarray, np.ndarray | None, str]],
    data: dict[str, Any],
) -> list[dict[str, Any]]:
    train = data["train"]
    validation = data["validation"]
    test = data["tests_by_snr"]["clean"]
    train_y = train.binary_label.to_numpy(dtype=int)
    validation_y = validation.binary_label.to_numpy(dtype=int)
    test_y = test.binary_label.to_numpy(dtype=int)
    model = SVC(kernel="precomputed", C=0.1, class_weight="balanced")
    model.fit(matrices["training"], train_y)

    def predict(matrix: np.ndarray, truth: np.ndarray) -> tuple[dict[str, Any], float]:
        started = time.perf_counter()
        prediction = model.predict(matrix)
        scores = model.decision_function(matrix)
        duration = time.perf_counter() - started
        return calculate_metrics(truth, prediction, scores), duration

    validation_metrics, validation_prediction_seconds = predict(
        matrices["validation"], validation_y
    )
    test_metrics, test_prediction_seconds = predict(matrices["test"], test_y)
    total_kernel_seconds = sum(noise_metadata["kernel_times_seconds"].values())
    total_kernel_evaluations = sum(noise_metadata["kernel_evaluations"].values())
    feature_map = _trainable_feature_map(data["theta"])
    depth, two_qubit_gates = _circuit_stats(feature_map)
    result_row = {
        "stage": 6,
        "model": "frozen_trainable_qsvc",
        "noise_label": label,
        "noise_type": noise_metadata["noise_type"],
        "error_level": error_level,
        "single_qubit_error": noise_metadata["single_qubit_error"],
        "two_qubit_error": noise_metadata["two_qubit_error"],
        "readout_error": noise_metadata["readout_error"],
        "shots": SHOTS,
        "seed": noise_metadata["seed"],
        "qubit_count": 4,
        "training_recordings": len(train),
        "training_class_0": int((train_y == 0).sum()),
        "training_class_1": int((train_y == 1).sum()),
        "training_snr": "clean",
        "evaluation_snr": "clean",
        "evaluation_split": "test",
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_f1": validation_metrics["f1"],
        "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
        "validation_drone_recall": validation_metrics["drone_recall"],
        **test_metrics,
        "circuit_depth": depth,
        "two_qubit_gate_count": two_qubit_gates,
        "estimated_fidelity_circuits": noise_metadata["estimated_fidelity_circuits"],
        "kernel_evaluations": total_kernel_evaluations,
        "kernel_training_evaluations": noise_metadata["kernel_evaluations"]["training"],
        "kernel_validation_evaluations": noise_metadata["kernel_evaluations"]["validation"],
        "kernel_test_evaluations": noise_metadata["kernel_evaluations"]["test"],
        "kernel_computation_seconds": total_kernel_seconds,
        "training_seconds": noise_metadata["classifier_fit_seconds"],
        "prediction_seconds": validation_prediction_seconds + test_prediction_seconds,
        "kernel_cache_hits": json.dumps(noise_metadata["cache_hits"], sort_keys=True),
        "training_recording_ids_sha256": noise_metadata["training_recording_ids_sha256"],
        "validation_recording_ids_sha256": noise_metadata["validation_recording_ids_sha256"],
        "test_recording_ids_sha256": noise_metadata["test_recording_ids_sha256"],
        "model_path": noise_metadata["fitted_model_path"],
    }
    return [result_row]


def _classical_ideal_baselines() -> dict[str, dict[str, float]]:
    # Matched Stage 5 recording-aggregated results at 4 features / 24 groups / clean.
    results = pd.read_csv(RESULTS_DIR / "quantum" / "ideal_simulator_results.csv")
    rows = results[
        (results.feature_count == 4)
        & (results.training_size_requested == 24)
        & (results.evaluation_snr.astype(str) == "clean")
        & (results.evaluation_split == "test")
        & results.model.isin(["rbf_svm", "small_mlp", "trainable_qsvc"])
    ]
    baselines: dict[str, dict[str, float]] = {}
    for model_name, frame in rows.groupby("model"):
        row = frame.iloc[0]
        baselines[str(model_name)] = {
            "accuracy": float(row.accuracy),
            "f1": float(row.f1),
            "balanced_accuracy": float(row.balanced_accuracy),
            "drone_recall": float(row.drone_recall),
            "non_drone_recall": float(row.non_drone_recall),
        }
    required = {"rbf_svm", "small_mlp", "trainable_qsvc"}
    if not required.issubset(baselines):
        raise ValueError(f"Missing Stage 5 baseline rows: {sorted(required-set(baselines))}")
    return baselines


def _threshold_summary(results: pd.DataFrame, baselines: dict[str, dict[str, float]]) -> dict[str, Any]:
    combined = results[(results.noise_type == "combined") & (results.noise_label.str.endswith("pct"))].copy()
    combined = combined.sort_values("error_level")
    mlp_f1 = baselines["small_mlp"]["f1"]
    svm_f1 = baselines["rbf_svm"]["f1"]
    classical_best_balanced = max(
        baselines["small_mlp"]["balanced_accuracy"],
        baselines["rbf_svm"]["balanced_accuracy"],
    )

    def crossing(metric: str, threshold: float) -> dict[str, Any]:
        passing = combined[combined[metric] >= threshold]
        failing = combined[combined[metric] < threshold]
        if passing.empty:
            return {"status": "below_threshold_at_zero_or_all_levels", "largest_passing_error": None, "first_failing_error": float(failing.iloc[0].error_level) if not failing.empty else None}
        if failing.empty:
            return {"status": "still_meets_threshold_at_highest_tested_error", "largest_passing_error": float(passing.error_level.max()), "first_failing_error": None}
        largest_pass = float(passing.error_level.max())
        first_fail = float(failing[failing.error_level > largest_pass].error_level.min())
        return {
            "status": "crossing_interval_measured",
            "largest_passing_error": largest_pass,
            "first_failing_error": first_fail,
            "interpretation": f"meets at {largest_pass*100:g}% and falls below by {first_fail*100:g}% under this synthetic model",
        }

    return {
        "classical_baselines_from_stage5_matched_aggregated_test": baselines,
        "quantum_competitive_thresholds": {
            "f1_at_least_mlp": {"threshold_f1": mlp_f1, **crossing("f1", mlp_f1)},
            "f1_at_least_svm": {"threshold_f1": svm_f1, **crossing("f1", svm_f1)},
            "balanced_accuracy_at_least_best_classical": {
                "threshold_balanced_accuracy": classical_best_balanced,
                **crossing("balanced_accuracy", classical_best_balanced),
            },
        },
        "tested_combined_error_levels": [float(level) for level in combined.error_level],
        "threshold_resolution_note": "Intervals are bounded only by tested synthetic noise levels; no finer threshold is claimed.",
    }


def _make_plots(results: pd.DataFrame, baselines: dict[str, dict[str, float]]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    combined = results[(results.noise_type == "combined") & (results.noise_label.str.endswith("pct"))].sort_values("error_level")
    plot_specs = [
        ("f1", "F1", "quantum_f1_vs_synthetic_error.png", baselines["small_mlp"]["f1"], baselines["rbf_svm"]["f1"]),
        ("balanced_accuracy", "Balanced accuracy", "quantum_balanced_accuracy_vs_synthetic_error.png", baselines["small_mlp"]["balanced_accuracy"], baselines["rbf_svm"]["balanced_accuracy"]),
        ("accuracy", "Accuracy", "quantum_accuracy_vs_synthetic_error.png", baselines["small_mlp"]["accuracy"], baselines["rbf_svm"]["accuracy"]),
        ("drone_recall", "Drone recall", "quantum_drone_recall_vs_synthetic_error.png", baselines["small_mlp"]["drone_recall"], baselines["rbf_svm"]["drone_recall"]),
    ]
    for metric, ylabel, filename, mlp_line, svm_line in plot_specs:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
        ax.plot(combined.error_level * 100, combined[metric], marker="o", color="#176b87", label="Trainable QSVC, Aer")
        ax.axhline(mlp_line, color="#d95f02", linestyle="--", label="Matched MLP, ideal Stage 5")
        ax.axhline(svm_line, color="#1b9e77", linestyle=":", label="Matched RBF SVM, ideal Stage 5")
        ax.set_xlabel("Synthetic error rate label (%)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Noisy-simulator quantum {ylabel.lower()}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(PLOT_DIR / filename, dpi=180)
        plt.close(fig)

    ideal_row = baselines["trainable_qsvc"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    x = [0.0] + (combined.error_level.to_numpy(dtype=float) * 100).tolist()
    y = [ideal_row["f1"]] + combined.f1.astype(float).tolist()
    ax.plot(x, y, marker="o", color="#176b87", label="Ideal Stage 5 + Aer noisy kernel")
    ax.axhline(baselines["small_mlp"]["f1"], color="#d95f02", linestyle="--", label="Matched MLP")
    ax.axhline(baselines["rbf_svm"]["f1"], color="#1b9e77", linestyle=":", label="Matched RBF SVM")
    ax.set_xlabel("Synthetic error rate label (%)")
    ax.set_ylabel("F1")
    ax.set_title("Ideal vs finite-shot noisy quantum F1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.savefig(PLOT_DIR / "ideal_vs_noisy_quantum.png", dpi=180)
    plt.close(fig)


def _make_stage7_frozen_config(
    data: dict[str, Any],
    results: pd.DataFrame,
    threshold_summary: dict[str, Any],
    seed: int,
) -> None:
    train = data["train"]
    test = data["tests_by_snr"]["clean"]
    stage5 = pd.read_csv(RESULTS_DIR / "quantum" / "ideal_simulator_results.csv")
    ideal = stage5[
        (stage5.model == "trainable_qsvc")
        & (stage5.feature_count == 4)
        & (stage5.training_size_requested == 24)
        & (stage5.training_snr.astype(str) == "clean")
        & (stage5.evaluation_snr.astype(str) == "clean")
        & (stage5.evaluation_split == "test")
    ].iloc[0]
    error_level_recommendation = 0.0
    passing = results[
        (results.noise_type == "combined")
        & (results.noise_label.str.endswith("pct"))
        & (results.f1 >= threshold_summary["classical_baselines_from_stage5_matched_aggregated_test"]["small_mlp"]["f1"])
    ]
    if not passing.empty:
        error_level_recommendation = float(passing.error_level.max())
    config = {
        "status": "prepared_not_submitted",
        "hardware_submission_allowed": False,
        "stage7_requires_user_command": "RUN REAL QPU",
        "feature_names": data["features"],
        "scaler_path": str(MODELS_DIR / "scaler_4.pkl"),
        "scaler_sha256": data["scaler_sha256"],
        "training_recording_ids": train.recording_id.astype(str).tolist(),
        "training_sample_ids": [sample_id for item in train.sample_ids for sample_id in json.loads(item)],
        "test_recording_ids": test.recording_id.astype(str).tolist(),
        "test_sample_ids": [sample_id for item in test.sample_ids for sample_id in json.loads(item)],
        "feature_map": "Stage 5 trainable product feature map, reps=1, linear nearest-neighbor entanglement",
        "trained_kernel_parameters": data["theta"].tolist(),
        "qubit_count": 4,
        "shots_recommendation": 1024,
        "seed": seed,
        "ideal_stage5_result": {
            key: float(ideal[key])
            for key in ("accuracy", "f1", "balanced_accuracy", "drone_recall", "non_drone_recall")
        },
        "noisy_simulator_summary": {
            "best_combined_error_meeting_mlp_f1": error_level_recommendation,
            "thresholds": threshold_summary["quantum_competitive_thresholds"],
        },
        "note": "No IBM backend, credentials, runtime service, or hardware job was used to create this configuration.",
    }
    FROZEN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    FROZEN_CONFIG_PATH.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def run_stage6(
    shots: int = SHOTS,
    seed: int = RANDOM_SEED,
    error_levels: tuple[float, ...] = ERROR_LEVELS,
    run_ablation: bool = True,
) -> dict[str, Any]:
    global SHOTS
    if shots not in {512, 1024}:
        raise ValueError("Stage 6 supports 512 or 1024 shots")
    SHOTS = shots
    data = _read_frozen_inputs(seed=42)
    theta_hash = hashlib.sha256(np.ascontiguousarray(data["theta"], dtype=np.float64).tobytes()).hexdigest()
    if not np.array_equal(data["theta"], np.asarray(data["frozen_parameter_record"]["optimal_parameters"])):
        raise AssertionError("Stage 6 parameters differ from the frozen Stage 5 values")

    rows: list[dict[str, Any]] = []
    noise_metadata_records: list[dict[str, Any]] = []
    for index, level in enumerate(error_levels):
        label = f"{level * 100:g}pct"
        noise_seed = seed + index
        metadata, matrices, arrays = _noise_configuration(
            label, level, "combined", noise_seed, data
        )
        noise_metadata_records.append(metadata)
        rows.extend(
            _evaluation_rows(label, level, metadata, matrices, arrays, data)
        )
        partial = pd.DataFrame(rows)
        QUANTUM_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        partial.to_csv(NOISY_RESULTS_PATH, index=False)
        NOISE_METADATA_PATH.write_text(
            json.dumps(noise_metadata_records, indent=2) + "\n", encoding="utf-8"
        )

    if run_ablation:
        for error_type in ("gate_only", "readout_only"):
            label = f"{error_type}_0p5pct"
            noise_seed = seed + 100 + (0 if error_type == "gate_only" else 1)
            metadata, matrices, arrays = _noise_configuration(
                label, 0.005, error_type, noise_seed, data
            )
            noise_metadata_records.append(metadata)
            rows.extend(
                _evaluation_rows(label, 0.005, metadata, matrices, arrays, data)
            )
            pd.DataFrame(rows).to_csv(NOISY_RESULTS_PATH, index=False)
            NOISE_METADATA_PATH.write_text(
                json.dumps(noise_metadata_records, indent=2) + "\n", encoding="utf-8"
            )

    results = pd.DataFrame(rows)
    baselines = _classical_ideal_baselines()
    threshold_summary = _threshold_summary(results, baselines)
    threshold_summary["trainable_stage5_ideal_baseline"] = baselines["trainable_qsvc"]
    threshold_summary["noise_model_definition"] = {
        "error_level_scale": "p2 equals the requested combined/gate-only label; p1=p2/10; readout=p2/2. Readout-only uses p_readout=label/2.",
        "gate_noise": "Aer depolarizing_error attached to transpiled u gates and cx gates",
        "readout_noise": "symmetric independent bit-flip ReadoutError attached to every qubit",
        "fidelity_definition": "global state fidelity; matched to Stage 5 ideal simulator",
        "shots": shots,
        "not_hardware_claim": "Controlled synthetic sensitivity model, not a characterization of any IBM backend.",
        "seed_base": seed,
    }
    THRESHOLD_PATH.write_text(
        json.dumps(threshold_summary, indent=2) + "\n", encoding="utf-8"
    )
    _make_plots(results, baselines)
    _make_stage7_frozen_config(data, results, threshold_summary, seed)
    print("NOISY SIMULATOR SUMMARY")
    print(
        results[results.noise_type == "combined"]
        [["noise_label", "accuracy", "f1", "balanced_accuracy", "drone_recall", "non_drone_recall", "kernel_computation_seconds"]]
        .to_string(index=False)
    )
    print("THRESHOLDS")
    print(json.dumps(threshold_summary["quantum_competitive_thresholds"], indent=2))
    print("ERROR ABLATION")
    print(
        results[results.noise_type.isin(["gate_only", "readout_only"])]
        [["noise_label", "f1", "balanced_accuracy", "drone_recall", "non_drone_recall"]]
        .to_string(index=False)
    )
    print(f"Saved noisy results: {NOISY_RESULTS_PATH}")
    print(f"Saved kernel cache: {NOISY_KERNEL_DIR}")
    print(f"Saved Stage 7 frozen config (not submitted): {FROZEN_CONFIG_PATH}")
    return {
        "results": results,
        "thresholds": threshold_summary,
        "baselines": baselines,
        "frozen_stage7_config": FROZEN_CONFIG_PATH,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, choices=(512, 1024), default=SHOTS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip the two representative gate-only/readout-only 0.5% ablation cases.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_stage6(args.shots, args.seed, ERROR_LEVELS, not args.skip_ablation)


if __name__ == "__main__":
    main()