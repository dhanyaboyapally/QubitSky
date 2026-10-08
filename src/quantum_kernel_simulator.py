"""Stage 5: ideal local-simulator quantum-kernel comparison only.

This module deliberately contains no IBM Runtime, backend, credential, or job
submission imports. It uses local statevector fidelity evaluation only.
"""

import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import (
    FidelityQuantumKernel,
    TrainableFidelityQuantumKernel,
)
from qiskit_machine_learning.kernels.algorithms import QuantumKernelTrainer
from qiskit_machine_learning.optimizers import SPSA
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.utils.loss_functions import SVCLoss
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from config import FEATURES_DIR, MODELS_DIR, PROJECT_ROOT, RANDOM_SEED, RESULTS_DIR
from classical_utils import (
    calculate_metrics,
    feature_names,
    load_subset_ids,
    score_frame,
    validation_key,
)


FEATURE_COUNTS = (4, 5, 6)
TRAIN_SIZES = (24, 50, 70)
SNR_CONDITIONS = ("clean", "20", "10", "5", "0")
MAX_PAIR_EVALUATIONS_PER_CONFIGURATION = 100_000
RESULTS_PATH = RESULTS_DIR / "quantum" / "ideal_simulator_results.csv"
KERNEL_CACHE_DIR = RESULTS_DIR / "quantum" / "kernels"
MODEL_DIR = MODELS_DIR / "quantum_ideal"
MANIFEST_PATH = RESULTS_DIR / "quantum" / "stage5_sample_manifest.json"


def _matrix_hash(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _ids_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _aggregate_recordings(table: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if table.empty:
        raise ValueError("Cannot aggregate an empty feature table")
    if table["external_test"].fillna(False).astype(bool).any():
        raise ValueError("NASA external rows are forbidden in Stage 5 model selection")
    if table["recording_id"].isna().any():
        raise ValueError("Feature rows need recording_id before aggregation")
    label_counts = table.groupby("recording_id")["binary_label"].nunique()
    if (label_counts > 1).any():
        raise ValueError("A recording_id has conflicting class labels")

    aggregated = table.groupby("recording_id", sort=True).agg(
        **{feature: (feature, "mean") for feature in features},
        binary_label=("binary_label", "first"),
        source_dataset=("source_dataset", lambda values: ";".join(sorted(set(values.astype(str))))),
        original_class=("original_class", lambda values: ";".join(sorted(set(values.astype(str))))),
        external_test=("external_test", "first"),
        split=("split", "first"),
        snr_db=("snr_db", "first"),
        sample_ids=("sample_id", lambda values: json.dumps(sorted(set(values.astype(str))))),
        n_segments=("sample_id", "nunique"),
    ).reset_index()
    aggregated["binary_label"] = aggregated["binary_label"].astype(int)
    if not np.isfinite(aggregated[features].to_numpy(dtype=float)).all():
        raise ValueError("Aggregated standardized features contain non-finite values")
    return aggregated


def _load_selected_features() -> dict[str, Any]:
    path = RESULTS_DIR / "selected_features.json"
    if not path.is_file():
        raise FileNotFoundError(f"Stage 3 feature mapping missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_scaled_internal(feature_count: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_tables = {}
    for split in ("train", "validation", "test"):
        path = FEATURES_DIR / f"{split}_features_{feature_count}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Stage 3 scaled table missing: {path}")
        table = pd.read_csv(path)
        if table["external_test"].fillna(False).astype(bool).any():
            raise ValueError(f"NASA external rows found in {split}")
        split_tables[split] = table
    return split_tables["train"], split_tables["validation"], split_tables["test"]


def _validate_scaler(feature_count: int, features: list[str], train: pd.DataFrame) -> str:
    scaler_path = MODELS_DIR / f"scaler_{feature_count}.pkl"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Stage 3 scaler not found: {scaler_path}")
    scaler = joblib.load(scaler_path)
    raw_train_path = FEATURES_DIR / "train_features.csv"
    raw_train = pd.read_csv(raw_train_path)
    raw_train = raw_train[
        raw_train["snr_db"].astype(str).eq("clean")
        & raw_train["recording_id"].astype(str).isin(train["recording_id"].astype(str))
    ]
    raw_aggregated = _aggregate_recordings(raw_train, features)
    transformed_aggregated = train.groupby("recording_id", sort=True)[features].mean()
    transformed_from_scaler = scaler.transform(
        raw_aggregated[features].to_numpy(dtype=float)
    )
    ordered_ids = raw_aggregated["recording_id"].astype(str).tolist()
    transformed_existing = transformed_aggregated.reindex(ordered_ids).to_numpy(dtype=float)
    if not np.allclose(transformed_from_scaler, transformed_existing, rtol=1e-7, atol=1e-7):
        raise ValueError(
            f"Scaler {scaler_path.name} does not reproduce the saved train feature matrix"
        )
    return hashlib.sha256(scaler_path.read_bytes()).hexdigest()


def _common_test_ids(test_table: pd.DataFrame) -> list[str]:
    ids_by_snr: list[set[str]] = []
    for snr in SNR_CONDITIONS:
        selected = test_table[test_table["snr_db"].fillna("").astype(str).eq(snr)]
        ids_by_snr.append(set(selected["recording_id"].astype(str)))
    common = set.intersection(*ids_by_snr)
    if not common:
        raise ValueError("No fixed held-out recordings are present at every SNR condition")
    labels = (
        test_table[test_table["recording_id"].astype(str).isin(common)]
        .groupby("recording_id")["binary_label"]
        .first()
    )
    if labels.nunique() != 2:
        raise ValueError("Common held-out SNR test subset lacks one class")
    return sorted(common)


def _aggregate_for_snr(
    table: pd.DataFrame,
    features: list[str],
    snr: str,
    recording_ids: list[str] | None = None,
) -> pd.DataFrame:
    selected = table[table["snr_db"].fillna("").astype(str).eq(snr)].copy()
    if recording_ids is not None:
        selected = selected[selected["recording_id"].astype(str).isin(recording_ids)]
    return _aggregate_recordings(selected, features)


def _filter_to_subset(train_table: pd.DataFrame, subset_ids: list[str], snr: str) -> pd.DataFrame:
    selected = train_table[
        train_table["recording_id"].astype(str).isin(subset_ids)
        & train_table["snr_db"].fillna("").astype(str).eq(snr)
    ]
    return selected


def _cache_name(
    kernel_name: str,
    feature_count: int,
    training_size: int,
    train_snr: str,
    split_name: str,
    seed: int,
    suffix: str = "",
) -> Path:
    suffix_part = f"_{suffix}" if suffix else ""
    name = (
        f"{kernel_name}_q{feature_count}_n{training_size}_snr_{train_snr}_"
        f"zz_reps1_linear_seed{seed}_{split_name}{suffix_part}.npy"
    )
    return KERNEL_CACHE_DIR / name


class CachedFidelityQuantumKernel(FidelityQuantumKernel):
    """Fidelity kernel that persists known Stage 5 matrices and fingerprints."""

    def __init__(self, *args: Any, cache_entries: dict[tuple[str, str | None], Path], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.cache_entries = cache_entries
        self.cache_hits: dict[str, bool] = {}
        self.compute_seconds = 0.0
        self.computed_pair_count = 0

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x_vec, dtype=np.float64)
        y = None if y_vec is None else np.asarray(y_vec, dtype=np.float64)
        key = (_matrix_hash(x), None if y is None else _matrix_hash(y))
        path = self.cache_entries.get(key)
        if path is not None and path.is_file():
            meta_path = path.with_suffix(".json")
            if meta_path.is_file():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if metadata.get("x_hash") == key[0] and metadata.get("y_hash") == key[1]:
                    self.cache_hits[path.stem] = True
                    return np.load(path, allow_pickle=False)

        start = time.perf_counter()
        matrix = super().evaluate(x, y)
        self.compute_seconds += time.perf_counter() - start
        self.cache_hits[path.stem if path is not None else "unregistered"] = False
        if y is None:
            self.computed_pair_count += x.shape[0] * (x.shape[0] - 1) // 2
        else:
            self.computed_pair_count += x.shape[0] * y.shape[0]
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix, allow_pickle=False)
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "x_hash": key[0],
                        "y_hash": key[1],
                        "shape": list(matrix.shape),
                        "kernel": type(self).__name__,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return matrix


class CachedTrainableFidelityQuantumKernel(TrainableFidelityQuantumKernel):
    """Trainable fidelity kernel with persistent final train/validation/test matrices."""

    def __init__(self, *args: Any, cache_entries: dict[tuple[str, str | None], Path], **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.cache_entries = cache_entries
        self.cache_hits: dict[str, bool] = {}
        self.compute_seconds = 0.0
        self.computed_pair_count = 0

    def evaluate(self, x_vec: np.ndarray, y_vec: np.ndarray | None = None) -> np.ndarray:
        x = np.asarray(x_vec, dtype=np.float64)
        y = None if y_vec is None else np.asarray(y_vec, dtype=np.float64)
        key = (_matrix_hash(x), None if y is None else _matrix_hash(y))
        path = self.cache_entries.get(key)
        if path is not None and path.is_file():
            meta_path = path.with_suffix(".json")
            if meta_path.is_file():
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                if metadata.get("x_hash") == key[0] and metadata.get("y_hash") == key[1]:
                    self.cache_hits[path.stem] = True
                    return np.load(path, allow_pickle=False)

        start = time.perf_counter()
        matrix = super().evaluate(x, y)
        self.compute_seconds += time.perf_counter() - start
        self.cache_hits[path.stem if path is not None else "unregistered"] = False
        if y is None:
            self.computed_pair_count += x.shape[0] * (x.shape[0] - 1) // 2
        else:
            self.computed_pair_count += x.shape[0] * y.shape[0]
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, matrix, allow_pickle=False)
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "x_hash": key[0],
                        "y_hash": key[1],
                        "shape": list(matrix.shape),
                        "kernel": type(self).__name__,
                        "trainable_parameters": self.parameter_values.tolist(),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        return matrix


def _cache_entries(
    kernel_name: str,
    feature_count: int,
    training_size: int,
    train_snr: str,
    seed: int,
    train_x: np.ndarray,
    validation_x: np.ndarray,
    test_by_snr: dict[str, np.ndarray],
    suffix: str = "",
) -> dict[tuple[str, str | None], Path]:
    entries = {
        (_matrix_hash(train_x), None): _cache_name(
            kernel_name, feature_count, training_size, train_snr, "training", seed, suffix
        ),
        (_matrix_hash(validation_x), _matrix_hash(train_x)): _cache_name(
            kernel_name, feature_count, training_size, train_snr, "validation", seed, suffix
        ),
    }
    for snr, test_x in test_by_snr.items():
        entries[(_matrix_hash(test_x), _matrix_hash(train_x))] = _cache_name(
            kernel_name,
            feature_count,
            training_size,
            train_snr,
            f"test_snr_{snr}dB",
            seed,
            suffix,
        )
    return entries


def _estimate_pair_evaluations(
    n_train: int,
    n_validation: int,
    test_sizes: list[int],
    trainable_iterations: int = 0,
) -> int:
    train_matrix = n_train * (n_train - 1) // 2
    validation_matrix = n_validation * n_train
    tests = sum(size * n_train for size in test_sizes)
    # SPSA calibration and updates usually need a small multiple of the training Gram matrix.
    optimizer = trainable_iterations * 4 * train_matrix
    return train_matrix + validation_matrix + tests + optimizer


def _circuit_stats(circuit: QuantumCircuit) -> tuple[int, int]:
    decomposed = circuit.decompose(reps=8)
    two_qubit_gates = sum(
        instruction.operation.num_qubits >= 2
        for instruction in decomposed.data
    )
    return int(decomposed.depth()), int(two_qubit_gates)


def _recording_manifest(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    subset_ids: list[str],
    scaler_sha256: str,
) -> dict[str, Any]:
    def record_map(frame: pd.DataFrame) -> list[dict[str, Any]]:
        return [
            {
                "recording_id": str(row.recording_id),
                "binary_label": int(row.binary_label),
                "sample_ids": json.loads(row.sample_ids),
                "n_segments": int(row.n_segments),
            }
            for row in frame.itertuples(index=False)
        ]

    return {
        "protocol": "Stage 5 ideal simulator, 4 features, clean, 24 balanced recording groups",
        "scaler": "models/scaler_4.pkl",
        "scaler_sha256": scaler_sha256,
        "training_recording_ids": subset_ids,
        "training": record_map(train),
        "validation": record_map(validation),
        "test": record_map(test),
        "train_test_recording_intersection": sorted(
            set(train.recording_id.astype(str)) & set(test.recording_id.astype(str))
        ),
        "train_validation_recording_intersection": sorted(
            set(train.recording_id.astype(str)) & set(validation.recording_id.astype(str))
        ),
        "validation_test_recording_intersection": sorted(
            set(validation.recording_id.astype(str)) & set(test.recording_id.astype(str))
        ),
    }


def _make_trainable_map(qubits: int) -> tuple[QuantumCircuit, ParameterVector]:
    data = ParameterVector("data", qubits)
    theta = ParameterVector("theta", qubits)
    circuit = QuantumCircuit(qubits)
    for index in range(qubits):
        circuit.h(index)
        circuit.p(theta[index] * data[index], index)
    for index in range(qubits - 1):
        circuit.cx(index, index + 1)
        circuit.p(
            theta[index]
            * theta[index + 1]
            * data[index]
            * data[index + 1],
            index + 1,
        )
        circuit.cx(index, index + 1)
    return circuit, theta


def _ensure_result_file(rows: list[dict[str, Any]]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    if RESULTS_PATH.exists():
        prior = pd.read_csv(RESULTS_PATH)
        table = pd.concat([prior, table], ignore_index=True, sort=False)
        key_columns = [
            # experiment is part of the key: the fixed-kernel noise-robustness rows
            # share every other column with the matched comparison row and were
            # overwriting it, which removed Fixed QSVC from the RESEARCH page.
            "experiment",
            "model",
            "feature_count",
            "training_size_requested",
            "training_snr",
            "evaluation_snr",
            "evaluation_split",
        ]
        table = table.drop_duplicates(key_columns, keep="last")
    table.to_csv(RESULTS_PATH, index=False)


def _fit_and_evaluate(
    *,
    model_name: str,
    feature_map: QuantumCircuit,
    trainable_parameters: ParameterVector | None,
    feature_count: int,
    training_size: int,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    tests_by_snr: dict[str, pd.DataFrame],
    seed: int,
    optimizer_iterations: int = 0,
    trainable_restarts: int = 10,
    use_cache: bool = True,
) -> tuple[list[dict[str, Any]], Any, dict[str, Any]]:
    features = [
        name
        for name in json.loads((RESULTS_DIR / "selected_features.json").read_text())[
            "qubit_mappings"
        ][str(feature_count)]["features"]
    ]
    train_x = train[features].to_numpy(dtype=float)
    train_y = train["binary_label"].to_numpy(dtype=int)
    validation_x = validation[features].to_numpy(dtype=float)
    test_arrays = {
        snr: frame[features].to_numpy(dtype=float)
        for snr, frame in tests_by_snr.items()
    }
    train_ids = train["recording_id"].astype(str).tolist()
    validation_ids = validation["recording_id"].astype(str).tolist()
    test_ids_hash = {snr: _ids_hash(frame.recording_id.astype(str).tolist()) for snr, frame in tests_by_snr.items()}
    training_ids_hash = _ids_hash(train_ids)
    feature_map_name = (
        "trainable_product_reps1_linear"
        if trainable_parameters is not None
        else "zz_reps1_linear"
    )

    estimated = _estimate_pair_evaluations(
        len(train), len(validation), [len(frame) for frame in tests_by_snr.values()], optimizer_iterations
    )
    print(
        f"ESTIMATE {model_name}: qubits={feature_count}, train_recordings={len(train)}, "
        f"validation_recordings={len(validation)}, test_recordings={len(next(iter(tests_by_snr.values())))}, "
        f"evaluations<={estimated}, feature_map={feature_map_name}, seed={seed}"
    )
    if estimated > MAX_PAIR_EVALUATIONS_PER_CONFIGURATION:
        raise RuntimeError(
            f"Refusing {model_name} configuration: estimated {estimated:,} fidelity evaluations "
            f"exceeds safety cap {MAX_PAIR_EVALUATIONS_PER_CONFIGURATION:,}."
        )

    parameter_suffix = ""
    trainable_result: dict[str, Any] = {}
    kernel_training_seconds = 0.0
    if trainable_parameters is not None:
        parameter_cache_path = (
            MODEL_DIR
            / f"trainable_params_q{feature_count}_n{training_size}_snr_clean_seed{seed}.json"
        )
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        if parameter_cache_path.is_file():
            trainable_result = json.loads(parameter_cache_path.read_text(encoding="utf-8"))
            fitted_parameters = np.asarray(trainable_result["optimal_parameters"], dtype=float)
            kernel = CachedTrainableFidelityQuantumKernel(
                feature_map=feature_map,
                training_parameters=trainable_parameters,
                cache_entries={},
            )
            kernel.assign_training_parameters(fitted_parameters)
            parameter_suffix = f"theta_{_matrix_hash(fitted_parameters.reshape(1,-1))[:10]}"
        else:
            # SPSA perturbations are random, so a single unseeded run is not
            # reproducible (test F1 ranged 0.43-0.71 across seeds). Run seeded
            # restarts and keep the one with the best validation score; the test
            # split never influences the choice.
            validation_y = validation["binary_label"].to_numpy(dtype=int)
            restart_records: list[dict[str, Any]] = []
            best: tuple[tuple[float, float, float], Any, Any, int] | None = None
            start = time.perf_counter()
            for restart in range(trainable_restarts):
                optimizer_seed = seed * 1000 + restart
                algorithm_globals.random_seed = optimizer_seed
                candidate_kernel = CachedTrainableFidelityQuantumKernel(
                    feature_map=feature_map,
                    training_parameters=trainable_parameters,
                    cache_entries={},
                )
                trainer = QuantumKernelTrainer(
                    quantum_kernel=candidate_kernel,
                    loss=SVCLoss(C=0.1, class_weight="balanced"),
                    optimizer=SPSA(maxiter=optimizer_iterations),
                    initial_point=np.full(len(trainable_parameters), 0.5, dtype=float),
                )
                candidate = trainer.fit(train_x, train_y)
                selector = SVC(kernel="precomputed", C=0.1, class_weight="balanced")
                selector.fit(candidate_kernel.evaluate(train_x), train_y)
                validation_matrix = candidate_kernel.evaluate(validation_x, train_x)
                candidate_metrics = calculate_metrics(
                    validation_y,
                    selector.predict(validation_matrix),
                    selector.decision_function(validation_matrix),
                )
                restart_records.append(
                    {
                        "optimizer_seed": optimizer_seed,
                        "parameters": np.asarray(candidate.optimal_point, dtype=float).tolist(),
                        "optimal_loss": float(candidate.optimal_value),
                        "validation_f1": candidate_metrics["f1"],
                        "validation_balanced_accuracy": candidate_metrics["balanced_accuracy"],
                        "validation_drone_recall": candidate_metrics["drone_recall"],
                    }
                )
                key = validation_key(candidate_metrics)
                if best is None or key > best[0]:
                    best = (key, candidate_kernel, candidate, optimizer_seed)
                print(
                    f"  trainable restart {restart + 1}/{trainable_restarts}: seed={optimizer_seed}, "
                    f"validation F1={candidate_metrics['f1']:.3f}"
                )
            kernel_training_seconds = time.perf_counter() - start
            _, kernel, optimized, selected_seed = best
            fitted_parameters = np.asarray(optimized.optimal_point, dtype=float)
            trainable_result = {
                "optimal_parameters": fitted_parameters.tolist(),
                "optimizer_evaluations": int(optimized.optimizer_evals),
                "optimal_loss": float(optimized.optimal_value),
                "training_recording_ids_sha256": training_ids_hash,
                "feature_map": feature_map_name,
                "iterations_max": optimizer_iterations,
                "selection": "best validation F1, then balanced accuracy, then drone recall",
                "selected_optimizer_seed": selected_seed,
                "restarts": restart_records,
            }
            parameter_cache_path.write_text(
                json.dumps(trainable_result, indent=2) + "\n", encoding="utf-8"
            )
            parameter_suffix = f"theta_{_matrix_hash(fitted_parameters.reshape(1,-1))[:10]}"
    else:
        kernel = CachedFidelityQuantumKernel(
            feature_map=feature_map,
            enforce_psd=True,
            evaluate_duplicates="off_diagonal",
            cache_entries={},
        )

    cache_name = "trainable" if trainable_parameters is not None else "fixed"
    if use_cache:
        kernel.cache_entries = _cache_entries(
            cache_name,
            feature_count,
            training_size,
            "clean",
            seed,
            train_x,
            validation_x,
            test_arrays,
            parameter_suffix,
        )
    else:
        kernel.cache_entries = {}

    matrix_start = time.perf_counter()
    train_kernel = kernel.evaluate(train_x)
    validation_kernel = kernel.evaluate(validation_x, train_x)
    test_kernels = {
        snr: kernel.evaluate(values, train_x) for snr, values in test_arrays.items()
    }
    matrix_wall_seconds = time.perf_counter() - matrix_start

    model_path = MODEL_DIR / f"{model_name}_q{feature_count}_n{training_size}_clean_seed{seed}.joblib"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    if model_name == "fixed_qsvc" or model_name == "trainable_qsvc":
        classifier = QSVC(quantum_kernel=kernel, C=0.1, class_weight="balanced")
        classifier.fit(train_x, train_y)
    elif model_name == "rbf_svm":
        classifier = SVC(
            kernel="rbf", C=0.1, gamma="scale", class_weight="balanced"
        )
        classifier.fit(train_x, train_y)
    elif model_name == "small_mlp":
        from classical_training import fit_model

        classifier = fit_model(
            "mlp", train, features, seed, validation=validation
        )
    else:
        raise ValueError(f"Unsupported model type: {model_name}")
    fit_seconds = time.perf_counter() - start
    joblib.dump(classifier, model_path)

    depth, two_qubit_count = _circuit_stats(feature_map)
    rows: list[dict[str, Any]] = []
    validation_start = time.perf_counter()
    validation_metrics, _ = score_frame(classifier, validation, features)
    validation_seconds = time.perf_counter() - validation_start
    for evaluation_snr, test_frame in tests_by_snr.items():
        test_start = time.perf_counter()
        test_metrics, _ = score_frame(classifier, test_frame, features)
        prediction_seconds = time.perf_counter() - test_start
        rows.append(
            {
                "model": model_name,
                "experiment": "matched_sample_clean_model_robustness",
                "simulator": "ideal_statevector_fidelity",
                "feature_count": feature_count,
                "qubit_count": feature_count,
                "training_size_requested": training_size,
                "training_recordings": len(train),
                "training_snr": "clean",
                "evaluation_snr": evaluation_snr,
                "evaluation_split": "test",
                "training_unique_sample_ids": sum(len(json.loads(ids)) for ids in train.sample_ids),
                "training_sample_manifest_hash": training_ids_hash,
                "validation_sample_manifest_hash": _ids_hash(validation_ids),
                "test_sample_manifest_hash": test_ids_hash[evaluation_snr],
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_precision": validation_metrics["precision"],
                "validation_recall": validation_metrics["recall"],
                "validation_f1": validation_metrics["f1"],
                "validation_roc_auc": validation_metrics["roc_auc"],
                "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
                "validation_drone_recall": validation_metrics["drone_recall"],
                "validation_non_drone_recall": validation_metrics["non_drone_recall"],
                **test_metrics,
                "feature_map": feature_map_name,
                "feature_map_reps": 1,
                "circuit_depth": depth,
                "two_qubit_gate_count": two_qubit_count,
                "estimated_quantum_evaluations": estimated,
                "kernel_evaluations_computed_this_run": int(
                    getattr(kernel, "computed_pair_count", 0)
                ),
                "kernel_construction_seconds": float(
                    getattr(kernel, "compute_seconds", matrix_wall_seconds)
                ),
                "kernel_matrix_wall_seconds": matrix_wall_seconds,
                "training_seconds": fit_seconds,
                "prediction_seconds": prediction_seconds,
                "validation_prediction_seconds": validation_seconds,
                "train_kernel_cache": str(
                    kernel.cache_entries.get((_matrix_hash(train_x), None), "")
                ),
                "validation_kernel_cache": str(
                    kernel.cache_entries.get(
                        (_matrix_hash(validation_x), _matrix_hash(train_x)), ""
                    )
                ),
                "test_kernel_cache": str(
                    kernel.cache_entries.get(
                        (_matrix_hash(test_arrays[evaluation_snr]), _matrix_hash(train_x)), ""
                    )
                ),
                "kernel_cache_hits": json.dumps(getattr(kernel, "cache_hits", {}), sort_keys=True),
                "model_path": str(model_path),
                "optimizer_evaluations": trainable_result.get("optimizer_evaluations", 0),
                "optimizer_loss": trainable_result.get("optimal_loss", float("nan")),
                "trainable_parameters": json.dumps(trainable_result.get("optimal_parameters", [])),
                "trainable_kernel_training_seconds": kernel_training_seconds,
            }
        )
    return rows, classifier, {
        "validation_metrics": validation_metrics,
        "kernel": kernel,
        "train": train,
        "validation": validation,
        "tests_by_snr": tests_by_snr,
        "features": features,
        "model_path": model_path,
        "trainable_result": trainable_result,
        "training_ids": train_ids,
        "validation_ids": validation_ids,
    }


def _load_stage_assets(feature_count: int, train_size: int, seed: int) -> dict[str, Any]:
    selected = _load_selected_features()
    features = list(selected["qubit_mappings"][str(feature_count)]["features"])
    scaler_path = MODELS_DIR / f"scaler_{feature_count}.pkl"
    if not scaler_path.is_file():
        raise FileNotFoundError(f"Training-fitted scaler missing: {scaler_path}")
    scaler_sha256 = _validate_scaler(feature_count, features)
    train_table = pd.read_csv(FEATURES_DIR / f"train_features_{feature_count}.csv")
    validation_table = pd.read_csv(FEATURES_DIR / f"validation_features_{feature_count}.csv")
    test_table = pd.read_csv(FEATURES_DIR / f"test_features_{feature_count}.csv")
    for name, table in (("train", train_table), ("validation", validation_table), ("test", test_table)):
        if table["external_test"].fillna(False).astype(bool).any():
            raise ValueError(f"NASA rows found in internal {name} feature matrix")

    subset_ids, manifest_count = load_subset_ids(train_size)
    if manifest_count != train_size:
        raise ValueError(f"Requested {train_size}, manifest contains {manifest_count}")
    clean_train_segments = train_table[
        train_table["recording_id"].astype(str).isin(subset_ids)
        & train_table["snr_db"].astype(str).eq("clean")
    ]
    train = _aggregate_recordings(clean_train_segments, features)
    if train["recording_id"].nunique() != train_size:
        raise ValueError("Clean Stage 5 train rows do not include every selected recording")
    class_counts = train.binary_label.value_counts().to_dict()
    if class_counts.get(0) != class_counts.get(1):
        raise ValueError(f"Stage 5 training subset is not balanced: {class_counts}")

    validation = _aggregate_for_snr(validation_table, features, "clean")
    if set(train.recording_id.astype(str)) & set(validation.recording_id.astype(str)):
        raise AssertionError("Stage 5 train and validation recording IDs overlap")

    common_test_ids = _common_test_ids(test_table)
    if set(train.recording_id.astype(str)) & set(common_test_ids):
        raise AssertionError("Stage 5 train and test recording IDs overlap")
    if set(validation.recording_id.astype(str)) & set(common_test_ids):
        raise AssertionError("Stage 5 validation and test recording IDs overlap")
    tests_by_snr = {
        snr: _aggregate_for_snr(test_table, features, snr, common_test_ids)
        for snr in SNR_CONDITIONS
    }
    for snr, frame in tests_by_snr.items():
        if frame.recording_id.astype(str).tolist() != common_test_ids:
            raise AssertionError(f"Test recording set differs at SNR={snr}")

    comparison_manifest = {
        "stage": "5 ideal simulator",
        "seed": seed,
        "feature_count": feature_count,
        "qubit_count": feature_count,
        "features": features,
        "training_size": train_size,
        "condition": "clean",
        "scaler_path": str(scaler_path),
        "scaler_sha256": scaler_sha256,
        "scaler_reproduces_training_rows": True,
        "train_class_counts": class_counts,
        "training_recordings": _record_map(train),
        "validation_recordings": _record_map(validation),
        "test_recordings": {
            snr: _record_map(frame) for snr, frame in tests_by_snr.items()
        },
        "train_validation_overlap": [],
        "train_test_overlap": [],
        "validation_test_overlap": [],
        "same_aggregated_samples_for_classical_and_quantum": True,
        "nasa_loaded_or_used": False,
    }
    if feature_count == 4 and train_size == 24:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(json.dumps(comparison_manifest, indent=2) + "\n", encoding="utf-8")
    return {
        "features": features,
        "train": train,
        "validation": validation,
        "tests_by_snr": tests_by_snr,
        "subset_ids": subset_ids,
        "scaler_sha256": scaler_sha256,
        "training_size": train_size,
    }


def _record_map(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "recording_id": str(row.recording_id),
            "binary_label": int(row.binary_label),
            "sample_ids": json.loads(row.sample_ids),
            "segment_count": int(row.n_segments),
        }
        for row in frame.itertuples(index=False)
    ]


def _validate_scaler(feature_count: int, features: list[str]) -> str:
    scaler_path = MODELS_DIR / f"scaler_{feature_count}.pkl"
    scaler = joblib.load(scaler_path)
    raw = pd.read_csv(FEATURES_DIR / "train_features.csv")
    raw = raw[raw.snr_db.astype(str).eq("clean")]
    raw_aggregated = _aggregate_recordings(raw, features)
    scaled = pd.read_csv(FEATURES_DIR / f"train_features_{feature_count}.csv")
    scaled = scaled[scaled.snr_db.astype(str).eq("clean")]
    scaled_aggregated = _aggregate_recordings(scaled, features)
    scaled_aggregated = scaled_aggregated.set_index("recording_id").reindex(
        raw_aggregated.recording_id.astype(str)
    )
    transformed = scaler.transform(raw_aggregated[features].to_numpy(dtype=float))
    if not np.allclose(
        transformed,
        scaled_aggregated[features].to_numpy(dtype=float),
        rtol=1e-7,
        atol=1e-7,
    ):
        raise ValueError(f"Saved scaler {scaler_path.name} does not reproduce Stage 3 values")
    return hashlib.sha256(scaler_path.read_bytes()).hexdigest()


def _fit_classical_comparators(data: dict[str, Any], seed: int) -> list[dict[str, Any]]:
    features = data["features"]
    train = data["train"]
    validation = data["validation"]
    tests_by_snr = data["tests_by_snr"]
    from classical_training import fit_model

    svm = SVC(kernel="rbf", C=0.1, gamma="scale", class_weight="balanced")
    start = time.perf_counter()
    svm.fit(train[features].to_numpy(dtype=float), train.binary_label.to_numpy(dtype=int))
    svm_training_seconds = time.perf_counter() - start
    mlp_start = time.perf_counter()
    mlp = fit_model("mlp", train, features, seed, validation=validation)
    mlp_training_seconds = time.perf_counter() - mlp_start
    output = []
    models = {"rbf_svm": (svm, svm_training_seconds), "small_mlp": (mlp, mlp_training_seconds)}
    for name, (model, fit_seconds) in models.items():
        model_path = (
            MODEL_DIR
            / f"{name}_q{len(features)}_n{len(train)}_clean_aggregated_seed{seed}.joblib"
        )
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        val_metrics, _ = score_frame(model, validation, features)
        for snr, test_frame in tests_by_snr.items():
            start = time.perf_counter()
            test_metrics, _ = score_frame(model, test_frame, features)
            output.append(
                {
                    "model": name,
                    "experiment": "matched_sample_clean_model_robustness",
                    "simulator": "classical_cpu",
                    # Label with the real configuration. Hard-coding 4 / 24 here let
                    # later grid configurations overwrite the matched 4/24 rows.
                    "feature_count": len(features),
                    "qubit_count": 0,
                    "training_size_requested": data["training_size"],
                    "training_recordings": len(train),
                    "training_snr": "clean",
                    "evaluation_snr": snr,
                    "evaluation_split": "test",
                    "training_sample_manifest_hash": _ids_hash(train.recording_id.astype(str).tolist()),
                    "validation_sample_manifest_hash": _ids_hash(validation.recording_id.astype(str).tolist()),
                    "test_sample_manifest_hash": _ids_hash(test_frame.recording_id.astype(str).tolist()),
                    "validation_accuracy": val_metrics["accuracy"],
                    "validation_precision": val_metrics["precision"],
                    "validation_recall": val_metrics["recall"],
                    "validation_f1": val_metrics["f1"],
                    "validation_roc_auc": val_metrics["roc_auc"],
                    "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "validation_drone_recall": val_metrics["drone_recall"],
                    "validation_non_drone_recall": val_metrics["non_drone_recall"],
                    **test_metrics,
                    "feature_map": "none",
                    "feature_map_reps": 0,
                    "circuit_depth": 0,
                    "two_qubit_gate_count": 0,
                    "estimated_quantum_evaluations": 0,
                    "kernel_evaluations_computed_this_run": 0,
                    "kernel_construction_seconds": 0.0,
                    "kernel_matrix_wall_seconds": 0.0,
                    "training_seconds": fit_seconds,
                    "prediction_seconds": time.perf_counter() - start,
                    "validation_prediction_seconds": 0.0,
                    "kernel_cache_hits": "{}",
                    "model_path": str(model_path),
                    "optimizer_evaluations": 0,
                    "optimizer_loss": float("nan"),
                    "trainable_parameters": "[]",
                    "trainable_kernel_training_seconds": 0.0,
                }
            )
    return output


def _fixed_feature_map(qubits: int) -> QuantumCircuit:
    return zz_feature_map(
        feature_dimension=qubits,
        reps=1,
        entanglement="linear",
        parameter_prefix="data",
    )


def _append_results(rows: list[dict[str, Any]]) -> None:
    _ensure_result_file(rows)


def _run_configuration(
    feature_count: int,
    training_size: int,
    seed: int,
    tests_snr: tuple[str, ...],
    trainable_iterations: int = 0,
) -> list[dict[str, Any]]:
    data = _load_stage_assets(feature_count, training_size, seed)
    tests_by_snr = {snr: data["tests_by_snr"][snr] for snr in tests_snr}
    rows: list[dict[str, Any]] = []
    feature_map = _fixed_feature_map(feature_count)
    fixed_rows, _, _ = _fit_and_evaluate(
        model_name="fixed_qsvc",
        feature_map=feature_map,
        trainable_parameters=None,
        feature_count=feature_count,
        training_size=training_size,
        train=data["train"],
        validation=data["validation"],
        tests_by_snr=tests_by_snr,
        seed=seed,
    )
    rows.extend(fixed_rows)
    return rows


def run_stage5(
    seed: int = RANDOM_SEED,
    initial_only: bool = False,
    trainable_iterations: int = 3,
    trainable_restarts: int = 10,
) -> dict[str, Any]:
    if trainable_iterations < 1 or trainable_iterations > 5:
        raise ValueError("Trainable optimizer iterations must be between 1 and 5")
    if trainable_restarts < 1:
        raise ValueError("At least one trainable restart is required")
    if RESULTS_PATH.exists():
        previous = pd.read_csv(RESULTS_PATH)
        if not previous.empty:
            # Stage 5 reruns replace only matching config rows via _ensure_result_file.
            pass

    initial = _load_stage_assets(4, 24, seed)
    train = initial["train"]
    validation = initial["validation"]
    tests_by_snr = initial["tests_by_snr"]
    features = initial["features"]
    if len(train) != 24 or train.binary_label.value_counts().to_dict() != {0: 12, 1: 12}:
        raise ValueError("Stage 5 initial training data must be exactly 24 balanced recordings")
    if len(set(train.recording_id.astype(str)) & set(tests_by_snr["clean"].recording_id.astype(str))):
        raise AssertionError("Stage 5 training and test recording IDs overlap")

    comparison_manifest = _recording_manifest(
        train,
        validation,
        tests_by_snr["clean"],
        initial["subset_ids"],
        initial["scaler_sha256"],
    )
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(comparison_manifest, indent=2) + "\n", encoding="utf-8")

    estimates = _estimate_pair_evaluations(
        len(train),
        len(validation),
        [len(frame) for frame in tests_by_snr.values()],
        trainable_iterations,
    )
    print(
        "STAGE 5 INITIAL ESTIMATE: "
        f"4 qubits, 24 balanced recordings, clean, validation={len(validation)}, "
        f"test={len(tests_by_snr['clean'])}, estimated fidelity evaluations<={estimates}"
    )
    if estimates > MAX_PAIR_EVALUATIONS_PER_CONFIGURATION:
        raise RuntimeError("Initial Stage 5 experiment exceeds the configured simulator evaluation cap")

    rows: list[dict[str, Any]] = _fit_classical_comparators(initial, seed)
    fixed_map = _fixed_feature_map(4)
    fixed_rows, fixed_model, fixed_info = _fit_and_evaluate(
        model_name="fixed_qsvc",
        feature_map=fixed_map,
        trainable_parameters=None,
        feature_count=4,
        training_size=24,
        train=train,
        validation=validation,
        tests_by_snr={"clean": tests_by_snr["clean"]},
        seed=seed,
    )
    rows.extend(fixed_rows)
    _append_results(rows)

    # Trainable map is also simulator-only; the data parameter prefix intentionally
    # differs from Qiskit's internal fidelity reparameterization prefix.
    trainable_map, theta = _make_trainable_map(4)
    trainable_rows, _, trainable_info = _fit_and_evaluate(
        model_name="trainable_qsvc",
        feature_map=trainable_map,
        trainable_parameters=theta,
        feature_count=4,
        training_size=24,
        train=train,
        validation=validation,
        tests_by_snr={"clean": tests_by_snr["clean"]},
        seed=seed,
        optimizer_iterations=trainable_iterations,
        trainable_restarts=trainable_restarts,
    )
    rows.extend(trainable_rows)
    _append_results(rows)

    initial_table = pd.DataFrame(rows)
    initial_metrics = initial_table[
        initial_table["evaluation_split"].eq("test")
        & initial_table["evaluation_snr"].astype(str).eq("clean")
    ]
    print("STAGE 5 INITIAL TEST COMPARISON")
    print(
        initial_metrics[
            ["model", "accuracy", "f1", "balanced_accuracy", "drone_recall", "non_drone_recall"]
        ].to_string(index=False)
    )

    if initial_only:
        return {"initial_rows": len(initial_table), "expanded": False}

    expanded_rows: list[dict[str, Any]] = []
    # Initial fixed 4-qubit model was already run; only expand its held-out noise tests.
    clean_tests = {
        snr: tests_by_snr[snr]
        for snr in SNR_CONDITIONS
    }
    if len(clean_tests) > 1:
        _, _, clean_info = _fit_and_evaluate(
            model_name="fixed_qsvc",
            feature_map=fixed_map,
            trainable_parameters=None,
            feature_count=4,
            training_size=24,
            train=train,
            validation=validation,
            tests_by_snr=clean_tests,
            seed=seed,
        )
        for snr in SNR_CONDITIONS:
            metric_row = {
                "model": "fixed_qsvc",
                "experiment": "clean_model_noise_robustness",
                "feature_count": 4,
                "qubit_count": 4,
                "training_size_requested": 24,
                "training_recordings": len(train),
                "training_snr": "clean",
                "evaluation_snr": snr,
                "evaluation_split": "test",
                "training_sample_manifest_hash": _ids_hash(train.recording_id.astype(str).tolist()),
                "validation_sample_manifest_hash": _ids_hash(validation.recording_id.astype(str).tolist()),
                "test_sample_manifest_hash": _ids_hash(clean_tests[snr].recording_id.astype(str).tolist()),
                "feature_map": "zz_reps1_linear",
                "feature_map_reps": 1,
                "circuit_depth": _circuit_stats(fixed_map)[0],
                "two_qubit_gate_count": _circuit_stats(fixed_map)[1],
                "estimated_quantum_evaluations": _estimate_pair_evaluations(len(train), len(validation), [len(frame) for frame in clean_tests.values()]),
                "kernel_evaluations_computed_this_run": int(getattr(clean_info["kernel"], "computed_pair_count", 0)),
                "kernel_construction_seconds": float(getattr(clean_info["kernel"], "compute_seconds", 0.0)),
                "training_seconds": float("nan"),
                "prediction_seconds": float("nan"),
                "model_path": str(fixed_info["model_path"]),
                "experiment": "clean_model_noise_robustness",
            }
            test_metrics, _ = score_frame(fixed_model, clean_tests[snr], features)
            metric_row.update(test_metrics)
            expanded_rows.append(metric_row)

    # Expand the ideal fixed kernel over 4/5/6 features and 24/50/70 training
    # recordings, with clean-trained models tested on the common fixed SNR cohort.
    for feature_count in FEATURE_COUNTS:
        for training_size in TRAIN_SIZES:
            if feature_count == 4 and training_size == 24:
                continue
            data = _load_stage_assets(feature_count, training_size, seed)
            expanded_rows.extend(
                _run_configuration(
                    feature_count,
                    training_size,
                    seed,
                    SNR_CONDITIONS,
                )
            )
            expanded_rows.extend(_fit_classical_comparators(data, seed))
    _append_results(expanded_rows)
    return {"initial_rows": len(initial_table), "expanded_rows": len(expanded_rows), "expanded": True}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--initial-only",
        action="store_true",
        help="Run only the first 4-qubit/24-recording comparison and skip the expanded fixed-kernel grid.",
    )
    parser.add_argument(
        "--trainable-iterations",
        type=int,
        default=3,
        help="SPSA max iterations for the initial trainable kernel (1-5).",
    )
    parser.add_argument(
        "--trainable-restarts",
        type=int,
        default=10,
        help="Seeded SPSA restarts; the restart with the best validation F1 is kept.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = run_stage5(
        args.seed, args.initial_only, args.trainable_iterations, args.trainable_restarts
    )
    print(f"Stage 5 results saved to {RESULTS_PATH}")
    print(f"Kernel matrices cached under {KERNEL_CACHE_DIR}")
    print(result)


if __name__ == "__main__":
    main()