"""Stage 7 IBM Quantum preflight and explicitly approved QPU execution.

The default invocation performs a backend preflight and local Aer preview only.
It never submits a job. Submission additionally requires RUN_REAL_QPU=YES,
--confirm-real-qpu, and the separate --submit-after-review approval flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from sklearn.svm import SVC

from config import MODELS_DIR, PROJECT_ROOT, RESULTS_DIR
from classical_utils import calculate_metrics
from quantum_kernel_simulator import _circuit_stats, _ids_hash, _make_trainable_map
from quantum_noise_experiments import _read_frozen_inputs


FROZEN_CONFIG_PATH = MODELS_DIR / "quantum" / "stage7_frozen_config.json"
RESULTS_QUANTUM_DIR = RESULTS_DIR / "quantum"
RESULTS_PLOT_DIR = RESULTS_DIR / "plots"
BACKEND_PROPERTIES_PATH = RESULTS_QUANTUM_DIR / "qpu_backend_properties.json"
BACKEND_PREVIEW_PATH = RESULTS_QUANTUM_DIR / "backend_noise_preview.csv"
QPU_RESULTS_PATH = RESULTS_QUANTUM_DIR / "qpu_results.csv"
JOBS_PATH = RESULTS_QUANTUM_DIR / "qpu_jobs.json"
COMPARISON_PATH = RESULTS_QUANTUM_DIR / "final_quantum_comparison.csv"
PLOT_PATH = RESULTS_PLOT_DIR / "simulator_vs_noisy_vs_real_qpu.png"
JOB_CACHE_DIR = RESULTS_QUANTUM_DIR / "qpu_job_cache"
SHOTS = 1024
PREFERRED_BACKEND = "ibm_pittsburgh"
PREFERRED_LAYOUT = [87, 97, 107, 108]
CIRCUITS_PER_JOB = 1137
MAX_EXECUTIONS_PER_JOB = 10_000_000
MAX_CALIBRATION_DEGRADATION = 1.25
APPROVED_CALIBRATION_REFERENCE = {
    "selected_median_1q_gate_error": 0.0001589,
    "selected_median_2q_gate_error": 0.0013857,
    "selected_median_readout_error": 0.0035400,
}
EXCLUDED_BACKENDS = {"ibm_miami"}
# Seconds between shots; None keeps the backend default.
REP_DELAY_SECONDS: float | None = None
INCLUDE_VALIDATION_KERNEL = True

# Per-backend settings selected with --backend. ibm_miami (Nighthawk) allows a
# 1-4 ms repetition delay (default 4 ms vs 250 us on Heron), so the full workload
# does not fit a 25-minute allocation; run it at the 1 ms minimum, with fewer shots
# and without the validation kernel, which the hardware score never uses.
BACKEND_PROFILES: dict[str, dict[str, Any]] = {
    "ibm_pittsburgh": {
        "layout": [87, 97, 107, 108],
        "calibration_reference": dict(APPROVED_CALIBRATION_REFERENCE),
        "rep_delay_seconds": None,
    },
    "ibm_kingston": {
        # Heron r2. Lowest-error linear chain on 2026-10-07 (CZ 0.16-0.26%,
        # readout 0.3-0.6%). Reference = selected-qubit medians from the 18:13 CDT
        # calibration; the 16:13 one (1Q 0.0001509) changed after batch 1, so the
        # run restarted in a single calibration window (first batch 1 archived).
        "layout": [42, 43, 44, 45],
        "calibration_reference": {
            "selected_median_1q_gate_error": 0.0003149,
            "selected_median_2q_gate_error": 0.0016605,
            "selected_median_readout_error": 0.0043945,
        },
        "rep_delay_seconds": None,
    },
    "ibm_miami": {
        # Lowest-error linear chain on 2026-10-07 (CZ 0.18-0.25%, readout 0.6-1.3%).
        "layout": [44, 54, 64, 74],
        # Selected-qubit medians from the approved 2026-10-07 calibration snapshot.
        "calibration_reference": {
            "selected_median_1q_gate_error": 0.0001746,
            "selected_median_2q_gate_error": 0.0020672,
            "selected_median_readout_error": 0.0096436,
        },
        "rep_delay_seconds": 0.001,
    },
}


def _apply_backend_profile(name: str, include_validation_kernel: bool) -> None:
    global PREFERRED_BACKEND, PREFERRED_LAYOUT, APPROVED_CALIBRATION_REFERENCE
    global REP_DELAY_SECONDS, INCLUDE_VALIDATION_KERNEL
    if name not in BACKEND_PROFILES:
        raise ValueError(f"No backend profile for {name}; known: {sorted(BACKEND_PROFILES)}")
    profile = BACKEND_PROFILES[name]
    if profile["calibration_reference"] is None:
        raise ValueError(f"{name} has no approved calibration reference yet")
    PREFERRED_BACKEND = name
    PREFERRED_LAYOUT = list(profile["layout"])
    APPROVED_CALIBRATION_REFERENCE = dict(profile["calibration_reference"])
    REP_DELAY_SECONDS = profile["rep_delay_seconds"]
    INCLUDE_VALIDATION_KERNEL = include_validation_kernel


def _expected_circuit_count(sizes: dict[str, int]) -> int:
    training_pairs = sizes["training"] * (sizes["training"] - 1) // 2
    validation_pairs = sizes["validation"] * sizes["training"] if INCLUDE_VALIDATION_KERNEL else 0
    return training_pairs + validation_pairs + sizes["test"] * sizes["training"]


class QPUJobFailure(RuntimeError):
    """A submitted job failed; the message contains its ID and circuit range."""


def _load_local_env(path: Path) -> None:
    """Load shell-style secrets without displaying their values."""
    if not path.is_file():
        raise FileNotFoundError(f"Local IBM credentials file not found: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :]
        name, separator, raw_value = stripped.partition("=")
        if not separator or not name.strip().isidentifier():
            raise ValueError("Malformed assignment in local IBM credentials file")
        try:
            parsed = shlex.split(raw_value, comments=True, posix=True)
        except ValueError as exc:
            raise ValueError("Malformed quoting in local IBM credentials file") from exc
        value = parsed[0] if parsed else ""
        os.environ.setdefault(name.strip(), value)


def _load_frozen_data() -> tuple[dict[str, Any], dict[str, Any]]:
    if not FROZEN_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Stage 7 frozen config not found: {FROZEN_CONFIG_PATH}")
    config = json.loads(FROZEN_CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("status") != "prepared_not_submitted":
        raise ValueError("Stage 7 frozen config is not in prepared_not_submitted state")
    if config.get("hardware_submission_allowed") is not False:
        raise ValueError("Frozen config unexpectedly enables hardware submission")
    if config.get("qubit_count") != 4 or len(config.get("feature_names", [])) != 4:
        raise ValueError("Stage 7 is frozen to exactly four features and four qubits")
    if len(config.get("training_recording_ids", [])) != 24:
        raise ValueError("Stage 7 config must contain exactly 24 training recordings")
    if len(config.get("trained_kernel_parameters", [])) != 4:
        raise ValueError("Stage 7 config must contain four frozen trainable parameters")
    if config.get("test_sample_ids") is None or config.get("test_recording_ids") is None:
        raise ValueError("Stage 7 config is missing the frozen test cohort")

    data = _read_frozen_inputs(seed=42)
    if data["features"] != config["feature_names"]:
        raise ValueError("Current Stage 5 feature mapping differs from the frozen Stage 7 config")
    if not np.array_equal(data["theta"], np.asarray(config["trained_kernel_parameters"], dtype=float)):
        raise ValueError("Stage 5 trainable parameters differ from the frozen Stage 7 config")
    scaler_path = Path(config["scaler_path"])
    scaler_digest = hashlib.sha256(scaler_path.read_bytes()).hexdigest()
    if scaler_digest != config["scaler_sha256"] or data["scaler_sha256"] != config["scaler_sha256"]:
        raise ValueError("Stage 3 scaler hash does not match the frozen Stage 7 config")
    train_ids = data["train"].recording_id.astype(str).tolist()
    test_ids = data["tests_by_snr"]["clean"].recording_id.astype(str).tolist()
    if train_ids != config["training_recording_ids"]:
        raise ValueError("Training recording IDs differ from the frozen Stage 7 config")
    if test_ids != config["test_recording_ids"]:
        raise ValueError("Test recording IDs differ from the frozen Stage 7 config")
    current_training_samples = sorted(
        sample_id
        for sample_ids in data["train"]["sample_ids"]
        for sample_id in json.loads(sample_ids)
    )
    current_test_samples = sorted(
        sample_id
        for sample_ids in data["tests_by_snr"]["clean"]["sample_ids"]
        for sample_id in json.loads(sample_ids)
    )
    if current_training_samples != sorted(config["training_sample_ids"]):
        raise ValueError("Training sample IDs differ from the frozen Stage 7 config")
    if current_test_samples != sorted(config["test_sample_ids"]):
        raise ValueError("Test sample IDs differ from the frozen Stage 7 config")
    if len(set(train_ids) & set(test_ids)):
        raise ValueError("Frozen Stage 7 training and test recording IDs overlap")
    if data["train"].binary_label.value_counts().to_dict() != {0: 12, 1: 12}:
        raise ValueError("Frozen Stage 7 training recordings are not balanced 12/12")
    if data["train"].external_test.fillna(False).astype(bool).any() or data["tests_by_snr"]["clean"].external_test.fillna(False).astype(bool).any():
        raise ValueError("External NASA records are not permitted in the Stage 7 experiment")
    return config, data


def _frozen_feature_map(data: dict[str, Any]) -> QuantumCircuit:
    circuit, theta_parameters = _make_trainable_map(4)
    return circuit.assign_parameters(
        dict(zip(theta_parameters, data["theta"].tolist())), inplace=False
    )


def _compute_uncompute_circuit(
    feature_map: QuantumCircuit,
    left: np.ndarray,
    right: np.ndarray,
) -> QuantumCircuit:
    data_parameters = sorted(feature_map.parameters, key=lambda parameter: parameter.name)
    if len(data_parameters) != 4:
        raise ValueError("Frozen feature map must have four data parameters")
    left_map = feature_map.assign_parameters(dict(zip(data_parameters, left.tolist())), inplace=False)
    right_map = feature_map.assign_parameters(dict(zip(data_parameters, right.tolist())), inplace=False)
    circuit = QuantumCircuit(4, 4)
    circuit.compose(right_map, inplace=True)
    circuit.compose(left_map.inverse(), inplace=True)
    circuit.measure(range(4), range(4))
    return circuit


def _median_errors(backend: Any, properties: Any) -> dict[str, float | None]:
    single: list[float] = []
    two: list[float] = []
    readout: list[float] = []
    if properties is not None:
        for gate in properties.gates:
            try:
                value = float(properties.gate_error(gate.gate, gate.qubits))
            except Exception:
                continue
            if not math.isfinite(value):
                continue
            if len(gate.qubits) == 1:
                single.append(value)
            elif len(gate.qubits) == 2:
                two.append(value)
        for physical_index in range(backend.num_qubits):
            try:
                value = float(properties.readout_error(physical_index))
            except Exception:
                continue
            if math.isfinite(value):
                readout.append(value)
    return {
        "median_1q_gate_error": float(np.median(single)) if single else None,
        "median_2q_gate_error": float(np.median(two)) if two else None,
        "median_readout_error": float(np.median(readout)) if readout else None,
    }


def _backend_record(backend: Any) -> tuple[dict[str, Any], Any]:
    status = backend.status()
    try:
        properties = backend.properties()
    except Exception:
        properties = None
    record = {
        "backend_name": backend.name,
        "qubit_count": backend.num_qubits,
        "simulator": bool(getattr(backend, "simulator", False)),
        "operational": bool(status.operational),
        "status_msg": str(status.status_msg),
        "pending_jobs": int(status.pending_jobs),
        **_median_errors(backend, properties),
    }
    return record, properties


def _get_operational_backends(service: Any) -> list[Any]:
    backends = service.backends(min_num_qubits=4)
    candidates = []
    for backend in backends:
        name = str(backend.name).lower()
        if name in EXCLUDED_BACKENDS or "miami" in name:
            continue
        try:
            status = backend.status()
        except Exception:
            continue
        if (
            not status.operational
            or str(status.status_msg).strip().lower() != "active"
            or bool(getattr(backend, "simulator", False))
        ):
            continue
        candidates.append(backend)
    return candidates


def _compiled_candidate(
    backend: Any,
    logical_circuit: QuantumCircuit,
    initial_layout: list[int] | None = None,
) -> tuple[Any, dict[str, Any]]:
    compiled = transpile(
        logical_circuit,
        backend=backend,
        optimization_level=3,
        seed_transpiler=42,
        initial_layout=initial_layout,
    )
    operations = compiled.count_ops()
    gate_instructions = [
        instruction
        for instruction in compiled.data
        if instruction.operation.name not in {"measure", "reset", "barrier", "delay"}
    ]
    one_qubit_count = sum(instruction.operation.num_qubits == 1 for instruction in gate_instructions)
    two_qubit_count = sum(instruction.operation.num_qubits == 2 for instruction in gate_instructions)
    physical_layout: list[int]
    if compiled.layout is not None:
        try:
            physical_layout = [
                int(index)
                for index in compiled.layout.initial_index_layout(filter_ancillas=True)
            ][:4]
        except Exception:
            physical_layout = list(range(4))
    else:
        physical_layout = list(range(4))
    info = {
        "logical_qubits": 4,
        "physical_qubits": physical_layout,
        "physical_qubits_used": sorted(
            {
                compiled.find_bit(qubit).index
                for instruction in compiled.data
                for qubit in instruction.qubits
            }
        ),
        "depth_before_transpilation": int(logical_circuit.depth()),
        "depth_after_transpilation": int(compiled.depth()),
        "two_qubit_gate_count": int(two_qubit_count),
        "one_qubit_gate_count": int(one_qubit_count),
        "swap_count": int(operations.get("swap", 0)),
        "basis_gates": sorted(operations.keys()),
        "layout": str(compiled.layout),
        "transpiled_circuit_qubits": int(compiled.num_qubits),
    }
    return compiled, info


def _candidate_score(record: dict[str, Any], transpilation: dict[str, Any]) -> tuple[float, ...]:
    one = record.get("selected_median_1q_gate_error", record["median_1q_gate_error"])
    two = record.get("selected_median_2q_gate_error", record["median_2q_gate_error"])
    readout = record.get("selected_median_readout_error", record["median_readout_error"])
    if one is None or two is None or readout is None:
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    combined_error = (
        one * transpilation["one_qubit_gate_count"]
        + two * transpilation["two_qubit_gate_count"]
        + readout * 4
    )
    return (
        combined_error,
        float(record["pending_jobs"]),
        float(transpilation["depth_after_transpilation"]),
        float(record["qubit_count"]),
    )


def _kernel_pairs(data: dict[str, Any]) -> tuple[list[tuple[str, int, int, np.ndarray, np.ndarray]], dict[str, int]]:
    features = data["features"]
    train = data["train"]
    validation = data["validation"]
    test = data["tests_by_snr"]["clean"]
    train_x = train[features].to_numpy(dtype=float)
    validation_x = validation[features].to_numpy(dtype=float)
    test_x = test[features].to_numpy(dtype=float)
    pairs: list[tuple[str, int, int, np.ndarray, np.ndarray]] = []
    for left_index in range(len(train_x)):
        for right_index in range(left_index + 1, len(train_x)):
            pairs.append(("training", left_index, right_index, train_x[left_index], train_x[right_index]))
    if INCLUDE_VALIDATION_KERNEL:
        for left_index, left in enumerate(validation_x):
            for right_index, right in enumerate(train_x):
                pairs.append(("validation", left_index, right_index, left, right))
    for left_index, left in enumerate(test_x):
        for right_index, right in enumerate(train_x):
            pairs.append(("test", left_index, right_index, left, right))
    shapes = {
        "training": len(train_x),
        "validation": len(validation_x),
        "test": len(test_x),
    }
    return pairs, shapes


def _physical_to_compact_layout(compiled: QuantumCircuit) -> tuple[dict[int, int], list[int]]:
    if compiled.layout is None:
        return {index: index for index in range(compiled.num_qubits)}, list(range(compiled.num_qubits))
    try:
        logical_to_physical = [
            int(index)
            for index in compiled.layout.initial_index_layout(filter_ancillas=True)
        ]
    except Exception as exc:
        raise RuntimeError("Unable to resolve selected physical layout for Aer preview") from exc
    return {physical: logical for logical, physical in enumerate(logical_to_physical)}, logical_to_physical


def _selected_calibration_summary(
    properties: Any,
    compiled: QuantumCircuit,
    layout: list[int],
) -> dict[str, float | None]:
    one_qubit: list[float] = []
    two_qubit: list[float] = []
    readout: list[float] = []
    for instruction in compiled.data:
        if instruction.operation.num_qubits not in {1, 2}:
            continue
        qargs = [compiled.find_bit(qubit).index for qubit in instruction.qubits]
        try:
            error = float(properties.gate_error(instruction.operation.name, qargs))
        except Exception:
            continue
        if not math.isfinite(error):
            continue
        (one_qubit if len(qargs) == 1 else two_qubit).append(error)
    for qubit in layout:
        try:
            error = float(properties.readout_error(qubit))
        except Exception:
            continue
        if math.isfinite(error):
            readout.append(error)
    return {
        "selected_median_1q_gate_error": float(np.median(one_qubit)) if one_qubit else None,
        "selected_median_2q_gate_error": float(np.median(two_qubit)) if two_qubit else None,
        "selected_median_readout_error": float(np.median(readout)) if readout else None,
    }


def _estimated_qpu_seconds(
    backend: Any,
    properties: Any,
    compiled: QuantumCircuit,
    circuit_count: int,
    shots: int,
) -> float | None:
    if backend is None or circuit_count <= 0 or shots <= 0:
        return None
    try:
        scheduled = transpile(
            compiled,
            backend=backend,
            optimization_level=0,
            scheduling_method="alap",
        )
        duration_dt = float(scheduled.duration)
        dt_seconds = float(backend.target.dt)
    except Exception:
        return None
    if not math.isfinite(duration_dt) or not math.isfinite(dt_seconds) or duration_dt <= 0 or dt_seconds <= 0:
        return None
    # Each shot also waits the repetition delay, which dwarfs a ~4 us circuit
    # (250 us on Heron, at least 1 ms on Nighthawk). Omitting it made the old
    # estimate roughly 70x too low and the 25% allocation reserve meaningless.
    rep_delay = REP_DELAY_SECONDS
    if rep_delay is None:
        rep_delay = float(getattr(backend.configuration(), "default_rep_delay", 0.0) or 0.0)
    return (duration_dt * dt_seconds + rep_delay) * circuit_count * shots


def _calibration_snapshot(
    backend: Any,
    properties: Any,
    compiled: QuantumCircuit,
    layout: list[int],
) -> dict[str, Any]:
    gate_errors: list[dict[str, Any]] = []
    for instruction in compiled.data:
        if instruction.operation.name in {"measure", "reset", "barrier", "delay"}:
            continue
        if instruction.operation.num_qubits not in {1, 2}:
            continue
        physical_qubits = [compiled.find_bit(qubit).index for qubit in instruction.qubits]
        try:
            error = float(properties.gate_error(instruction.operation.name, physical_qubits))
        except Exception:
            error = None
        gate_errors.append(
            {
                "gate": instruction.operation.name,
                "physical_qubits": physical_qubits,
                "error": error,
            }
        )
    readout_errors = {}
    for qubit in layout:
        try:
            readout_errors[str(qubit)] = float(properties.readout_error(qubit))
        except Exception:
            readout_errors[str(qubit)] = None
    status = backend.status()
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend_name": str(backend.name),
        "operational": bool(status.operational),
        "status_msg": str(status.status_msg),
        "pending_jobs": int(status.pending_jobs),
        "physical_layout": list(layout),
        "selected_median_errors": _selected_calibration_summary(
            properties, compiled, layout
        ),
        "readout_error_by_physical_qubit": readout_errors,
        "compiled_gate_calibration_errors": gate_errors,
        "properties_last_update": str(getattr(properties, "last_update_date", None)),
    }


def _check_calibration_drift(
    current: dict[str, Any],
    reference: dict[str, Any],
    maximum_degradation: float = MAX_CALIBRATION_DEGRADATION,
) -> None:
    if not current.get("operational") or current.get("status_msg", "").lower() != "active":
        raise RuntimeError("Selected backend is not currently active")
    if current.get("physical_layout") != PREFERRED_LAYOUT:
        raise RuntimeError("Transpiled physical layout differs from the approved fixed layout")
    current_errors = current.get("selected_median_errors", {})
    reference_errors = reference.get("selected_median_errors", {})
    for key in (
        "selected_median_1q_gate_error",
        "selected_median_2q_gate_error",
        "selected_median_readout_error",
    ):
        current_value = current_errors.get(key)
        reference_value = reference_errors.get(key)
        if reference_value is None:
            reference_value = reference.get(key)
        if current_value is None or reference_value is None or reference_value <= 0:
            raise RuntimeError(f"Missing calibration value required for drift check: {key}")
        if current_value > reference_value * maximum_degradation:
            raise RuntimeError(
                f"Calibration drift exceeds {maximum_degradation:.0%} policy for {key}"
            )


def _calibration_comparison(current: dict[str, Any]) -> list[dict[str, float | str]]:
    labels = {
        "selected_median_1q_gate_error": "1Q median",
        "selected_median_2q_gate_error": "2Q median",
        "selected_median_readout_error": "Readout median",
    }
    current_errors = current.get("selected_median_errors", {})
    return [
        {
            "metric": labels[key],
            "reference": reference,
            "current": float(current_errors[key]),
            "change_percent": (float(current_errors[key]) / reference - 1.0) * 100.0,
        }
        for key, reference in APPROVED_CALIBRATION_REFERENCE.items()
    ]


def _backend_noise_model(properties: Any, used_physical: list[int]) -> tuple[NoiseModel, dict[str, Any]]:
    if properties is None:
        raise RuntimeError("Selected backend has no calibration properties for noise preview")
    physical_to_local = {physical: local for local, physical in enumerate(used_physical)}
    model = NoiseModel()
    gate_records = []
    for gate in properties.gates:
        qargs = tuple(int(qubit) for qubit in gate.qubits)
        if not qargs or not set(qargs).issubset(physical_to_local):
            continue
        try:
            error_probability = float(properties.gate_error(gate.gate, list(qargs)))
        except Exception:
            continue
        if not math.isfinite(error_probability) or error_probability <= 0 or error_probability >= 1:
            continue
        local_qargs = [physical_to_local[qubit] for qubit in qargs]
        error = depolarizing_error(error_probability, len(local_qargs))
        model.add_quantum_error(error, gate.gate, local_qargs)
        gate_records.append(
            {
                "gate": gate.gate,
                "physical_qubits": list(qargs),
                "error_probability": error_probability,
            }
        )
    readout_records = []
    for physical in used_physical:
        try:
            probability = float(properties.readout_error(physical))
        except Exception:
            continue
        if not math.isfinite(probability) or probability <= 0 or probability >= 1:
            continue
        model.add_readout_error(
            ReadoutError([[1 - probability, probability], [probability, 1 - probability]]),
            [physical_to_local[physical]],
        )
        readout_records.append({"physical_qubit": physical, "readout_error": probability})
    metadata = {
        "method": "Per-gate depolarizing and symmetric readout errors derived from selected IBM backend properties; T1/T2 not modeled.",
        "physical_to_local_qubits": physical_to_local,
        "gate_error_entries": gate_records,
        "readout_error_entries": readout_records,
    }
    return model, metadata


def _compact_compiled_circuit(
    circuit: QuantumCircuit,
    used_physical: list[int],
    physical_to_compact: dict[int, int],
) -> QuantumCircuit:
    compact = QuantumCircuit(len(used_physical), 4)
    compact.global_phase = circuit.global_phase
    for instruction in circuit.data:
        physical_qargs = [circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if any(index not in physical_to_compact for index in physical_qargs):
            raise RuntimeError("Transpiled circuit uses a physical qubit outside the preview layout")
        compact.append(
            instruction.operation,
            [compact.qubits[physical_to_compact[index]] for index in physical_qargs],
            [compact.clbits[index] for index in (circuit.find_bit(bit).index for bit in instruction.clbits)],
        )
    return compact


def _run_backend_noise_preview(
    data: dict[str, Any],
    backend: Any,
    representative_compiled: QuantumCircuit,
    properties: Any,
    shots: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs, sizes = _kernel_pairs(data)
    _, logical_layout = _physical_to_compact_layout(representative_compiled)
    used_physical = sorted(set(logical_layout))
    physical_to_compact = {physical: local for local, physical in enumerate(used_physical)}
    noise_model, noise_metadata = _backend_noise_model(properties, used_physical)
    simulator = AerSimulator(method="matrix_product_state", noise_model=noise_model)
    feature_map = _frozen_feature_map(data)
    compiled_circuits: list[QuantumCircuit] = []
    pair_batches: list[list[tuple[str, int, int, np.ndarray, np.ndarray]]] = []
    batch_size = 128
    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start : start + batch_size]
        raw = [
            _compute_uncompute_circuit(feature_map, left, right)
            for _, _, _, left, right in batch_pairs
        ]
        backend_native = transpile(
            raw,
            backend=backend,
            optimization_level=3,
            seed_transpiler=42,
            initial_layout=logical_layout,
            layout_method="trivial",
            routing_method="sabre",
        )
        for circuit in backend_native:
            circuit_physical_qubits = sorted(
                {circuit.find_bit(qubit).index for instruction in circuit.data for qubit in instruction.qubits}
            )
            if not set(circuit_physical_qubits).issubset(set(used_physical)):
                raise RuntimeError("Backend routing expanded beyond the selected 4-qubit layout")
            compiled_circuits.append(circuit)
        pair_batches.append(batch_pairs)

    matrices = {
        "training": np.eye(sizes["training"], dtype=float),
        "validation": np.zeros((sizes["validation"], sizes["training"]), dtype=float),
        "test": np.zeros((sizes["test"], sizes["training"]), dtype=float),
    }
    start_time = time.perf_counter()
    cursor = 0
    for batch_pairs in pair_batches:
        backend_circuits = compiled_circuits[cursor : cursor + len(batch_pairs)]
        cursor += len(batch_pairs)
        compact_circuits = [
            _compact_compiled_circuit(circuit, used_physical, physical_to_compact)
            for circuit in backend_circuits
        ]
        aer_circuits = transpile(
            compact_circuits,
            backend=simulator,
            optimization_level=0,
            seed_transpiler=42,
        )
        job = simulator.run(aer_circuits, shots=shots)
        result = job.result()
        for local_index, pair in enumerate(batch_pairs):
            split, left_index, right_index, _, _ = pair
            counts = result.get_counts(local_index)
            probability_zero = counts.get("0" * 4, 0) / shots
            if split == "training":
                matrices[split][left_index, right_index] = probability_zero
                matrices[split][right_index, left_index] = probability_zero
            else:
                matrices[split][left_index, right_index] = probability_zero

    train_y = data["train"].binary_label.to_numpy(dtype=int)
    model = SVC(kernel="precomputed", C=0.1, class_weight="balanced")
    model.fit(matrices["training"], train_y)
    test = data["tests_by_snr"]["clean"]
    prediction = model.predict(matrices["test"])
    decision = model.decision_function(matrices["test"])
    metrics = calculate_metrics(test.binary_label.to_numpy(dtype=int), prediction, decision)
    metadata = {
        "shots": shots,
        "circuit_count": len(pairs),
        "estimated_jobs_if_submitted": math.ceil(len(pairs) / CIRCUITS_PER_JOB),
        "local_preview_seconds": time.perf_counter() - start_time,
        "physical_qubits": used_physical,
        "preview_basis_gates": sorted(set().union(*(set(circuit.count_ops()) for circuit in compiled_circuits))),
        "noise_model": noise_metadata,
        "metric_source": "Local Aer preview only; not a hardware result.",
    }
    row = {
        "model": "backend_derived_noisy_simulator",
        "evaluation_split": "clean_test",
        "shots": shots,
        "backend_name": "backend-calibration-derived-Aer",
        "physical_qubits": json.dumps(used_physical),
        "circuit_count": len(pairs),
        **metrics,
    }
    return metadata, [row]


def _stage6_rows() -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    ideal = pd.read_csv(RESULTS_QUANTUM_DIR / "ideal_simulator_results.csv")
    ideal = ideal[
        (ideal.feature_count == 4)
        & (ideal.training_size_requested == 24)
        & ideal.training_snr.astype(str).eq("clean")
        & ideal.evaluation_snr.astype(str).eq("clean")
        & ideal.evaluation_split.eq("test")
    ]
    stage6 = pd.read_csv(RESULTS_QUANTUM_DIR / "noisy_simulator_results.csv")
    stage6 = stage6[(stage6.noise_type == "combined") & (stage6.noise_label == "0pct")]
    if len(stage6) != 1:
        raise ValueError("Expected one Stage 6 finite-shot 0%-noise row")
    output: dict[str, dict[str, float]] = {}
    for model_name, frame in ideal.groupby("model"):
        if model_name not in {"rbf_svm", "small_mlp", "trainable_qsvc"}:
            continue
        row = frame.iloc[0]
        output[str(model_name)] = {
            key: float(row[key])
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy", "drone_recall", "non_drone_recall")
        }
    noisy_row = stage6.iloc[0]
    output["stage6_zero_noise_finite_shot"] = {
        key: float(noisy_row[key])
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy", "drone_recall", "non_drone_recall")
    }
    stage6_high = stage6.copy()
    all_stage6 = pd.read_csv(RESULTS_QUANTUM_DIR / "noisy_simulator_results.csv")
    all_stage6 = all_stage6[(all_stage6.noise_type == "combined") & (all_stage6.noise_label == "2pct")]
    if len(all_stage6) == 1:
        row = all_stage6.iloc[0]
        output["stage6_combined_2pct"] = {
            key: float(row[key])
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy", "drone_recall", "non_drone_recall")
        }
    return output, {str(row.model): row._asdict() for row in []}


def _write_comparison(
    baselines: dict[str, dict[str, float]],
    preview_row: dict[str, Any] | None,
    qpu_row: dict[str, Any] | None,
) -> None:
    rows: list[dict[str, Any]] = []
    labels = {
        "rbf_svm": "RBF SVM",
        "small_mlp": "Small MLP",
        "trainable_qsvc": "Ideal quantum simulator",
        "stage6_zero_noise_finite_shot": "Finite-shot zero-noise simulator",
        "stage6_combined_2pct": "Stage 6 noisy simulator (2% synthetic)",
    }
    for key, label in labels.items():
        if key not in baselines:
            continue
        rows.append({"model": label, "status": "measured", **baselines[key]})
    preview_metrics = {
        key: value
        for key, value in (preview_row or {}).items()
        if key not in {"model", "status"}
    }
    rows.append(
        {
            **preview_metrics,
            "model": "Backend-derived noisy simulator",
            "status": "measured" if preview_row is not None else "not_run",
        }
    )
    rows.append(
        {
            **(qpu_row or {}),
            "model": "REAL IBM QPU",
            # Set after the spread: qpu_row carries status "completed", which
            # previously overwrote "measured" and dropped the bar from the plot.
            "status": "measured" if qpu_row is not None else "not_submitted",
        }
    )
    RESULTS_QUANTUM_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(COMPARISON_PATH, index=False)

    metrics = ("f1", "balanced_accuracy", "drone_recall", "non_drone_recall")
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    plotted = [row for row in rows if row.get("status") == "measured"]
    x = np.arange(len(plotted))
    width = 0.19
    for index, metric in enumerate(metrics):
        values = [float(row[metric]) if row.get(metric) is not None and pd.notna(row.get(metric)) else np.nan for row in plotted]
        ax.bar(x + (index - 1.5) * width, values, width, label=metric.replace("_", " "))
    ax.set_xticks(x, [row["model"] for row in plotted], rotation=22, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Frozen QubitSky model: simulator and hardware comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    RESULTS_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_PATH, dpi=180)
    plt.close(fig)


def _print_performance_deltas(
    baselines: dict[str, dict[str, float]],
    preview_row: dict[str, Any],
    qpu_row: dict[str, Any],
) -> None:
    metric_names = ("f1", "balanced_accuracy", "drone_recall", "non_drone_recall")
    stages = [
        ("ideal simulator", baselines["trainable_qsvc"]),
        ("backend-derived Aer", preview_row),
        ("real QPU", qpu_row),
    ]
    print("PERFORMANCE CHANGES (absolute score points)")
    for (from_name, from_metrics), (to_name, to_metrics) in zip(stages, stages[1:]):
        changes = {
            metric: float(to_metrics[metric]) - float(from_metrics[metric])
            for metric in metric_names
            if metric in from_metrics
            and metric in to_metrics
            and pd.notna(from_metrics[metric])
            and pd.notna(to_metrics[metric])
        }
        print(f"{from_name} -> {to_name}: {json.dumps(changes, sort_keys=True)}")


def _write_preflight_artifacts(
    backend_records: list[dict[str, Any]],
    selected_record: dict[str, Any],
    transpile_info: dict[str, Any],
    preview_metadata: dict[str, Any],
    preview_row: dict[str, Any],
    baselines: dict[str, dict[str, float]],
) -> None:
    RESULTS_QUANTUM_DIR.mkdir(parents=True, exist_ok=True)
    safe_records = {
        "selected_backend": selected_record,
        "all_operational_physical_backends": backend_records,
        "transpilation": transpile_info,
        "aer_noise_preview": preview_metadata,
        "credential_values_written": False,
    }
    BACKEND_PROPERTIES_PATH.write_text(json.dumps(safe_records, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame([preview_row]).to_csv(BACKEND_PREVIEW_PATH, index=False)
    _write_comparison(baselines, preview_row, None)
    if not JOBS_PATH.exists():
        JOBS_PATH.write_text(json.dumps({"status": "not_submitted", "jobs": []}, indent=2) + "\n", encoding="utf-8")
    if not QPU_RESULTS_PATH.exists():
        pd.DataFrame([{"status": "not_submitted"}]).to_csv(QPU_RESULTS_PATH, index=False)


def _submit_batches(
    backend: Any,
    properties: Any,
    data: dict[str, Any],
    selected_layout: list[int],
    shots: int,
) -> dict[str, Any]:
    """Submit independent Runtime jobs, persisting each job ID before waiting."""
    from qiskit_ibm_runtime import SamplerV2

    if backend.name != PREFERRED_BACKEND or selected_layout != PREFERRED_LAYOUT:
        raise RuntimeError("Submission target differs from the explicitly approved backend/layout")
    pairs, sizes = _kernel_pairs(data)
    if len(pairs) != _expected_circuit_count(sizes):
        raise RuntimeError(f"Frozen workload changed unexpectedly: {len(pairs)} circuits")
    jobs_data, completed_until = _validated_resume_state(pairs, selected_layout, shots)
    feature_map = _frozen_feature_map(data)
    logical_circuits = [
        _compute_uncompute_circuit(feature_map, left, right)
        for _, _, _, left, right in pairs[completed_until:]
    ]
    all_compiled = transpile(
        logical_circuits,
        backend=backend,
        optimization_level=3,
        seed_transpiler=42,
        initial_layout=selected_layout,
    )
    if len(all_compiled) != len(pairs) - completed_until:
        raise RuntimeError("Transpiler changed the number of frozen kernel circuits")
    for compiled_circuit in all_compiled:
        if compiled_circuit.layout is None:
            raise RuntimeError("A submitted circuit lost its physical layout")
        compiled_layout = [
            int(index)
            for index in compiled_circuit.layout.initial_index_layout(filter_ancillas=True)
        ][:4]
        if compiled_layout != selected_layout:
            raise RuntimeError("A submitted circuit does not preserve the approved physical layout")
    train_y = data["train"].binary_label.to_numpy(dtype=int)
    matrices = {
        "training": np.eye(sizes["training"], dtype=float),
        "validation": np.zeros((sizes["validation"], sizes["training"]), dtype=float),
        "test": np.zeros((sizes["test"], sizes["training"]), dtype=float),
    }
    JOB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    jobs_data["status"] = "running"
    jobs_data["resume_started_at_utc"] = datetime.now(timezone.utc).isoformat()

    for batch_start in range(completed_until, len(pairs), CIRCUITS_PER_JOB):
        batch_pairs = pairs[batch_start : batch_start + CIRCUITS_PER_JOB]
        compiled_start = batch_start - completed_until
        batch_circuits = all_compiled[compiled_start : compiled_start + len(batch_pairs)]
        batch_end = batch_start + len(batch_pairs)
        execution_count = len(batch_circuits) * shots
        if execution_count > MAX_EXECUTIONS_PER_JOB:
            raise RuntimeError(
                f"Batch {batch_start}:{batch_end} exceeds the {MAX_EXECUTIONS_PER_JOB:,} execution/job limit"
            )
        split_counts = {
            split: sum(pair[0] == split for pair in batch_pairs)
            for split in ("training", "validation", "test")
        }
        current_snapshot = _calibration_snapshot(
            backend, properties, batch_circuits[0], selected_layout
        )
        entry: dict[str, Any] = {
            "job_id": None,
            "backend": backend.name,
            "physical_layout": list(selected_layout),
            "circuit_index_start": batch_start,
            "circuit_index_end_exclusive": batch_end,
            "circuit_index_range_inclusive": [batch_start, batch_end - 1],
            "circuit_count": len(batch_circuits),
            "execution_count": execution_count,
            "shots": shots,
            "split_circuit_counts": split_counts,
            "submitted_at_utc": None,
            "status": "submitting",
            "calibration_snapshot": current_snapshot,
        }
        jobs_data["jobs"].append(entry)
        JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
        JOBS_PATH.write_text(json.dumps(jobs_data, indent=2) + "\n", encoding="utf-8")
        sampler = SamplerV2(mode=backend)
        if REP_DELAY_SECONDS is not None:
            sampler.options.execution.rep_delay = REP_DELAY_SECONDS
        try:
            job = sampler.run(batch_circuits, shots=shots)
        except Exception as exc:
            entry["status"] = "submission_error_unknown_job_state"
            entry["failure_type"] = type(exc).__name__
            JOBS_PATH.write_text(json.dumps(jobs_data, indent=2) + "\n", encoding="utf-8")
            raise QPUJobFailure(
                f"Submission call failed for circuit range {batch_start}-{batch_end - 1}; "
                "job acceptance is unknown, so no automatic retry was attempted."
            ) from exc
        try:
            job_id = str(job.job_id())
        except Exception as exc:
            entry["status"] = "submitted_job_id_unavailable"
            entry["failure_type"] = type(exc).__name__
            JOBS_PATH.write_text(json.dumps(jobs_data, indent=2) + "\n", encoding="utf-8")
            raise QPUJobFailure(
                f"Runtime accepted or may have accepted circuit range {batch_start}-{batch_end - 1}, "
                "but returned no readable job ID; stopped without retry."
            ) from exc
        entry["job_id"] = job_id
        entry["submitted_at_utc"] = datetime.now(timezone.utc).isoformat()
        entry["status"] = "submitted"
        JOBS_PATH.write_text(json.dumps(jobs_data, indent=2) + "\n", encoding="utf-8")
        try:
            result = job.result()
            counts_by_pair = []
            for index in range(len(batch_pairs)):
                data_registers = result[index].data
                register_name = next(
                    name
                    for name in data_registers.keys()
                    if hasattr(getattr(data_registers, name), "get_counts")
                )
                counts_by_pair.append(getattr(data_registers, register_name).get_counts())
            batch_cache = {
                "job_id": job_id,
                "backend": backend.name,
                "physical_layout": list(selected_layout),
                "circuit_index_start": batch_start,
                "circuit_index_end_exclusive": batch_end,
                "shots": shots,
                "counts": counts_by_pair,
                "circuit_indices": list(range(batch_start, batch_end)),
                "pairs": [
                    {"split": split, "left_index": i, "right_index": j}
                    for split, i, j, _, _ in batch_pairs
                ],
            }
            cache_file = JOB_CACHE_DIR / f"{job_id}.json"
            cache_file.write_text(json.dumps(batch_cache, indent=2) + "\n", encoding="utf-8")
            entry["status"] = "completed"
            entry["result_cache"] = str(cache_file)
            entry["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            JOBS_PATH.write_text(json.dumps(jobs_data, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            entry["status"] = "failed"
            entry["failure_type"] = type(exc).__name__
            entry["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
            JOBS_PATH.write_text(json.dumps(jobs_data, indent=2) + "\n", encoding="utf-8")
            raise QPUJobFailure(
                f"QPU job {job_id} failed for circuit range {batch_start}-{batch_end - 1}; "
                "execution stopped and no automatic retry was attempted."
            ) from exc

        if batch_end < len(pairs):
            jobs_data["status"] = f"paused_after_batch_{len(jobs_data['jobs'])}"
            jobs_data["paused_at_utc"] = datetime.now(timezone.utc).isoformat()
            JOBS_PATH.write_text(json.dumps(jobs_data, indent=2) + "\n", encoding="utf-8")
            return {
                "status": "batch_completed",
                "job_id": job_id,
                "circuit_index_start": batch_start,
                "circuit_index_end_exclusive": batch_end,
                "circuit_count": len(batch_circuits),
                "execution_count": execution_count,
            }

    covered_circuit_indices: list[int] = []
    for job_record in jobs_data["jobs"]:
        if job_record.get("status") != "completed":
            raise RuntimeError(
                f"Job {job_record.get('job_id')} is not completed for circuit range "
                f"{job_record.get('circuit_index_start')}-{job_record.get('circuit_index_end_exclusive', 1) - 1}"
            )
        cache = json.loads(Path(job_record["result_cache"]).read_text(encoding="utf-8"))
        covered_circuit_indices.extend(cache["circuit_indices"])
        for pair, counts in zip(cache["pairs"], cache["counts"]):
            probability_zero = counts.get("0" * 4, 0) / shots
            split = pair["split"]
            i = int(pair["left_index"])
            j = int(pair["right_index"])
            if split == "training":
                matrices[split][i, j] = probability_zero
                matrices[split][j, i] = probability_zero
            else:
                matrices[split][i, j] = probability_zero

    if covered_circuit_indices != list(range(len(pairs))):
        raise RuntimeError("QPU result cache does not cover the exact ordered circuit index range")
    expected_shapes = {
        "training": (24, 24),
        "validation": (80, 24),
        "test": (98, 24),
    }
    for name, matrix in matrices.items():
        if matrix.shape != expected_shapes[name] or not np.isfinite(matrix).all():
            raise RuntimeError(f"Reconstructed {name} kernel matrix has invalid shape or values")
    if not np.allclose(matrices["training"], matrices["training"].T, rtol=0, atol=1e-12):
        raise RuntimeError("Reconstructed training kernel is not symmetric")

    model = SVC(kernel="precomputed", C=0.1, class_weight="balanced")
    model.fit(matrices["training"], train_y)
    test = data["tests_by_snr"]["clean"]
    prediction = model.predict(matrices["test"])
    decision = model.decision_function(matrices["test"])
    metrics = calculate_metrics(test.binary_label.to_numpy(dtype=int), prediction, decision)
    confusion = json.loads(metrics["confusion_matrix"])
    qpu_row = {
        "model": "REAL IBM QPU",
        "status": "completed",
        "backend_name": backend.name,
        "physical_qubits": json.dumps(selected_layout),
        "shots": shots,
        "circuit_count": len(pairs),
        "job_count": len(jobs_data["jobs"]),
        "job_ids": json.dumps([job_record["job_id"] for job_record in jobs_data["jobs"]]),
        "physical_layout": json.dumps(selected_layout),
        "training_kernel_shape": json.dumps(list(matrices["training"].shape)),
        "validation_kernel_shape": json.dumps(list(matrices["validation"].shape)),
        "test_kernel_shape": json.dumps(list(matrices["test"].shape)),
        "training_kernel_symmetric": True,
        **metrics,
        "confusion_matrix": json.dumps(confusion),
    }
    pd.DataFrame([qpu_row]).to_csv(QPU_RESULTS_PATH, index=False)
    result_cache_path = JOB_CACHE_DIR / "assembled_kernel_metrics.json"
    result_cache_path.write_text(
        json.dumps({"training_kernel": matrices["training"].tolist(), "validation_kernel": matrices["validation"].tolist(), "test_kernel": matrices["test"].tolist(), "metrics": qpu_row}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    jobs_data["status"] = "completed"
    jobs_data["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    jobs_data["matrix_shapes"] = {name: list(matrix.shape) for name, matrix in matrices.items()}
    jobs_data["training_kernel_symmetric"] = True
    jobs_data["result_cache"] = str(result_cache_path)
    JOBS_PATH.write_text(json.dumps(jobs_data, indent=2) + "\n", encoding="utf-8")
    return qpu_row


def _validated_resume_state(
    pairs: list[tuple[Any, ...]],
    selected_layout: list[int],
    shots: int,
) -> tuple[dict[str, Any], int]:
    """Validate cached completed batches as an exact prefix before resuming."""
    initial_state: dict[str, Any] = {
        "status": "running",
        "backend": PREFERRED_BACKEND,
        "physical_layout": list(selected_layout),
        "shots": shots,
        "total_circuits": len(pairs),
        "total_executions": len(pairs) * shots,
        "circuits_per_job_target": CIRCUITS_PER_JOB,
        "maximum_executions_per_job": MAX_EXECUTIONS_PER_JOB,
        "jobs": [],
    }
    if not JOBS_PATH.is_file():
        return initial_state, 0

    previous = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
    if previous.get("status") == "not_submitted" and not previous.get("jobs"):
        # Placeholder written by a preflight-only run (_write_preflight_artifacts);
        # nothing was ever submitted, so there is no prefix to resume.
        return initial_state, 0
    expected_metadata = {
        "backend": PREFERRED_BACKEND,
        "physical_layout": list(selected_layout),
        "shots": shots,
        "total_circuits": len(pairs),
        "total_executions": len(pairs) * shots,
        "circuits_per_job_target": CIRCUITS_PER_JOB,
        "maximum_executions_per_job": MAX_EXECUTIONS_PER_JOB,
    }
    for key, expected in expected_metadata.items():
        if previous.get(key) != expected:
            raise RuntimeError(f"Existing QPU ledger {key} differs from the frozen workload")

    previous_jobs = previous.get("jobs", [])
    if not isinstance(previous_jobs, list):
        raise RuntimeError("Existing QPU ledger has an invalid jobs list")

    covered_until = 0
    for job in previous_jobs:
        expected_start = covered_until
        expected_end = min(expected_start + CIRCUITS_PER_JOB, len(pairs))
        if job.get("status") != "completed":
            raise RuntimeError(
                "Existing QPU batch is not completed; refusing to retry an uncertain or failed job."
            )
        expected_job_fields = {
            "backend": PREFERRED_BACKEND,
            "physical_layout": list(selected_layout),
            "shots": shots,
            "circuit_index_start": expected_start,
            "circuit_index_end_exclusive": expected_end,
            "circuit_count": expected_end - expected_start,
        }
        for key, expected in expected_job_fields.items():
            if job.get(key) != expected:
                raise RuntimeError(
                    "Existing QPU jobs are not an exact contiguous prefix of the frozen workload"
                )

        job_id = job.get("job_id")
        cache_path_value = job.get("result_cache")
        if not job_id or not cache_path_value:
            raise RuntimeError("Completed QPU batch is missing its job ID or result cache")
        cache_path = Path(cache_path_value)
        if not cache_path.is_absolute():
            cache_path = PROJECT_ROOT / cache_path
        if not cache_path.is_file():
            raise RuntimeError(f"Completed QPU result cache is missing for job {job_id}")
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        cache_fields = {
            "job_id": job_id,
            "backend": PREFERRED_BACKEND,
            "physical_layout": list(selected_layout),
            "shots": shots,
            "circuit_index_start": expected_start,
            "circuit_index_end_exclusive": expected_end,
        }
        if any(cache.get(key) != value for key, value in cache_fields.items()):
            raise RuntimeError(f"Cached results do not match ledger metadata for job {job_id}")

        expected_indices = list(range(expected_start, expected_end))
        expected_pairs = [
            {"split": pair[0], "left_index": int(pair[1]), "right_index": int(pair[2])}
            for pair in pairs[expected_start:expected_end]
        ]
        counts = cache.get("counts")
        if (
            cache.get("circuit_indices") != expected_indices
            or cache.get("pairs") != expected_pairs
            or not isinstance(counts, list)
            or len(counts) != expected_end - expected_start
        ):
            raise RuntimeError(f"Cached circuit mapping is incomplete or mismatched for job {job_id}")
        for count_map in counts:
            if (
                not isinstance(count_map, dict)
                or sum(count_map.values()) != shots
                or any(
                    not isinstance(bitstring, str)
                    or len(bitstring) != 4
                    or set(bitstring).difference("01")
                    or not isinstance(value, int)
                    or value < 0
                    for bitstring, value in count_map.items()
                )
            ):
                raise RuntimeError(f"Cached shot counts are invalid for job {job_id}")
        covered_until = expected_end

    resumed_state = dict(previous)
    resumed_state["jobs"] = previous_jobs
    return resumed_state, covered_until


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-real-qpu", action="store_true", help="First required QPU authorization gate.")
    parser.add_argument("--submit-after-review", action="store_true", help="Second, post-preflight approval gate. Never use on the initial preflight run.")
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--skip-noise-preview", action="store_true", help="Skip local backend-calibration-derived Aer preview.")
    parser.add_argument("--credentials", type=Path, default=PROJECT_ROOT / "secrets" / "ibm_quantum.env")
    parser.add_argument("--backend", default="ibm_pittsburgh", choices=sorted(BACKEND_PROFILES), help="Backend profile (layout, calibration reference, repetition delay).")
    parser.add_argument("--skip-validation-kernel", action="store_true", help="Omit validation-vs-training circuits; the hardware score uses only training and test kernels.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.shots not in {256, 512, 1024}:
        print("Preflight blocked: shots must be 256, 512 or 1024.", file=sys.stderr)
        return 2
    _apply_backend_profile(args.backend, include_validation_kernel=not args.skip_validation_kernel)
    try:
        _load_local_env(args.credentials)
        token = os.environ.get("QISKIT_IBM_TOKEN", "").strip()
        instance = os.environ.get("QISKIT_IBM_INSTANCE", "").strip()
        if not token or not instance:
            print("Preflight blocked: local token and instance CRN must be configured.", file=sys.stderr)
            return 2
        config, data = _load_frozen_data()
        pairs, sizes = _kernel_pairs(data)
        if len(pairs) != _expected_circuit_count(sizes):
            raise RuntimeError(f"Frozen workload changed unexpectedly: {len(pairs)} circuits")
        _, completed_until = _validated_resume_state(pairs, PREFERRED_LAYOUT, args.shots)
        remaining_ranges = [
            (start, min(start + CIRCUITS_PER_JOB, len(pairs)))
            for start in range(completed_until, len(pairs), CIRCUITS_PER_JOB)
        ]
        if not remaining_ranges:
            raise RuntimeError("No unsubmitted circuit ranges remain in the frozen workload")
        batch_start, batch_end = remaining_ranges[0]
        batch_number = batch_start // CIRCUITS_PER_JOB + 1
        if os.environ.get("RUN_REAL_QPU", "NO").strip().upper() != "YES" or not args.confirm_real_qpu:
            print("Submission gates are not both enabled; preflight may proceed, but submission is disabled.")

        # Runtime connection is deliberately scoped to this Stage 7 script.
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(
            channel="ibm_quantum_platform", token=token, instance=instance
        )
        backend = service.backend(PREFERRED_BACKEND)
        backend_record, properties = _backend_record(backend)
        if (
            not backend_record["operational"]
            or backend_record["status_msg"].strip().lower() != "active"
            or backend_record["simulator"]
        ):
            raise RuntimeError(
                f"Approved backend {PREFERRED_BACKEND} is not an active physical QPU"
            )

        logical_map = _frozen_feature_map(data)
        representative = _compute_uncompute_circuit(
            logical_map,
            data["train"][data["features"]].iloc[0].to_numpy(dtype=float),
            data["train"][data["features"]].iloc[1].to_numpy(dtype=float),
        )
        compiled, transpile_info = _compiled_candidate(
            backend, representative, initial_layout=PREFERRED_LAYOUT
        )
        if transpile_info["physical_qubits"] != PREFERRED_LAYOUT:
            raise RuntimeError("Initial transpilation did not preserve the preferred physical layout")
        selected_record = dict(backend_record)
        selected_record.update(transpile_info)
        selected_record.update(
            _selected_calibration_summary(properties, compiled, PREFERRED_LAYOUT)
        )
        batch_circuit_count = batch_end - batch_start
        batch_execution_count = batch_circuit_count * args.shots
        selected_record["estimated_qpu_seconds"] = _estimated_qpu_seconds(
            backend, properties, compiled, batch_circuit_count, args.shots
        )
        calibration_snapshot = _calibration_snapshot(
            backend, properties, compiled, PREFERRED_LAYOUT
        )
        calibration_pass = True
        calibration_failure = None
        try:
            _check_calibration_drift(
                calibration_snapshot,
                {"selected_median_errors": APPROVED_CALIBRATION_REFERENCE},
            )
        except RuntimeError as exc:
            calibration_pass = False
            calibration_failure = str(exc)
        candidate_rows = [selected_record]
        total_executions = len(pairs) * args.shots
        if total_executions > MAX_EXECUTIONS_PER_JOB * 4:
            raise RuntimeError("Frozen workload exceeds the planned aggregate Runtime execution envelope")
        if batch_execution_count > MAX_EXECUTIONS_PER_JOB:
            raise RuntimeError("Planned job exceeds the current 10,000,000-execution Sampler limit")
        estimated_qpu_seconds = _estimated_qpu_seconds(
            backend, properties, compiled, batch_circuit_count, args.shots
        )
        if estimated_qpu_seconds is None:
            raise RuntimeError("Could not estimate backend-scheduled QPU execution time")
        usage = service.usage()
        remaining_seconds = usage.get("usage_remaining_seconds")
        remaining_ru = None
        if usage.get("usage_limit_ru") is not None and usage.get("usage_consumed_ru") is not None:
            remaining_ru = float(usage["usage_limit_ru"]) - float(usage["usage_consumed_ru"])
        allocation_pass = (
            remaining_seconds is not None
            and float(remaining_seconds) >= estimated_qpu_seconds * 1.25
            and not usage.get("usage_limit_reached")
        )

        print("\nREAD-ONLY STAGE 7 RESUME PREFLIGHT")
        print(f"VALIDATED COMPLETED BATCHES: {completed_until // CIRCUITS_PER_JOB} ({completed_until} circuits)")
        print(f"BACKEND: {backend.name}; status={backend_record['status_msg']}; operational={backend_record['operational']}; queue={backend_record['pending_jobs']} pending")
        print(f"LAYOUT: logical 4 -> physical {PREFERRED_LAYOUT}; shots={args.shots}")
        for metric in _calibration_comparison(calibration_snapshot):
            print(
                f"CALIBRATION {metric['metric']}: current={metric['current']:.8f}; "
                f"reference={metric['reference']:.8f}; change={metric['change_percent']:+.2f}%"
            )
        print(f"CALIBRATION GUARD: {'PASS' if calibration_pass else 'FAIL'} (maximum degradation 25%)")
        print(
            f"REMAINING ALLOCATION: {remaining_seconds if remaining_seconds is not None else 'unknown'} seconds; "
            f"{remaining_ru if remaining_ru is not None else 'unknown'} RUs"
        )
        print(
            f"BATCH {batch_number}: circuit range [{batch_start}, {batch_end - 1}] "
            f"(end-exclusive {batch_end}); circuits={batch_circuit_count}; "
            f"executions={batch_execution_count:,}"
        )
        print(
            f"EXPECTED USAGE: estimated {estimated_qpu_seconds:.2f} seconds; "
            f"25% reserve required={estimated_qpu_seconds * 1.25:.2f} seconds"
        )
        if not calibration_pass:
            print(f"STOP: calibration guard failed: {calibration_failure}. No QPU job was submitted.")
            return 1
        if not allocation_pass:
            print("STOP: IBM allocation is unavailable or lacks the 25% execution-time reserve. No QPU job was submitted.")
            return 1

        if args.skip_noise_preview:
            if not BACKEND_PREVIEW_PATH.is_file():
                raise RuntimeError("Cannot skip noise preview because no saved backend preview exists")
            preview_row = pd.read_csv(BACKEND_PREVIEW_PATH).iloc[0].to_dict()
            preview_metadata = {
                "status": "reused_saved_preview",
                "source": str(BACKEND_PREVIEW_PATH),
                "note": "Current backend calibration is rechecked independently before submission.",
            }
        else:
            preview_metadata, preview_rows = _run_backend_noise_preview(
                data, backend, compiled, properties, args.shots
            )
            preview_row = preview_rows[0]
        baselines, _ = _stage6_rows()
        selected_record["selected_median_errors"] = calibration_snapshot["selected_median_errors"]
        selected_record["readout_error_by_physical_qubit"] = calibration_snapshot[
            "readout_error_by_physical_qubit"
        ]
        selected_record["calibration_captured_at_utc"] = calibration_snapshot[
            "captured_at_utc"
        ]
        selected_record["calibration_degradation_limit"] = MAX_CALIBRATION_DEGRADATION
        _write_preflight_artifacts(
            candidate_rows,
            selected_record,
            transpile_info,
            preview_metadata,
            preview_row,
            baselines,
        )
        print(f"Backend properties: {BACKEND_PROPERTIES_PATH}")
        print(f"Backend noise preview: {BACKEND_PREVIEW_PATH}")
        print(f"Comparison table: {COMPARISON_PATH}")
        print(f"Comparison plot: {PLOT_PATH}")

        print("STOP: fresh preflight complete. Wait for explicit batch approval; no QPU job was submitted.")

        both_requested_gates = (
            os.environ.get("RUN_REAL_QPU", "NO").strip().upper() == "YES"
            and args.confirm_real_qpu
        )
        if not both_requested_gates:
            print("STOP: QPU submission not authorized. No IBM job was submitted.")
            return 0
        if not args.submit_after_review:
            print("STOP: submission approval flag is absent. No QPU job was submitted.")
            return 0

        if not both_requested_gates:
            print("STOP: RUN_REAL_QPU=YES and --confirm-real-qpu are both required.")
            return 0

        execution_plan = [
            {"start": start, "end_exclusive": end}
            for start, end in remaining_ranges
        ]
        print("\nFINAL IMMEDIATE SUBMISSION SUMMARY")
        print(f"BACKEND: {backend.name} ({backend_record['status_msg']}; {backend_record['pending_jobs']} pending)")
        print(f"LAYOUT: logical 4 -> physical {PREFERRED_LAYOUT}")
        print(f"CALIBRATION MEDIANS: 1Q={calibration_snapshot['selected_median_errors']['selected_median_1q_gate_error']:.8f}; 2Q={calibration_snapshot['selected_median_errors']['selected_median_2q_gate_error']:.8f}; readout={calibration_snapshot['selected_median_errors']['selected_median_readout_error']:.8f}")
        print(f"SHOTS: {args.shots}")
        print(f"CIRCUITS REMAINING: {sum(end - start for start, end in remaining_ranges)}")
        print(f"JOBS REMAINING: {len(execution_plan)}; ranges (zero-based, end-exclusive): {execution_plan}")
        print(f"CURRENT BATCH EXECUTIONS: {batch_execution_count:,}; per-job limit: {MAX_EXECUTIONS_PER_JOB:,}")
        print(f"ESTIMATED QPU EXECUTION: {estimated_qpu_seconds:.2f} seconds")
        print(f"REMAINING ALLOCATION: {float(remaining_seconds):.0f} seconds; {remaining_ru if remaining_ru is not None else 'unknown'} RUs")
        print("FROZEN MODEL: unchanged Stage 7 config; 24 train / 80 validation / 98 test recordings; NASA excluded")
        print("No job has been submitted yet. Type exactly YES to submit these jobs, or anything else to stop.")
        if input("Submit approved QPU workload? ").strip() != "YES":
            print("STOP: final confirmation was not YES. No IBM job was submitted.")
            return 0

        # Repeat live guards after the user's confirmation and immediately before job 1.
        latest_status = backend.status()
        latest_properties = backend.properties()
        latest_record = {
            "operational": bool(latest_status.operational),
            "status_msg": str(latest_status.status_msg),
            "pending_jobs": int(latest_status.pending_jobs),
        }
        latest_snapshot = _calibration_snapshot(
            backend, latest_properties, compiled, PREFERRED_LAYOUT
        )
        _check_calibration_drift(
            latest_snapshot,
            {"selected_median_errors": APPROVED_CALIBRATION_REFERENCE},
        )
        latest_usage = service.usage()
        latest_remaining = latest_usage.get("usage_remaining_seconds")
        latest_batch_seconds = _estimated_qpu_seconds(
            backend, latest_properties, compiled, batch_circuit_count, args.shots
        )
        if (
            latest_record["status_msg"].lower() != "active"
            or not latest_record["operational"]
            or latest_remaining is None
            or latest_batch_seconds is None
            or float(latest_remaining) < latest_batch_seconds * 1.25
            or latest_usage.get("usage_limit_reached")
        ):
            raise RuntimeError("Immediate pre-submit backend/allocation guard failed; no job was submitted")
        print(
            f"Immediate recheck passed: {latest_record['status_msg']}, "
            f"{latest_record['pending_jobs']} pending; no material calibration drift."
        )

        qpu_result = _submit_batches(
            backend,
            latest_properties,
            data,
            PREFERRED_LAYOUT,
            args.shots,
        )
        if qpu_result.get("status") == "batch_completed":
            usage_after = service.usage()
            remaining_after = usage_after.get("usage_remaining_seconds")
            ru_remaining_after = None
            if usage_after.get("usage_limit_ru") is not None and usage_after.get("usage_consumed_ru") is not None:
                ru_remaining_after = float(usage_after["usage_limit_ru"]) - float(usage_after["usage_consumed_ru"])
            consumed_before = usage.get("usage_consumed_seconds")
            consumed_after = usage_after.get("usage_consumed_seconds")
            if consumed_before is None and usage.get("usage_limit_seconds") is not None and remaining_seconds is not None:
                consumed_before = float(usage["usage_limit_seconds"]) - float(remaining_seconds)
            if consumed_after is None and usage_after.get("usage_limit_seconds") is not None and remaining_after is not None:
                consumed_after = float(usage_after["usage_limit_seconds"]) - float(remaining_after)
            batch_usage_seconds = (
                float(consumed_after) - float(consumed_before)
                if consumed_before is not None and consumed_after is not None
                else None
            )
            consumed_ru_before = usage.get("usage_consumed_ru")
            consumed_ru_after = usage_after.get("usage_consumed_ru")
            batch_usage_ru = (
                float(consumed_ru_after) - float(consumed_ru_before)
                if consumed_ru_before is not None and consumed_ru_after is not None
                else None
            )

            post_properties = backend.properties()
            post_snapshot = _calibration_snapshot(
                backend, post_properties, compiled, PREFERRED_LAYOUT
            )
            post_calibration_pass = True
            post_calibration_failure = None
            try:
                _check_calibration_drift(
                    post_snapshot,
                    {"selected_median_errors": APPROVED_CALIBRATION_REFERENCE},
                )
            except RuntimeError as exc:
                post_calibration_pass = False
                post_calibration_failure = str(exc)

            pairs, _ = _kernel_pairs(data)
            _, covered_after = _validated_resume_state(pairs, PREFERRED_LAYOUT, args.shots)
            next_range = (
                (covered_after, min(covered_after + CIRCUITS_PER_JOB, len(pairs)))
                if covered_after < len(pairs)
                else None
            )
            next_batch_seconds = (
                _estimated_qpu_seconds(
                    backend,
                    post_properties,
                    compiled,
                    next_range[1] - next_range[0],
                    args.shots,
                )
                if next_range is not None
                else None
            )
            batch4_safe = (
                next_range is not None
                and post_calibration_pass
                and backend_record["operational"]
                and str(backend.status().status_msg).strip().lower() == "active"
                and remaining_after is not None
                and next_batch_seconds is not None
                and float(remaining_after) >= next_batch_seconds * 1.25
                and not usage_after.get("usage_limit_reached")
            )
            print("\nBATCH COMPLETED; STOPPING BEFORE NEXT BATCH")
            print(f"JOB ID: {qpu_result['job_id']}")
            print(
                f"COMPLETED RANGE: [{qpu_result['circuit_index_start']}, "
                f"{qpu_result['circuit_index_end_exclusive'] - 1}] "
                f"({qpu_result['circuit_count']} circuits; {qpu_result['execution_count']:,} executions)"
            )
            print(
                f"ACTUAL BATCH USAGE: {batch_usage_seconds if batch_usage_seconds is not None else 'unknown'} seconds; "
                f"{batch_usage_ru if batch_usage_ru is not None else 'unknown'} RUs"
            )
            print(
                f"CUMULATIVE USAGE: {consumed_after if consumed_after is not None else 'unknown'} seconds; "
                f"{consumed_ru_after if consumed_ru_after is not None else 'unknown'} RUs"
            )
            print(
                f"REMAINING ALLOCATION: {remaining_after if remaining_after is not None else 'unknown'} seconds; "
                f"{ru_remaining_after if ru_remaining_after is not None else 'unknown'} RUs"
            )
            print(
                f"FRESH CALIBRATION AFTER BATCH: {'PASS' if post_calibration_pass else 'FAIL'}; "
                f"{_calibration_comparison(post_snapshot)}"
            )
            if post_calibration_failure:
                print(f"CALIBRATION DETAIL: {post_calibration_failure}")
            if next_range is not None:
                print(
                    f"BATCH 4 RANGE: [{next_range[0]}, {next_range[1] - 1}]; "
                    f"estimated {next_batch_seconds if next_batch_seconds is not None else 'unknown'} seconds"
                )
            print(f"BATCH 4 SAFE UNDER CURRENT CHECKS: {'YES' if batch4_safe else 'NO'}")
            print("STOP: Batch 4 was not submitted. A separate explicit approval is required.")
            return 0

        qpu_row = qpu_result
        _write_comparison(baselines, preview_row, qpu_row)
        print("QPU RESULT")
        print(json.dumps(qpu_row, indent=2))
        _print_performance_deltas(baselines, preview_row, qpu_row)
        return 0
    except QPUJobFailure as exc:
        print(f"Stage 7 submission stopped: {exc}", file=sys.stderr)
        return 4
    except Exception as exc:
        # Runtime exceptions may include request context; do not render their text.
        print(f"Stage 7 stopped: {type(exc).__name__}. No automatic retry was attempted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())