"""Stage 7b: compare the measured QPU kernel with the ideal kernel (offline only).

Reads the assembled hardware kernels written by run_ibm_qpu.py, recomputes the
ideal statevector kernel for the same frozen model and recordings, and writes
results/quantum/hardware_analysis.json for the RESEARCH page. It never contacts
IBM Quantum.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC

import run_ibm_qpu as qpu
from classical_utils import calculate_metrics
from config import RESULTS_DIR

QUANTUM_DIR = RESULTS_DIR / "quantum"
ASSEMBLED_PATH = QUANTUM_DIR / "qpu_job_cache" / "assembled_kernel_metrics.json"
ARCHIVE_DIR = QUANTUM_DIR / "qpu_archive"
OUTPUT_PATH = QUANTUM_DIR / "hardware_analysis.json"
SCATTER_POINTS = 800


def _kernel_fit(hardware: np.ndarray, ideal: np.ndarray) -> dict[str, float]:
    slope, intercept = np.polyfit(ideal, hardware, 1)
    return {
        "correlation": float(np.corrcoef(hardware, ideal)[0, 1]),
        "mean_abs_diff": float(np.mean(np.abs(hardware - ideal))),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def _drift() -> dict | None:
    """Compare an archived pre-recalibration batch with its rerun, if both exist."""
    for archived in sorted(ARCHIVE_DIR.glob("*/*.json")):
        if archived.name == "qpu_jobs.json":
            continue
        old = json.loads(archived.read_text(encoding="utf-8"))
        ledger = json.loads(qpu.JOBS_PATH.read_text(encoding="utf-8"))
        first_job = ledger["jobs"][0]["job_id"]
        rerun_path = qpu.JOB_CACHE_DIR / f"{first_job}.json"
        if not rerun_path.is_file():
            return None
        new = json.loads(rerun_path.read_text(encoding="utf-8"))
        key = lambda pair: (pair["split"], pair["left_index"], pair["right_index"])
        if [key(p) for p in old["pairs"]] != [key(p) for p in new["pairs"]]:
            return None
        shots = int(ledger["shots"])
        old_p = np.array([c.get("0" * 4, 0) / shots for c in old["counts"]])
        new_p = np.array([c.get("0" * 4, 0) / shots for c in new["counts"]])
        return {
            "archived_run": archived.parent.name,
            "circuits": int(len(old_p)),
            "correlation": float(np.corrcoef(old_p, new_p)[0, 1]),
            "mean_abs_diff": float(np.mean(np.abs(old_p - new_p))),
        }
    return None


def analyze() -> Path:
    qpu_row = pd.read_csv(qpu.QPU_RESULTS_PATH).iloc[0]
    if str(qpu_row.get("status")) not in {"completed", "measured"}:
        raise RuntimeError("No completed QPU run to analyze")
    backend = str(qpu_row["backend_name"])
    qpu._apply_backend_profile(backend, include_validation_kernel=True)
    _, data = qpu._load_frozen_data()
    features = data["features"]
    train, test = data["train"], data["tests_by_snr"]["clean"]
    y_train = train.binary_label.to_numpy(dtype=int)
    y_test = test.binary_label.to_numpy(dtype=int)

    assembled = json.loads(ASSEMBLED_PATH.read_text(encoding="utf-8"))
    hw_train = np.asarray(assembled["training_kernel"], dtype=float)
    hw_test = np.asarray(assembled["test_kernel"], dtype=float)

    feature_map = qpu._frozen_feature_map(data)
    params = sorted(feature_map.parameters, key=lambda p: p.name)

    def states(frame: pd.DataFrame) -> np.ndarray:
        return np.array([
            Statevector(feature_map.assign_parameters(dict(zip(params, row)))).data
            for row in frame[features].to_numpy(dtype=float)
        ])

    s_train, s_test = states(train), states(test)
    id_train = np.abs(s_train.conj() @ s_train.T) ** 2
    id_test = np.abs(s_test.conj() @ s_train.T) ** 2

    upper = np.triu_indices(len(train), 1)
    predictions = {}
    metrics = {}
    for name, k_train, k_test in (("ideal", id_train, id_test), ("hardware", hw_train, hw_test)):
        model = SVC(kernel="precomputed", C=0.1, class_weight="balanced").fit(k_train, y_train)
        predictions[name] = model.predict(k_test)
        metrics[name] = calculate_metrics(y_test, predictions[name], model.decision_function(k_test))

    rng = np.random.default_rng(0)
    pick = rng.choice(id_test.size, size=min(SCATTER_POINTS, id_test.size), replace=False)
    result = {
        "backend": backend,
        "physical_qubits": json.loads(qpu_row["physical_qubits"]),
        "shots": int(qpu_row["shots"]),
        "circuits": int(qpu_row["circuit_count"]),
        "jobs": json.loads(qpu_row["job_ids"]),
        "kernel_agreement": {
            "training": _kernel_fit(hw_train[upper], id_train[upper]),
            "test": _kernel_fit(hw_test.ravel(), id_test.ravel()),
        },
        "metrics": {name: {k: v for k, v in m.items()} for name, m in metrics.items()},
        "predictions_changed": int((predictions["ideal"] != predictions["hardware"]).sum()),
        "test_recordings": int(len(y_test)),
        "test_kernel_sample": {
            "ideal": id_test.ravel()[pick].round(4).tolist(),
            "hardware": hw_test.ravel()[pick].round(4).tolist(),
        },
        "drift": _drift(),
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return OUTPUT_PATH


def main() -> None:
    path = analyze()
    summary = json.loads(path.read_text(encoding="utf-8"))
    print(f"Wrote {path}")
    print(
        f"{summary['backend']}: hardware F1 {summary['metrics']['hardware']['f1']:.3f} vs ideal "
        f"{summary['metrics']['ideal']['f1']:.3f}; {summary['predictions_changed']}/"
        f"{summary['test_recordings']} test predictions changed; test-kernel correlation "
        f"{summary['kernel_agreement']['test']['correlation']:.3f}"
    )


if __name__ == "__main__":
    main()
