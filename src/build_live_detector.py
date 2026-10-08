"""Package the frozen 4-qubit trainable QSVC for live detection in the web app.

Writes models/quantum/live_detector.json with the trained kernel parameters, the
24 scaled training points, the training kernel measured on IBM hardware and the
hardware kernel response, plus held-out test metrics for every sensitivity mode
(one-second windows, share-of-windows vote). Offline only; never contacts IBM.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.svm import SVC

from config import FEATURES_DIR, MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from live_quantum import statevectors as closed_form_statevectors
from quantum_kernel_simulator import _load_stage_assets, _make_trainable_map

FROZEN_CONFIG = MODELS_DIR / "quantum" / "stage7_frozen_config.json"
ASSEMBLED_KERNELS = RESULTS_DIR / "quantum" / "qpu_job_cache" / "assembled_kernel_metrics.json"
HARDWARE_ANALYSIS = RESULTS_DIR / "quantum" / "hardware_analysis.json"
OUTPUT = MODELS_DIR / "quantum" / "live_detector.json"
SENSITIVITY = {"High alert": 0.25, "Balanced": 0.50, "Low false alarms": 0.75}
SVC_C = 0.1


def statevectors(theta: np.ndarray, points: np.ndarray) -> np.ndarray:
    circuit, theta_parameters = _make_trainable_map(len(theta))
    bound = circuit.assign_parameters(dict(zip(theta_parameters, theta)))
    data_parameters = sorted(bound.parameters, key=lambda parameter: parameter.name)
    return np.array([Statevector(bound.assign_parameters(dict(zip(data_parameters, row)))).data for row in points])


def fidelity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left.conj() @ right.T) ** 2


def mode_metrics(model: SVC, window_kernel: np.ndarray, windows: pd.DataFrame, recordings: pd.DataFrame) -> dict:
    votes = windows.assign(vote=model.predict(window_kernel))
    share = votes.groupby("recording_id")["vote"].mean().reindex(recordings["recording_id"]).to_numpy()
    truth = recordings["binary_label"].to_numpy(dtype=int)
    out = {}
    for name, threshold in SENSITIVITY.items():
        prediction = (share >= threshold).astype(int)
        out[name] = {
            "f1": float(f1_score(truth, prediction, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "drone_recall": float(recall_score(truth, prediction, zero_division=0)),
            "non_drone_recall": float(recall_score(truth, prediction, pos_label=0, zero_division=0)),
        }
    return out


def build() -> Path:
    config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))
    theta = np.asarray(config["trained_kernel_parameters"], dtype=float)
    names = list(config["feature_names"])
    data = _load_stage_assets(len(names), 24, RANDOM_SEED)
    train, test = data["train"], data["tests_by_snr"]["clean"]
    if train["recording_id"].astype(str).tolist() != config["training_recording_ids"]:
        raise ValueError("Training recordings differ from the frozen hardware configuration")
    train_x = train[names].to_numpy(dtype=float)
    train_y = train["binary_label"].to_numpy(dtype=int)

    windows = pd.read_csv(FEATURES_DIR / f"test_features_{len(names)}.csv")
    windows = windows[windows["snr_db"].astype(str).eq("clean") & windows["recording_id"].isin(test["recording_id"])]

    train_states = statevectors(theta, train_x)
    window_states = statevectors(theta, windows[names].to_numpy(dtype=float))
    # The web app uses the numpy closed form; it must match Qiskit's simulation.
    closed_form_error = float(max(
        np.max(np.abs(closed_form_statevectors(theta, train_x) - train_states)),
        np.max(np.abs(closed_form_statevectors(theta, windows[names].to_numpy(dtype=float)) - window_states)),
    ))
    if closed_form_error > 1e-9:
        raise ValueError(f"Closed-form statevectors differ from Qiskit by {closed_form_error:.2e}")
    ideal_train = fidelity(train_states, train_states)
    ideal_windows = fidelity(window_states, train_states)
    ideal_model = SVC(kernel="precomputed", C=SVC_C, class_weight="balanced").fit(ideal_train, train_y)

    result = {
        "model": "trainable 4-qubit fidelity kernel + SVM (QSVC)",
        "feature_names": names,
        "theta": theta.tolist(),
        "svc_c": SVC_C,
        "scaler_path": f"models/scaler_{len(names)}.pkl",
        "train_points": train_x.round(8).tolist(),
        "train_labels": train_y.tolist(),
        "sensitivity_thresholds": SENSITIVITY,
        "test_recordings": int(len(test)),
        "test_windows": int(len(windows)),
        "closed_form_max_error_vs_qiskit": closed_form_error,
        "metrics": {"ideal": mode_metrics(ideal_model, ideal_windows, windows, test)},
    }

    if ASSEMBLED_KERNELS.is_file() and HARDWARE_ANALYSIS.is_file():
        assembled = json.loads(ASSEMBLED_KERNELS.read_text(encoding="utf-8"))
        analysis = json.loads(HARDWARE_ANALYSIS.read_text(encoding="utf-8"))
        hardware_train = np.asarray(assembled["training_kernel"], dtype=float)
        response = analysis["kernel_agreement"]["test"]
        hardware_model = SVC(kernel="precomputed", C=SVC_C, class_weight="balanced").fit(hardware_train, train_y)
        emulated = np.clip(response["slope"] * ideal_windows + response["intercept"], 0.0, 1.0)
        result["hardware"] = {
            "backend": analysis["backend"],
            "physical_qubits": analysis["physical_qubits"],
            "shots": analysis["shots"],
            "train_kernel": hardware_train.round(6).tolist(),
            "response_slope": response["slope"],
            "response_intercept": response["intercept"],
            "measured_recording_f1": analysis["metrics"]["hardware"]["f1"],
        }
        result["metrics"]["hardware"] = mode_metrics(hardware_model, emulated, windows, test)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return OUTPUT


def main() -> None:
    path = build()
    summary = json.loads(path.read_text(encoding="utf-8"))
    print(f"Wrote {path}")
    for mode, metrics in summary["metrics"].items():
        balanced = metrics["Balanced"]
        print(f"{mode:9s} Balanced: F1 {balanced['f1']:.3f}, balanced accuracy {balanced['balanced_accuracy']:.3f}")


if __name__ == "__main__":
    main()
