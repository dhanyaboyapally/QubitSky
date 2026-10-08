"""Reconstruct frozen Stage 7 kernel matrices from completed local job caches.

This utility is offline-only. It does not access IBM credentials or Runtime and
does not submit, retrieve, retry, or otherwise interact with hardware jobs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config import RESULTS_DIR
from run_ibm_qpu import JOB_CACHE_DIR, JOBS_PATH, _kernel_pairs, _load_frozen_data


EXPECTED_CIRCUITS = 4548
EXPECTED_SHOTS = 1024
EXPECTED_BACKEND = "ibm_pittsburgh"
EXPECTED_LAYOUT = [87, 97, 107, 108]
EXPECTED_SHAPES = {
    "training": (24, 24),
    "validation": (80, 24),
    "test": (98, 24),
}
RECONSTRUCTED_PATH = RESULTS_DIR / "quantum" / "qpu_reconstructed_kernels.npz"


def reconstruct_completed_caches() -> Path:
    ledger = json.loads(JOBS_PATH.read_text(encoding="utf-8"))
    job_records = ledger.get("jobs", [])
    if ledger.get("backend") != EXPECTED_BACKEND:
        raise ValueError("Job ledger backend differs from the frozen Stage 7 backend")
    if ledger.get("physical_layout") != EXPECTED_LAYOUT:
        raise ValueError("Job ledger layout differs from the frozen Stage 7 layout")
    if ledger.get("shots") != EXPECTED_SHOTS or ledger.get("total_circuits") != EXPECTED_CIRCUITS:
        raise ValueError("Job ledger shots or frozen circuit count do not match Stage 7")

    ordered_pairs, _ = _kernel_pairs(_load_frozen_data()[1])
    if len(ordered_pairs) != EXPECTED_CIRCUITS:
        raise ValueError("Frozen ordered kernel-pair list no longer has 4,548 entries")

    matrices = {
        "training": np.eye(EXPECTED_SHAPES["training"][0], dtype=np.float64),
        "validation": np.zeros(EXPECTED_SHAPES["validation"], dtype=np.float64),
        "test": np.zeros(EXPECTED_SHAPES["test"], dtype=np.float64),
    }
    covered: set[int] = set()
    incomplete: list[str] = []

    for job in job_records:
        start = int(job["circuit_index_start"])
        end = int(job["circuit_index_end_exclusive"])
        if job.get("backend") != EXPECTED_BACKEND or job.get("physical_layout") != EXPECTED_LAYOUT:
            raise ValueError(f"Job {job.get('job_id')} backend/layout does not match the frozen config")
        if job.get("shots") != EXPECTED_SHOTS:
            raise ValueError(f"Job {job.get('job_id')} shot count differs from the frozen config")
        if job.get("status") != "completed" or not job.get("result_cache"):
            incomplete.append(f"{job.get('job_id')} [{start}:{end}) status={job.get('status')}")
            continue

        cache_path = Path(job["result_cache"])
        if not cache_path.is_file():
            incomplete.append(f"{job.get('job_id')} [{start}:{end}) cache missing")
            continue
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        indices = cache.get("circuit_indices", [])
        cache_pairs = cache.get("pairs", [])
        counts = cache.get("counts", [])
        expected_indices = list(range(start, end))
        if indices != expected_indices or len(cache_pairs) != len(indices) or len(counts) != len(indices):
            raise ValueError(f"Job {job.get('job_id')} cache index/count lengths are inconsistent")
        if cache.get("backend") != EXPECTED_BACKEND or cache.get("physical_layout") != EXPECTED_LAYOUT:
            raise ValueError(f"Job {job.get('job_id')} cache backend/layout mismatch")
        if cache.get("shots") != EXPECTED_SHOTS:
            raise ValueError(f"Job {job.get('job_id')} cache shot count mismatch")
        if covered.intersection(indices):
            raise ValueError(f"Duplicate circuit indices found in cache for job {job.get('job_id')}")

        for circuit_index, pair_record, shot_counts in zip(indices, cache_pairs, counts):
            split, left_index, right_index, _, _ = ordered_pairs[circuit_index]
            expected_pair = {
                "split": split,
                "left_index": left_index,
                "right_index": right_index,
            }
            if pair_record != expected_pair:
                raise ValueError(f"Circuit ordering mismatch at frozen index {circuit_index}")
            total_shots = sum(int(value) for value in shot_counts.values())
            if total_shots != EXPECTED_SHOTS:
                raise ValueError(f"Circuit {circuit_index} returned {total_shots} shots, expected {EXPECTED_SHOTS}")
            probability_zero = int(shot_counts.get("0000", 0)) / EXPECTED_SHOTS
            if split == "training":
                matrices[split][left_index, right_index] = probability_zero
                matrices[split][right_index, left_index] = probability_zero
            else:
                matrices[split][left_index, right_index] = probability_zero
        covered.update(indices)

    missing = sorted(set(range(EXPECTED_CIRCUITS)) - covered)
    if incomplete or missing:
        raise RuntimeError(
            "Full reconstruction is not ready; no output was written. "
            f"Incomplete jobs: {incomplete}; missing circuit indices: "
            f"{len(missing)} (first missing: {missing[:12]})."
        )
    for name, matrix in matrices.items():
        if matrix.shape != EXPECTED_SHAPES[name] or not np.isfinite(matrix).all():
            raise ValueError(f"Reconstructed {name} matrix has an invalid shape or non-finite values")
    if not np.allclose(matrices["training"], matrices["training"].T, rtol=0, atol=1e-12):
        raise ValueError("Reconstructed training Gram matrix is not symmetric")

    RECONSTRUCTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RECONSTRUCTED_PATH,
        training=matrices["training"],
        validation=matrices["validation"],
        test=matrices["test"],
    )
    return RECONSTRUCTED_PATH


if __name__ == "__main__":
    output = reconstruct_completed_caches()
    print(f"Reconstructed frozen QPU kernel matrices: {output}")