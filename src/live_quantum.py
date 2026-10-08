"""Closed-form statevectors for the trainable feature map, for live inference.

quantum_kernel_simulator._make_trainable_map applies H to every qubit, then
P(theta_i * x_i) on each qubit, then CX(i, i+1) P(theta_i theta_{i+1} x_i x_{i+1}) CX(i, i+1)
along the chain. Every gate after the Hadamards is diagonal (a CX-P-CX block adds
its phase when the two bits differ), so each basis state |b> has amplitude

    2^(-n/2) * exp(i * [sum_i theta_i x_i b_i + sum_i theta_i theta_{i+1} x_i x_{i+1} (b_i XOR b_{i+1})])

with qubit i as bit i of the basis index (Qiskit's little-endian order). This
module evaluates that formula with numpy, so the web app needs no Qiskit at
request time. build_live_detector.py checks it against Qiskit's Statevector.
"""

from __future__ import annotations

import numpy as np


def basis_bits(qubits: int) -> np.ndarray:
    indices = np.arange(2**qubits)
    return ((indices[:, None] >> np.arange(qubits)[None, :]) & 1).astype(float)


def statevectors(theta: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Statevectors of the trained feature map for each row of points, shape (rows, 2**n)."""
    theta = np.asarray(theta, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    qubits = len(theta)
    bits = basis_bits(qubits)
    single = (points * theta) @ bits.T
    pair_strength = theta[:-1] * theta[1:] * points[:, :-1] * points[:, 1:]
    differs = np.abs(bits[:, :-1] - bits[:, 1:])
    phases = single + pair_strength @ differs.T
    return np.exp(1j * phases) / np.sqrt(2**qubits)


def fidelity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left.conj() @ right.T) ** 2
