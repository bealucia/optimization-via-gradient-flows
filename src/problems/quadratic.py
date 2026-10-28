"""Familia quadratica SPD utilizada no Experimento 1."""

import numpy as np


def quadratic_matrix(kappa: float = 100.0, theta: float = np.pi / 4) -> np.ndarray:
    """Retorna Q diag(1, kappa) Q^T, com Q uma rotacao."""
    if not np.isfinite(kappa) or kappa < 1:
        raise ValueError("kappa deve ser finito e >= 1")
    if not np.isfinite(theta):
        raise ValueError("theta deve ser finito")
    c, s = np.cos(theta), np.sin(theta)
    Q = np.array([[c, -s], [s, c]])
    return Q @ np.diag([1.0, float(kappa)]) @ Q.T


def quadratic_value(x: np.ndarray, A: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(0.5 * x @ A @ x)


def quadratic_gradient(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    return A @ np.asarray(x, dtype=float)


def quadratic_hessian(x: np.ndarray, A: np.ndarray) -> np.ndarray:
    return A


def exact_flow(A: np.ndarray, x0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Trajetoria exata x(t)=exp(-At)x0, por decomposicao ortogonal."""
    eigvals, V = np.linalg.eigh(A)
    times = np.atleast_1d(np.asarray(times, dtype=float))
    coefs = V.T @ np.asarray(x0, dtype=float)
    return (np.exp(-np.outer(times, eigvals)) * coefs) @ V.T
