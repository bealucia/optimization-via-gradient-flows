"""Família quadrática SPD: mantém interface exigida pelo Experimento 1."""
import numpy as np


def quadratic_matrix(kappa=100.0, theta=np.pi / 4):
    if not np.isfinite(kappa) or kappa < 1:
        raise ValueError("kappa deve ser finito e >=1")
    c, s = np.cos(theta), np.sin(theta)
    Q = np.array([[c, -s], [s, c]])
    return Q @ np.diag([1.0, float(kappa)]) @ Q.T


def quadratic_value(x, A):
    return float(0.5 * np.asarray(x) @ A @ np.asarray(x))


def quadratic_gradient(x, A):
    return A @ np.asarray(x)


def quadratic_hessian(x, A):
    return A


def exact_flow(A, x0, times):
    vals, vecs = np.linalg.eigh(A)
    t = np.atleast_1d(np.asarray(times, dtype=float))
    modal0 = vecs.T @ np.asarray(x0, dtype=float)
    return (np.exp(-np.outer(t, vals)) * modal0) @ vecs.T
