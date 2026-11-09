"""Mínimos quadrados: condicionamento, inconsistência e otimalidade."""
import numpy as np

from src.problems.least_squares import make_problem


def test_controlled_hessian_condition_number():
    for kappa in (1, 10, 100, 1000):
        p = make_problem(12, kappa, seed=123)
        np.testing.assert_allclose(np.linalg.cond(p.A.T @ p.A), kappa, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(np.linalg.svd(p.A, compute_uv=False)[-1], 1, rtol=1e-10)


def test_inconsistent_data_and_known_minimizer():
    p = make_problem(12, 100, seed=123)
    residual = p.A @ p.x_star - p.b
    assert np.linalg.norm(residual) > .01  # o sistema Ax=b não tem solução exata
    np.testing.assert_allclose(p.A.T @ residual, 0, atol=1e-11)
    assert np.linalg.norm(p.gradient(p.x_star)) < 1e-11
    np.testing.assert_allclose(np.linalg.norm(p.x0-p.x_star), np.sqrt(2), atol=1e-12)
    np.testing.assert_allclose(p.hessian(p.x_star), p.A.T @ p.A)


def test_least_squares_gradient_by_finite_difference():
    p = make_problem(10, 50, seed=7)
    x = p.x0
    step = 1e-6
    for j in range(x.size):
        direction = np.eye(x.size)[j]
        finite_difference = (p.value(x+step*direction)-p.value(x-step*direction))/(2*step)
        np.testing.assert_allclose(finite_difference, p.gradient(x)[j], rtol=1e-7, atol=1e-7)
