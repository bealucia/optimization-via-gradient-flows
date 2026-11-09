"""Discretizações de Euler: identidades exatas, estabilidade e critérios de parada."""
import numpy as np
import pytest

from src.problems.quadratic import quadratic_matrix
from src.methods.euler import forward_euler, backward_euler


@pytest.fixture
def quadratic_case():
    A = quadratic_matrix(100)
    x0 = np.array([1.0, -0.5])
    return A, x0, lambda x: A @ x, lambda x: A


def test_forward_matches_discrete_matrix_power(quadratic_case):
    A, x0, grad, _ = quadratic_case
    h, n = 0.015, 20
    X, info = forward_euler(grad, x0, h, n)
    reference = np.linalg.matrix_power(np.eye(2) - h*A, n) @ x0
    np.testing.assert_allclose(X[-1], reference, atol=1e-12, rtol=1e-12)
    assert X.shape == (n+1, 2)
    assert info['iterations'] == n
    assert info['status'] == 'Fixed Steps'


def test_backward_matches_discrete_linear_system(quadratic_case):
    A, x0, grad, hess = quadratic_case
    h, n = 0.021, 20
    X, info = backward_euler(grad, hess, x0, h, n)
    B = np.eye(2) + h*A
    expected = x0.copy()
    for _ in range(n):
        expected = np.linalg.solve(B, expected)  # evita formar inversa explícita
    np.testing.assert_allclose(X[-1], expected, atol=1e-11, rtol=1e-11)
    assert X.shape == (n+1, 2)
    assert info['iterations'] == n
    assert info['max_implicit_residual'] < 1e-10
    for previous, nxt in zip(X[:-1], X[1:]):
        np.testing.assert_allclose(nxt - previous + h*(A @ nxt), 0, atol=1e-10)


@pytest.mark.parametrize('h,expected_factor', [(0.005, 0.5), (0.015,-0.5), (0.020,-1.0), (0.021,-1.1)])
def test_fast_mode_forward_amplification(h, expected_factor):
    A = quadratic_matrix(100)
    _, V = np.linalg.eigh(A)
    fast = V[:, -1]
    X, _ = forward_euler(lambda x:A@x, fast, h, 5)
    np.testing.assert_allclose(X @ fast, expected_factor**np.arange(6), atol=1e-11)


def test_implicit_damps_fast_mode_with_large_step():
    A = quadratic_matrix(100)
    _, V = np.linalg.eigh(A)
    fast = V[:, -1]
    X, _ = backward_euler(lambda x:A@x, lambda x:A, fast, h=1.0, n_steps=3)
    np.testing.assert_allclose(X @ fast, (1/101)**np.arange(4), atol=1e-12)


def test_gradient_tolerance_and_iteration_count(quadratic_case):
    A, x0, grad, hess = quadratic_case
    for run in (lambda: forward_euler(grad, x0, 0.015, 2000, tol_grad=1e-6),
                lambda: backward_euler(grad, hess, x0, 1.0, 100, tol_grad=1e-6)):
        X, info = run()
        assert info['status'] == 'Converged'
        assert info['iterations'] == len(X)-1
        assert np.linalg.norm(A @ X[-1]) <= 1e-6


def test_invalid_step_rejected(quadratic_case):
    _, x0, grad, hess = quadratic_case
    with pytest.raises(ValueError):
        forward_euler(grad, x0, -0.1, 10)
    with pytest.raises(ValueError):
        backward_euler(grad, hess, x0, 0.0, 10)
