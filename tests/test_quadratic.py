"""Propriedades espectrais, consistência do modelo e fluxo quadrático exato."""
import numpy as np
import pytest

from src.problems.quadratic import (
    quadratic_matrix, quadratic_gradient, quadratic_hessian,
    quadratic_value, exact_flow,
)


@pytest.mark.parametrize("kappa", [1, 10, 100, 1000])
def test_spectrum_and_positive_definiteness(kappa):
    A = quadratic_matrix(kappa)
    np.testing.assert_allclose(A, A.T, atol=1e-13)
    np.testing.assert_allclose(np.linalg.eigvalsh(A), [1, kappa], rtol=1e-12, atol=1e-12)
    assert np.linalg.eigvalsh(A).min() > 0


def test_gradient_and_hessian_of_quadratic():
    A = quadratic_matrix(100)
    x = np.array([0.7, -0.6])
    h = 1e-6
    for j in range(2):
        e = np.eye(2)[j]
        numerical = (quadratic_value(x + h * e, A) - quadratic_value(x - h * e, A)) / (2 * h)
        np.testing.assert_allclose(numerical, quadratic_gradient(x, A)[j], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(quadratic_hessian(x, A), A)


def test_exact_flow_initial_condition_and_modal_evolution():
    A = quadratic_matrix(100)
    eigenvalues, V = np.linalg.eigh(A)
    x0 = np.array([1.0, -0.5])  # vetor genérico: ambas as direções podem estar presentes
    times = np.array([0.0, 0.02, 0.1])
    X = exact_flow(A, x0, times)
    np.testing.assert_allclose(X[0], x0, atol=1e-12)
    expected_modal = np.exp(-times[:, None] * eigenvalues[None, :]) * (V.T @ x0)
    np.testing.assert_allclose(X @ V, expected_modal, rtol=1e-12, atol=1e-12)


def test_invalid_condition_number_is_rejected():
    for kappa in (0, -1, float('nan')):
        with pytest.raises(ValueError):
            quadratic_matrix(kappa)
