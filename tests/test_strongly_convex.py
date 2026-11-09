"""Derivadas, dissipação e contração do problema não quadrático do Módulo 2."""
import numpy as np
from scipy.integrate import solve_ivp

from src.problems.strongly_convex import StronglyConvex
from src.methods.euler import forward_euler, backward_euler


def test_derivatives_against_centered_finite_differences():
    p = StronglyConvex(17)
    x = np.array([0.7, -0.6])
    eps = 1e-6
    for i in range(2):
        direction = np.eye(2)[i]
        grad_num = (p.value(x+eps*direction)-p.value(x-eps*direction))/(2*eps)
        hess_col = (p.gradient(x+eps*direction)-p.gradient(x-eps*direction))/(2*eps)
        np.testing.assert_allclose(grad_num, p.gradient(x)[i], atol=1e-7)
        np.testing.assert_allclose(hess_col, p.hessian(x)[:,i], atol=1e-7)
    np.testing.assert_allclose(p.gradient(p.minimizer), 0, atol=1e-14)


def test_global_hessian_bounds_at_multiple_positions():
    p = StronglyConvex(17)
    for x in ([-10, 5], [1, 1], [0, 0], [7, -3]):
        eigenvalues = np.linalg.eigvalsh(p.hessian(x))
        assert eigenvalues.min() >= p.m-1e-12
        assert eigenvalues.max() <= p.L+1e-12


def test_forward_descent_for_sufficient_step():
    p = StronglyConvex(100)
    X, info = forward_euler(p.gradient, np.array([1.,1.]), 1.8/p.L, 50)
    values = np.array([p.value(x) for x in X])
    assert info['status'] == 'Fixed Steps'
    assert np.max(np.diff(values)) <= 1e-12


def test_backward_descent_and_implicit_residual():
    p = StronglyConvex(100)
    x0 = np.array([-2., .5])
    h = 4.0
    X, info = backward_euler(p.gradient, p.hessian, x0, h, 20)
    values = np.array([p.value(x) for x in X])
    assert X.shape == (21, 2)
    assert np.max(np.diff(values)) <= 1e-11
    assert info['max_implicit_residual'] <= 1e-9
    for old, new in zip(X[:-1], X[1:]):
        assert np.linalg.norm(new-old+h*p.gradient(new)) <= 1e-9


def test_backward_distance_respects_strong_convexity_bound():
    p = StronglyConvex(20)
    x0 = np.array([1.,1.]); h = 1.
    X, _ = backward_euler(p.gradient, p.hessian, x0, h, 12)
    theoretical = np.linalg.norm(x0) / (1+h*p.m)**np.arange(len(X))
    assert np.all(np.linalg.norm(X, axis=1) <= theoretical+2e-9)


def test_temporal_error_diminishes_under_refinement():
    p = StronglyConvex(100)
    x0 = np.array([1.,1.]); T = .05
    sol = solve_ivp(lambda t,x: -p.gradient(x), (0,T), x0,
                    method='DOP853', rtol=2e-13, atol=2e-15)
    assert sol.success
    reference = sol.y[:,-1]
    for integrator in (
        lambda n: forward_euler(p.gradient, x0, T/n, n)[0],
        lambda n: backward_euler(p.gradient, p.hessian, x0, T/n, n)[0],
    ):
        errors = [np.linalg.norm(integrator(n)[-1]-reference) for n in (20,80,320)]
        assert errors[0] > errors[1] > errors[2] > 0
