"""Solvers implícitos: solução algébrica, reuso e resíduo verificado."""
import numpy as np
import pytest

from src.problems.least_squares import make_problem
from src.methods.linear_implicit import prepare_implicit, iterate_until_tolerance


@pytest.mark.parametrize('backend', ('Cholesky','LU','QR','SVD'))
def test_backend_matches_reference_step_without_truncation(backend):
    p = make_problem(12, 80, seed=42)
    h = 1.
    B = np.eye(12)+h*(p.A.T@p.A)
    expected = np.linalg.solve(B,p.x0+h*p.A.T@p.b)
    solver = prepare_implicit(p.A,p.b,h,backend)
    actual = solver.step(p.x0)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert np.linalg.norm(actual-p.x0+h*p.gradient(actual)) < 1e-9


@pytest.mark.parametrize('backend', ('Cholesky','LU','QR','SVD'))
def test_reused_solver_satisfies_implicit_equation_each_step(backend):
    p = make_problem(16, 150, seed=3)
    h = 1.
    solver = prepare_implicit(p.A,p.b,h,backend)
    x = p.x0.copy()
    for _ in range(5):
        nxt = solver.step(x)
        assert np.linalg.norm(nxt-x+h*p.gradient(nxt)) < 1e-9
        x = nxt


def test_implicit_optimization_convergence_and_count():
    p = make_problem(8, 100, seed=2)
    solver = prepare_implicit(p.A,p.b,1.,'Cholesky')
    r = iterate_until_tolerance(p.gradient, lambda x,g: solver.step(x), p.x0,
                                max_steps=1000, tol_grad=1e-6)
    assert r['status']=='Converged'
    assert r['iterations']==20
    assert r['grad_final'] <= 1e-6
    assert r['n_grad']==r['iterations']+1


def test_invalid_step_and_backend_rejected():
    p = make_problem(8, 10, seed=2)
    with pytest.raises(ValueError):
        prepare_implicit(p.A,p.b,0,'Cholesky')
    with pytest.raises(ValueError):
        prepare_implicit(p.A,p.b,1,'inexistente')
