"""Sela quadrática não degenerada (sem minimizador global).

f(x,y)=(x^2-y^2)/2. A origem é um ponto crítico de sela.
As duas soluções de EDO são exatas e definidas para todo t real.
"""
import numpy as np


H = np.diag([1.0, -1.0])


def _state(x):
    x = np.asarray(x, dtype=float)
    if x.shape != (2,) or not np.all(np.isfinite(x)):
        raise ValueError('Estado da sela deve ser vetor finito de dimensão 2')
    return x


def value(x):
    v = _state(x)
    return float(0.5 * (v[0]**2 - v[1]**2))


def gradient(x):
    return H @ _state(x)


def hessian(x):
    _state(x)
    return H.copy()


def exact_gradient_flow(x0, times):
    """Solução de x'=-grad f: (x0 e^-t, y0 e^t)."""
    x0 = _state(x0)
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if t.ndim != 1 or not np.isfinite(t).all():
        raise ValueError('Tempos inválidos')
    return np.column_stack((x0[0]*np.exp(-t), x0[1]*np.exp(t)))


def exact_newton_flow(x0, times):
    """Solução de x'=-H^-1 grad f=-x (neste exemplo)."""
    x0 = _state(x0)
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if t.ndim != 1 or not np.isfinite(t).all():
        raise ValueError('Tempos inválidos')
    return np.exp(-t)[:, None] * x0[None, :]
