"""Newton amortecido externo: Euler forward do fluxo de Newton.

Diferente do Newton interno usado para resolver Euler backward em euler.py.
Não usa inversa explícita da Hessiana e não esconde singularidades.
"""
import numpy as np


def newton_damped(grad_f, hess_f, x0, h, n_steps):
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError('x0 inválido')
    if not (np.isfinite(h) and h > 0):
        raise ValueError('Passo h deve ser finito e positivo')
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError('n_steps deve ser inteiro não negativo')
    traj = [x.copy()]
    for _ in range(n_steps):
        g = np.asarray(grad_f(x), dtype=float)
        H = np.asarray(hess_f(x), dtype=float)
        if g.shape != x.shape or H.shape != (x.size, x.size):
            raise ValueError('Dimensões inválidas de gradiente ou Hessiana')
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(H))):
            raise FloatingPointError('Gradiente ou Hessiana inválidos')
        # Resolve H d = -g; se H singular, a exceção é propagada ao chamador.
        direction = np.linalg.solve(H, -g)
        x = x + h*direction
        if not np.all(np.isfinite(x)):
            raise FloatingPointError('Newton amortecido produziu estado não finito')
        traj.append(x.copy())
    return np.asarray(traj)
