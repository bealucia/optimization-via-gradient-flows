"""Solvers para o passo linear de Euler backward em mínimos quadrados.

Quatro formas ALGEBRICAMENTE equivalentes em aritmética exata:
- Cholesky/LU: B=(I+h A^T A), B y=x+h A^T b;
- QR: mínimos quadrados aumentados [sqrt(h)A; I]y≈[sqrt(h)b; x];
- SVD: fórmula espectral usando uma SVD de A, sem truncamento.

As fatorações são feitas UMA VEZ para h fixo, e reutilizadas por iteração.
"""
from time import perf_counter
from dataclasses import dataclass
import numpy as np
from scipy import linalg


@dataclass
class PreparedImplicit:
    name: str
    step: object
    factor_seconds: float
    monitoring_gradient: object | None = None


def prepare_implicit(A, b, h, backend):
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or b.shape != (A.shape[0],):
        raise ValueError('A deve ser matriz e b vetor com dimensões compatíveis')
    if A.shape[0] < 1 or A.shape[1] < 1 or not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError('A e b devem ter dimensão positiva e valores finitos')
    if h <= 0 or not np.isfinite(h):
        raise ValueError('Exigir h positivo')
    m, n = A.shape
    if backend == 'SVD' and m < n:
        raise ValueError('A fórmula SVD implementada requer m >= n')
    tic = perf_counter()
    monitoring_gradient = None
    if backend == 'Cholesky':
        G = A.T @ A
        c = A.T @ b
        B = np.eye(n) + h*G
        const = h*c
        monitoring_gradient = lambda x: G @ x - c
        factor = linalg.cho_factor(B, lower=True, check_finite=False)
        def step(x):
            return linalg.cho_solve(factor, x+const, check_finite=False)
    elif backend == 'LU':
        G = A.T @ A
        c = A.T @ b
        B = np.eye(n) + h*G
        const = h*c
        monitoring_gradient = lambda x: G @ x - c
        factor = linalg.lu_factor(B, check_finite=False)
        def step(x):
            return linalg.lu_solve(factor, x+const, check_finite=False)
    elif backend == 'QR':
        # Evita formar A^T A na fatoração; sistema aumentado sempre bem posto.
        C = np.vstack((np.sqrt(h)*A, np.eye(n)))
        Q, R = linalg.qr(C, mode='economic', check_finite=False)
        constant = np.sqrt(h)*(Q[:m, :].T @ b)
        Qtail = Q[m:, :].T
        def step(x):
            rhs = constant + Qtail @ x
            return linalg.solve_triangular(R, rhs, check_finite=False)
    elif backend == 'SVD':
        # Sem pseudoinversa truncada nem regularização extra.
        U, singular, Vh = linalg.svd(A, full_matrices=False, check_finite=False)
        attenuation = 1/(1+h*singular**2)
        constant = (h*singular*attenuation)*(U.T @ b)
        def step(x):
            return Vh.T @ (attenuation*(Vh @ x)+constant)
    else:
        raise ValueError(f'backend inválido: {backend}')
    return PreparedImplicit(backend, step, perf_counter()-tic, monitoring_gradient)


def iterate_until_tolerance(grad_f, step_f, x0, tol_grad=1e-6,
                            max_steps=100000):
    """Medição do laço com checagem de gradiente igual para EF e EB.

    Retorna tanto a duração do laço quanto a soma das durações dos passos.
    O custo de construir o solver NÃO entra aqui: deve ser somado à parte.
    """
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size < 1 or not np.all(np.isfinite(x)):
        raise ValueError('x0 deve ser vetor finito')
    if not (np.isfinite(tol_grad) and tol_grad > 0):
        raise ValueError('tol_grad deve ser finito e positivo')
    if not isinstance(max_steps, (int, np.integer)) or max_steps < 0:
        raise ValueError('max_steps deve ser inteiro não negativo')
    ng = 0
    tic = perf_counter()
    step_elapsed = 0.0
    previous = x.copy()
    status = 'Max Steps'
    grad_norm = float('nan')
    steps = 0
    for k in range(max_steps+1):
        g = np.asarray(grad_f(x), dtype=float)
        ng += 1
        if g.shape != x.shape:
            raise ValueError('Formato do gradiente incompatível com o iterado')
        grad_norm = float(np.linalg.norm(g))
        if not np.isfinite(grad_norm):
            status = 'Nonfinite'
            break
        if grad_norm <= tol_grad:
            status = 'Converged'
            break
        if k == max_steps:
            break
        previous = x.copy()
        step_tic = perf_counter()
        x = np.asarray(step_f(x, g), dtype=float)
        step_elapsed += perf_counter()-step_tic
        steps += 1
        if x.shape != previous.shape or not np.all(np.isfinite(x)):
            status = 'Nonfinite'
            break
    return dict(status=status, iterations=steps, n_grad=ng,
                grad_final=grad_norm, x_final=x, x_prev=previous,
                loop_seconds=perf_counter()-tic, step_seconds=step_elapsed)
