"""Mínimos quadrados sintéticos, sob controle do espectro e com ruído inconsistente.

A é m x n (m>n), posto coluna completo; cond_2(A.T@A)=kappa.
O ruído é ortogonal ao espaço coluna, de modo que x_true é o mínimo
exato em aritmética real, mas Ax=b não tem solução.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class LeastSquaresProblem:
    A: np.ndarray
    b: np.ndarray
    x_star: np.ndarray
    x0: np.ndarray
    kappa_hessian: float
    sigma_min: float
    sigma_max: float
    noise_norm: float
    init_mode: str

    def value(self, x):
        r = self.A @ x-self.b
        return .5 * float(r @ r)

    def gradient(self, x):
        return self.A.T @ (self.A @ x-self.b)

    def hessian(self, x):
        return self.A.T @ self.A


def make_problem(n, kappa, seed=2026, oversampling=2, noise_scale=.05,
                 init_mode="extremos"):
    if not (isinstance(n, int) and n >= 2 and np.isfinite(kappa) and kappa >= 1):
        raise ValueError('Exigir n>=2 inteiro e kappa>=1 finito')
    if init_mode not in ("extremos", "multimodal"):
        raise ValueError("init_mode deve ser extremos ou multimodal")
    if not isinstance(oversampling, int) or oversampling < 2:
        raise ValueError("oversampling deve ser inteiro >= 2")
    if not np.isfinite(noise_scale) or noise_scale <= 0:
        raise ValueError("noise_scale deve ser positivo e finito")
    m = oversampling*n
    if m <= n:
        raise ValueError('Exigir número de observações m>n')
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(m, n)), mode='reduced')
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    singular = np.geomspace(1., np.sqrt(kappa), n)
    A = (U*singular) @ V.T
    x_star = rng.normal(size=n)
    raw_noise = rng.normal(size=m)
    orth_noise = raw_noise - U @ (U.T @ raw_noise)
    norm = np.linalg.norm(orth_noise)
    if norm <= 1e-12:
        raise RuntimeError('Ruído ortogonal degenerado')
    noise = noise_scale*orth_noise/norm
    b = A @ x_star + noise
    # Mesma distância inicial sqrt(2) entre cenários. Na base V, colunas
    # correspondem aos valores singulares em ordem crescente.
    if init_mode == "extremos":
        modal = np.zeros(n)
        modal[0] = modal[-1] = 1.0
    else:
        modal = np.full(n, np.sqrt(2.0 / n))
    x0 = x_star + V @ modal
    return LeastSquaresProblem(A, b, x_star, x0, float(kappa), 1.,
                               float(np.sqrt(kappa)), float(np.linalg.norm(noise)), init_mode)
