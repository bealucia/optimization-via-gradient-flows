"""Família não quadrática fortemente convexa com acoplamento entre coordenadas.

f(x,y) = (x² + kappa*y²)/2 + epsilon*(sin(x+y) - (x+y)).

Para kappa >= 1 e 0 < epsilon < 1/2:
    (1 - 2*epsilon) I <= Hessiana <= (kappa + 2*epsilon) I.
Esses limites são globais e suficientes, não necessariamente ótimos.
A origem é o único minimizador global.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoupledNonQuadratic:
    kappa: float = 2.0
    epsilon: float = 0.35

    def __post_init__(self):
        if not (np.isfinite(self.kappa) and self.kappa >= 1.0):
            raise ValueError("kappa deve ser finito e >= 1")
        if not (np.isfinite(self.epsilon) and 0.0 < self.epsilon < 0.5):
            raise ValueError("Exigir 0 < epsilon < 1/2 para garantir convexidade forte")

    @staticmethod
    def _x(x):
        x = np.asarray(x, dtype=float)
        if x.shape != (2,) or not np.all(np.isfinite(x)):
            raise ValueError("Esperado vetor finito em R²")
        return x

    @property
    def m(self):
        """Limite inferior global certificado para autovalores da Hessiana."""
        return 1.0 - 2.0 * self.epsilon

    @property
    def L(self):
        """Limite superior global certificado para autovalores da Hessiana."""
        return self.kappa + 2.0 * self.epsilon

    @property
    def minimizer(self):
        return np.zeros(2)

    def value(self, x):
        x = self._x(x)
        s = x[0] + x[1]
        return float(0.5*(x[0]**2 + self.kappa*x[1]**2)
                     + self.epsilon*(np.sin(s)-s))

    def gradient(self, x):
        x = self._x(x)
        perturbation = self.epsilon*(np.cos(x[0]+x[1])-1.0)
        return np.array((x[0]+perturbation, self.kappa*x[1]+perturbation))

    def hessian(self, x):
        x = self._x(x)
        coupling = -self.epsilon*np.sin(x[0]+x[1])
        return np.array(((1.0+coupling, coupling),
                         (coupling, self.kappa+coupling)), dtype=float)
