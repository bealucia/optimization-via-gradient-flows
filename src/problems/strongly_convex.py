"""Função não quadrática fortemente convexa usada no Módulo 2."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class StronglyConvex:
    kappa: float = 100.0
    epsilon: float = 0.25

    def __post_init__(self):
        if not (np.isfinite(self.kappa) and self.kappa >= 1):
            raise ValueError("kappa deve ser finito e >= 1")
        if not (0 <= self.epsilon < 1):
            raise ValueError("Exigir 0 <= epsilon < 1 para convexidade forte")

    @property
    def m(self):
        return 1.0 - self.epsilon

    @property
    def L(self):
        return self.kappa + self.epsilon

    @staticmethod
    def _x(x):
        x = np.asarray(x, dtype=float)
        if x.shape != (2,):
            raise ValueError("Esperado x em R^2")
        return x

    def value(self, x):
        x = self._x(x)
        return float(.5 * (x[0]**2 + self.kappa * x[1]**2)
                     + self.epsilon * (np.sin(x[0])-x[0]+np.sin(x[1])-x[1]))

    def gradient(self, x):
        x = self._x(x)
        return np.array([x[0] + self.epsilon * (np.cos(x[0]) - 1),
                         self.kappa * x[1] + self.epsilon * (np.cos(x[1])-1)])

    def hessian(self, x):
        x = self._x(x)
        return np.diag([1.0-self.epsilon*np.sin(x[0]),
                        self.kappa-self.epsilon*np.sin(x[1])])

    @property
    def minimizer(self):
        return np.zeros(2)
