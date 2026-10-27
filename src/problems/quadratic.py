
import numpy as np


def quadratic_matrix(kappa=100.0, theta=np.pi / 4):
    """
    Matriz simétrica positiva definida com
    autovalores 1 e kappa, rotacionada por theta.
    """

    if not np.isfinite(kappa) or kappa < 1:
        raise ValueError(
            "kappa deve ser finito e maior ou igual a 1."
        )

    c = np.cos(theta)
    s = np.sin(theta)

    Q = np.array([
        [c, -s],
        [s, c]
    ])

    D = np.diag([1.0, kappa])

    return Q @ D @ Q.T


def quadratic_value(x, A):
    """Valor f(x) = 0.5 * x^T A x."""

    x = np.asarray(x, dtype=float)

    return float(0.5 * x @ A @ x)


def quadratic_gradient(x, A):
    """Gradiente: A @ x."""

    x = np.asarray(x, dtype=float)

    return A @ x


def quadratic_hessian(x, A):
    """Hessiana constante da quadrática."""

    return A
