
import numpy as np


def _check_inputs(x0, h, n_steps):
    """Valida as entradas dos métodos."""

    x = np.asarray(x0, dtype=float).copy()

    if (
        x.ndim != 1
        or x.size == 0
        or not np.all(np.isfinite(x))
    ):
        raise ValueError(
            "x0 deve ser um vetor finito e nao vazio."
        )

    if not np.isfinite(h) or h <= 0:
        raise ValueError(
            "h deve ser positivo e finito."
        )

    if (
        not isinstance(n_steps, (int, np.integer))
        or n_steps < 0
    ):
        raise ValueError(
            "n_steps deve ser inteiro nao negativo."
        )

    return x


# =====================================================
# EULER FORWARD
# =====================================================

def forward_euler(grad_f, x0, h, n_steps):
    """
    Euler explicito aplicado ao fluxo gradiente.

    x_(k+1) = x_k - h * grad_f(x_k)

    Executa exatamente n_steps passos.
    """

    x = _check_inputs(x0, h, n_steps)

    trajectory = [x.copy()]

    for _ in range(n_steps):

        g = np.asarray(
            grad_f(x),
            dtype=float
        )

        if (
            g.shape != x.shape
            or not np.all(np.isfinite(g))
        ):
            raise RuntimeError(
                "Gradiente invalido em Euler forward."
            )

        x = x - h * g

        if not np.all(np.isfinite(x)):
            raise RuntimeError(
                "Iterado nao finito em Euler forward."
            )

        trajectory.append(x.copy())

    return np.asarray(trajectory)


# =====================================================
# EULER BACKWARD
# =====================================================

def backward_euler(
    grad_f,
    hess_f,
    x0,
    h,
    n_steps,
    inner_tol=1e-12,
    max_newton=20
):
    """
    Euler implicito aplicado ao fluxo gradiente.

    x_(k+1) = x_k - h * grad_f(x_(k+1))

    Resolve cada passo utilizando Newton interno.

    O passo somente e aceito quando o residuo
    da equacao implicita satisfaz a tolerancia.
    """

    x = _check_inputs(x0, h, n_steps)

    if not np.isfinite(inner_tol) or inner_tol <= 0:
        raise ValueError(
            "inner_tol deve ser positiva e finita."
        )

    if (
        not isinstance(max_newton, (int, np.integer))
        or max_newton < 1
    ):
        raise ValueError(
            "max_newton deve ser inteiro positivo."
        )

    trajectory = [x.copy()]

    I = np.eye(x.size)

    for _ in range(n_steps):

        # Aproximacao inicial para Newton
        y = x.copy()

        threshold = inner_tol * (
            1.0 + np.linalg.norm(x)
        )

        for _ in range(max_newton + 1):

            g = np.asarray(
                grad_f(y),
                dtype=float
            )

            if (
                g.shape != x.shape
                or not np.all(np.isfinite(g))
            ):
                raise RuntimeError(
                    "Gradiente invalido no passo implicito."
                )

            # Equacao implicita F(y) = 0
            F = y - x + h * g

            residual = np.linalg.norm(F)

            if residual <= threshold:
                break

            H = np.asarray(
                hess_f(y),
                dtype=float
            )

            if (
                H.shape != (x.size, x.size)
                or not np.all(np.isfinite(H))
            ):
                raise RuntimeError(
                    "Hessiana invalida no passo implicito."
                )

            # Jacobiana de F
            J = I + h * H

            try:
                direction = np.linalg.solve(
                    J, -F
                )

            except np.linalg.LinAlgError as exc:
                raise RuntimeError(
                    "Falha no sistema linear interno."
                ) from exc

            y = y + direction

            if not np.all(np.isfinite(y)):
                raise RuntimeError(
                    "Iterado interno nao finito."
                )

        else:
            raise RuntimeError(
                "Newton interno nao convergiu."
            )

        x = y

        trajectory.append(x.copy())

    return np.asarray(trajectory)
