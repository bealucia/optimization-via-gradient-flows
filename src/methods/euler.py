"""Euler explicito e implicito para fluxos gradientes.

Com tol_grad=None: exatamente n_steps passos (estabilidade e erro temporal).
Com tol_grad>0: para por norma do gradiente ou pelo limite n_steps.
O dicionario info registra o custo e o estado da execucao.
"""

import numpy as np


def _validate(x0, h, n_steps, tol_grad):
    x = np.asarray(x0, dtype=float).copy()
    if x.ndim != 1 or x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("x0 deve ser um vetor finito nao vazio")
    if not np.isfinite(h) or h <= 0:
        raise ValueError("h deve ser positivo e finito")
    if not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
        raise ValueError("n_steps deve ser inteiro nao negativo")
    if tol_grad is not None and (not np.isfinite(tol_grad) or tol_grad <= 0):
        raise ValueError("tol_grad deve ser positivo e finito")
    return x


def _grad(grad_f, x, info):
    g = np.asarray(grad_f(x), dtype=float)
    info["grad_evals"] += 1
    if g.shape != x.shape or not np.all(np.isfinite(g)):
        raise RuntimeError("gradiente invalido ou nao finito")
    return g


def _hess(hess_f, x, info):
    H = np.asarray(hess_f(x), dtype=float)
    info["hess_evals"] += 1
    if H.shape != (x.size, x.size) or not np.all(np.isfinite(H)):
        raise RuntimeError("Hessiana invalida ou nao finita")
    return H


def _info():
    return {
        "status": "Fixed Steps", "iterations": 0,
        "grad_evals": 0, "hess_evals": 0, "linear_solves": 0,
        "inner_iterations": 0, "max_implicit_residual": 0.0,
    }


def _finish(traj, info):
    X = np.asarray(traj)
    info["iterations"] = len(X) - 1
    return X, info


def forward_euler(grad_f, x0, h, n_steps, tol_grad=None):
    """x_{k+1}=x_k-h grad f(x_k); retorna (trajetoria, info)."""
    x = _validate(x0, h, n_steps, tol_grad)
    traj = [x.copy()]
    info = _info()
    for _ in range(n_steps):
        g = _grad(grad_f, x, info)
        if tol_grad is not None and np.linalg.norm(g) <= tol_grad:
            info["status"] = "Converged"
            return _finish(traj, info)
        x = x - h * g
        if not np.all(np.isfinite(x)):
            info["status"] = "Numerical Failure"
            return _finish(traj, info)
        traj.append(x.copy())
    if tol_grad is not None:
        g = _grad(grad_f, x, info)
        info["status"] = "Converged" if np.linalg.norm(g) <= tol_grad else "Max Steps"
    return _finish(traj, info)


def backward_euler(
    grad_f, hess_f, x0, h, n_steps,
    tol_grad=None, inner_tol=1e-12, max_newton=15,
):
    """Euler implicito; resolve y-x+h grad f(y)=0 por Newton interno.

    Para a quadratica SPD, a equacao implicita e linear e Newton
    converge em uma atualizacao (salvo erros de arredondamento).
    """
    x = _validate(x0, h, n_steps, tol_grad)
    if not np.isfinite(inner_tol) or inner_tol <= 0:
        raise ValueError("inner_tol deve ser positivo e finito")
    if not isinstance(max_newton, (int, np.integer)) or max_newton < 1:
        raise ValueError("max_newton deve ser inteiro positivo")
    I = np.eye(x.size)
    traj = [x.copy()]
    info = _info()

    for _ in range(n_steps):
        if tol_grad is not None:
            g = _grad(grad_f, x, info)
            if np.linalg.norm(g) <= tol_grad:
                info["status"] = "Converged"
                return _finish(traj, info)

        y = x.copy()
        threshold = inner_tol * (1 + np.linalg.norm(x))
        success = False
        for j in range(max_newton + 1):
            g = _grad(grad_f, y, info)
            F = y - x + h * g
            residual = float(np.linalg.norm(F))
            if not np.isfinite(residual):
                raise RuntimeError("residuo implicito nao finito")
            if residual <= threshold:
                success = True
                info["max_implicit_residual"] = max(
                    info["max_implicit_residual"], residual
                )
                break
            if j == max_newton:
                break
            H = _hess(hess_f, y, info)
            try:
                direction = np.linalg.solve(I + h * H, -F)
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("falha na resolucao do passo implicito") from exc
            info["linear_solves"] += 1
            info["inner_iterations"] += 1
            y = y + direction
            if not np.all(np.isfinite(y)):
                raise RuntimeError("iterado implicito nao finito")
        if not success:
            raise RuntimeError(
                f"Newton interno nao convergiu: residuo={residual:.3e}"
            )
        x = y
        traj.append(x.copy())

    if tol_grad is not None:
        g = _grad(grad_f, x, info)
        info["status"] = "Converged" if np.linalg.norm(g) <= tol_grad else "Max Steps"
    return _finish(traj, info)
