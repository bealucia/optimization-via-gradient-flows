"""Experimento 1 do TCC: estabilidade espectral, rigidez e custos.

Executar da raiz:
    python -m src.experiments.exp1_quadratic

Arquivos PNG, CSV e TXT em results/exp1_quadratic/.
"""

import csv
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.problems.quadratic import (
    exact_flow, quadratic_gradient, quadratic_hessian,
    quadratic_matrix, quadratic_value,
)
from src.methods.euler import forward_euler, backward_euler

OUT = Path(__file__).resolve().parents[2] / "results" / "exp1_quadratic"
FIGURES, TABLES, LOGS = (OUT / s for s in ("figures", "tables", "logs"))


def save_csv(name, rows):
    if not rows:
        return
    with (TABLES / name).open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def save_plot(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def setup(kappa):
    A = quadratic_matrix(kappa)
    eigvals, V = np.linalg.eigh(A)
    grad = lambda x: quadratic_gradient(x, A)
    hess = lambda x: quadratic_hessian(x, A)
    return A, eigvals, V, grad, hess


def experiment_stability():
    """Passos abaixo, no limite e acima de 2/L, em modo puro."""
    A, vals, V, grad, hess = setup(100)
    L = vals[-1]
    fast = V[:, -1]
    h_values = [0.005, 0.015, 0.020, 0.021]
    labels = [
        "Decaimento sem alternancia", "Oscilacao amortecida",
        "Fronteira de estabilidade", "Oscilacao crescente",
    ]
    n = 20
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    rows = []
    for ax, h, label in zip(axs.flat, h_values, labels):
        ef, _ = forward_euler(grad, fast, h, n)
        eb, info = backward_euler(grad, hess, fast, h, n)
        mu_ef = 1 - h * L
        mu_eb = 1 / (1 + h * L)
        for method, X, mu, style in (
            ("EF", ef, mu_ef, "o-"),
            ("EB", eb, mu_eb, "s--"),
        ):
            a = X @ fast
            ax.plot(range(n + 1), a, style, markersize=2.7, label=method)
            for k, value in enumerate(a):
                rows.append({
                    "metodo": method, "kappa": 100, "h": h,
                    "iteracao": k, "componente": value,
                    "previsto": mu ** k, "erro": abs(value - mu ** k),
                    "mu": mu, "limite_ef": 2 / L,
                })
        ax.set_title(f"{label}\nh={h:g}  |  mu EF={mu_ef:.2f}  |  mu EB={mu_eb:.2f}", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.grid(alpha=0.18)
        ax.legend(loc="best")
        ax.set_xlabel("Iteracao k")
        ax.set_ylabel("Componente no modo rapido")
    fig.suptitle("Euler explicito x implicito: estabilidade (kappa=100)", fontsize=13)
    fig.tight_layout()
    save_plot(fig, "01_estabilidade")
    save_csv("01_estabilidade.csv", rows)
    return max(r["erro"] for r in rows)


def experiment_factors():
    """Fatores teoricos de amplificacao ao variar s=h*L."""
    s = np.linspace(0, 3, 401)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhspan(-1, 1, facecolor="gray", alpha=0.10, label="|mu| < 1")
    ax.plot(s, 1 - s, linewidth=2, label="Euler forward")
    ax.plot(s, 1 / (1 + s), linewidth=2, label="Euler backward")
    ax.axvline(2, color="black", linestyle=":", label="s=2 (fronteira EF)")
    ax.axhline(-1, color="gray", linestyle="--", linewidth=0.8)
    ax.set(xlabel="s = h * lambda_max", ylabel="Fator de amplificacao mu",
           title="Fatores espectrais para o modo rapido")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "02_fatores_espectrais")


def experiment_geometry():
    """Isola geometricamente o zigue-zague com ambos os modos ativos."""
    A, _, V, grad, hess = setup(100)
    x0 = V @ np.array([1.0, 1.0])
    x = np.linspace(-1.6, 1.6, 250)
    X, Y = np.meshgrid(x, x)
    Z = 0.5 * (A[0, 0] * X**2 + 2 * A[0, 1] * X * Y + A[1, 1] * Y**2)
    cases = [
        ("EF", 0.018, 36, "EF: alternancia do modo rapido"),
        ("EB", 0.018, 36, "EB: mesmo passo que EF"),
        ("EB", 1.0, 9, "EB: passo maior permitido"),
    ]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.9), sharex=True, sharey=True)
    for ax, (method, h, n, title) in zip(axs, cases):
        if method == "EF":
            traj, _ = forward_euler(grad, x0, h, n)
        else:
            traj, _ = backward_euler(grad, hess, x0, h, n)
        ax.contour(X, Y, Z, levels=np.geomspace(0.03, 100, 10),
                   linewidths=0.85, alpha=0.48)
        ax.plot(traj[:, 0], traj[:, 1], "o-", markersize=2.7, linewidth=1.1)
        ax.plot(x0[0], x0[1], "*", markersize=12, label="Inicio")
        ax.plot(0, 0, "x", markersize=8, label="Minimo")
        ax.set(title=f"{title}\nh={h:g}, N={n}, T={n*h:g}",
               xlabel="x1", ylabel="x2", xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
        ax.set_aspect("equal")
        ax.grid(alpha=0.12)
    axs[-1].legend(fontsize=8)
    fig.suptitle("Trajetorias nas curvas de nivel (kappa=100); passos indicados em cada painel")
    fig.tight_layout()
    save_plot(fig, "03_geometria")


def experiment_modes():
    """Mesmo passo h: distingue modo lento, modo rapido e solucao exata."""
    A, vals, V, grad, hess = setup(100)
    x0 = V @ np.array([1.0, 1.0])
    h, n = 0.015, 20
    t = h * np.arange(n + 1)
    tref = np.linspace(0, n * h, 501)
    ef, _ = forward_euler(grad, x0, h, n)
    eb, _ = backward_euler(grad, hess, x0, h, n)
    ref = exact_flow(A, x0, tref)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, ax in enumerate(axs):
        ax.plot(tref, ref @ V[:, j], "k-", linewidth=2, label="Continuo exato")
        ax.plot(t, ef @ V[:, j], "o--", markersize=3, label="Euler forward")
        ax.plot(t, eb @ V[:, j], "s-.", markersize=3, label="Euler backward")
        ax.set(title=f"Modo {'lento' if j == 0 else 'rapido'}: lambda={vals[j]:g}",
               xlabel="Tempo t", ylabel="Componente espectral")
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Mesmo h=0.015 e mesmo intervalo temporal")
    fig.tight_layout()
    save_plot(fig, "04_modos_temporais")


def experiment_accuracy():
    """Erro global no tempo final fixo T, com exatamente N=T/h passos."""
    A, _, V, grad, hess = setup(100)
    x0 = V @ np.array([1.0, 1.0])
    T = 0.05
    n_values = [20, 40, 80, 160, 320, 640, 1280]
    target = exact_flow(A, x0, [T])[0]
    rows = []
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    for method, marker in (("EF", "o"), ("EB", "s")):
        hs, errors = [], []
        for n in n_values:
            h = T / n
            if method == "EF":
                traj, _ = forward_euler(grad, x0, h, n)
            else:
                traj, _ = backward_euler(grad, hess, x0, h, n)
            error = float(np.linalg.norm(traj[-1] - target))
            hs.append(h)
            errors.append(error)
            rows.append({"metodo": method, "kappa": 100, "T": T,
                         "h": h, "N": n, "erro_final": error})
        ax.loglog(hs, errors, marker + "-", label=method)
        if method == "EF":
            h_ref = np.array([min(hs), max(hs)])
            ax.loglog(h_ref, errors[-1] * h_ref / hs[-1],
                      "k:", label="Referencia proporcional a h")
    ax.set(xlabel="Passo h, com T=Nh fixo", ylabel="Erro ||x_N-x(T)||",
           title="Precisao temporal: convergencia de primeira ordem")
    ax.grid(which="both", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "05_erro_temporal")
    save_csv("05_erro_temporal.csv", rows)


def experiment_stiffness():
    """Otimizacao ate a mesma norma de gradiente; passos diferentes.

    EF usa h=1.8/kappa (restricao espectral).
    EB usa h=1 (permitido para SPD, mas exige sistema linear interno).
    Contagens de operacoes NAO sao custos unitarios comparaveis.
    """
    kappas = [10, 100, 1000, 10000]
    repetitions, tol = 3, 1e-6
    rows = []
    for kappa in kappas:
        A, vals, V, grad, hess = setup(kappa)
        x0 = V @ np.array([1.0, 1.0])
        for method, h in (("EF", 1.8 / vals[-1]), ("EB", 1.0)):
            times = []
            for _ in range(repetitions):
                tic = perf_counter()
                if method == "EF":
                    traj, info = forward_euler(grad, x0, h, 100000,
                                               tol_grad=tol)
                else:
                    traj, info = backward_euler(grad, hess, x0, h, 100000,
                                                tol_grad=tol)
                times.append(perf_counter() - tic)
                if info["status"] != "Converged":
                    raise RuntimeError(f"{method} nao convergiu para kappa={kappa}: {info['status']}")
            row = {
                "metodo": method, "kappa": kappa, "h": h,
                "tol_grad": tol, "status": info["status"],
                "iteracoes": info["iterations"],
                "tempo_simulado_final": h * info["iterations"],
                "grad_final": float(np.linalg.norm(grad(traj[-1]))),
                "f_final": quadratic_value(traj[-1], A),
                "grad_evals": info["grad_evals"],
                "hess_evals": info["hess_evals"],
                "linear_solves": info["linear_solves"],
                "inner_iterations": info["inner_iterations"],
                "max_implicit_residual": info["max_implicit_residual"],
                "tempo_mediano_s": float(np.median(times)),
                "repeticoes": repetitions,
            }
            rows.append(row)
            print(f"  kappa={kappa:5} | {method} | h={h:.5g} | "
                  f"{info['iterations']:6} passos | {np.median(times):.4f}s")

    save_csv("06_rigidez_custo.csv", rows)
    fig, axs = plt.subplots(1, 3, figsize=(13.3, 4.2))
    for method, style in (("EF", "o-"), ("EB", "s-")):
        rr = [r for r in rows if r["metodo"] == method]
        ks = [r["kappa"] for r in rr]
        axs[0].loglog(ks, [r["iteracoes"] for r in rr], style, label=method)
        axs[1].loglog(ks, [r["grad_evals"] for r in rr], style,
                      label=f"{method}: gradiente")
        axs[2].loglog(ks, [r["tempo_mediano_s"] for r in rr], style, label=method)
        if method == "EB":
            axs[1].loglog(ks, [r["linear_solves"] for r in rr], "^--",
                          label="EB: sistemas lineares")
    titles = ["Iteracoes externas", "Contagem de operacoes", "Tempo mediano (s)"]
    for ax, title in zip(axs, titles):
        ax.set(title=title, xlabel="Numero de condicao kappa")
        ax.grid(which="both", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Mesma tolerancia de gradiente; EF: h=1.8/kappa, EB: h=1")
    fig.tight_layout()
    save_plot(fig, "06_rigidez_custo")
    return rows


def main():
    for folder in (FIGURES, TABLES, LOGS):
        folder.mkdir(parents=True, exist_ok=True)
    print("[1/6] Estabilidade")
    max_error = experiment_stability()
    print("[2/6] Fatores espectrais")
    experiment_factors()
    print("[3/6] Trajetorias")
    experiment_geometry()
    print("[4/6] Modos temporais")
    experiment_modes()
    print("[5/6] Precisao temporal")
    experiment_accuracy()
    print("[6/6] Rigidez e custos")
    rows = experiment_stiffness()

    info = [
        "Experimento 1 - Estabilidade espectral e rigidez",
        "A=Q diag(1,kappa) Q^T, Q: rotacao de pi/4.",
        "EF: x_(k+1)=x_k-h A x_k.",
        "EB: (I+h A)x_(k+1)=x_k, resolvido por Newton interno.",
        "Estabilidade: kappa=100, x0=autovetor do modo rapido, N=20.",
        "Regimes: h=0.005, 0.015, 0.020 e 0.021.",
        "Geometria: x0 com componentes nos dois modos; passos/horizontes indicados.",
        "Modos: mesmo passo h=0.015; fluxo continuo exato por espectro.",
        "Precisao: T=0.05 fixo, N variavel, norma do erro no tempo final.",
        "Otimizacao: kappa=10,100,1000,10000; ||grad f||<=1e-6.",
        "Custo: EF h=1.8/kappa; EB h=1; estrategias de passo diferentes.",
        "Nao comparar tempos de chegada sem considerar que os h sao diferentes.",
        "Tempos: mediana de 3 execucoes no mesmo ambiente.",
        "O benchmark de EB resolve cada passo com Newton; contagens nao tem pesos iguais.",
        "Nenhum grafico PDF: seis PNGs (300 dpi) e CSVs.",
        f"Erro maximo na verificacao espectral: {max_error:.3e}",
        f"Casos no benchmark: {len(rows)}",
    ]
    (LOGS / "metodologia.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    print(f"Concluido. Resultados em: {OUT}")


if __name__ == "__main__":
    main()
