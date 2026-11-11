"""Experimento 1: estabilidade espectral, geometria, precisão e rigidez.

Executar a partir da raiz do projeto:
    python -m src.experiments.exp1_quadratic

Quatro análises principais (quatro figuras PNG de 300 dpi):
    01_estabilidade.png
    02_geometria.png
    03_erro_temporal.png
    04_rigidez_iteracoes.png

As rotinas auxiliares de fatores espectrais e modos temporais são preservadas,
mas não integram as quatro figuras principais. Para gerá-las:
    python -m src.experiments.exp1_quadratic --auxiliares

Saídas: results/exp1_quadratic/{figures,tables,logs}/.
Requer src.problems.quadratic e src.methods.euler já existentes no projeto.
"""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.problems.quadratic import (
    exact_flow,
    quadratic_gradient,
    quadratic_hessian,
    quadratic_matrix,
    quadratic_value,
)
from src.methods.euler import forward_euler, backward_euler
from src.output import csv_write, png


OUT = Path(__file__).resolve().parents[2] / "results" / "exp1_quadratic"
FIGURES = OUT / "figures"
TABLES = OUT / "tables"
LOGS = OUT / "logs"
AUX_FIGURES = FIGURES / "auxiliares"

def save_csv(name, rows):
    """Adaptador de saída que preserva os nomes históricos dos CSVs."""
    csv_write(TABLES / name, rows)


def save_plot(fig, name, auxiliary=False):
    """Adaptador de saída: a figura permanece PNG, 300 dpi."""
    folder = AUX_FIGURES if auxiliary else FIGURES
    png(fig, folder / f"{name}.png")


def setup(kappa):
    A = quadratic_matrix(kappa)
    eigvals, V = np.linalg.eigh(A)
    grad = lambda x: quadratic_gradient(x, A)
    hess = lambda x: quadratic_hessian(x, A)
    return A, eigvals, V, grad, hess


def _fixed_length(trajectory, n, description):
    """Protege ensaios temporais contra parada precoce silenciosa."""
    if np.asarray(trajectory).shape[0] != n + 1:
        raise RuntimeError(
            f"{description}: esperado N+1={n+1} iterados, "
            f"obtidos {np.asarray(trajectory).shape[0]}."
        )


def experiment_stability():
    """Análise 1: quatro regimes, isolando o autovetor de lambda_max."""
    _, vals, V, grad, hess = setup(100)
    lam_max = vals[-1]
    fast = V[:, -1]  # ||fast||=1; x_0=fast implica c_0=1.
    n = 20
    regimes = (
        (0.005, "Decaimento sem alternância"),
        (0.015, "Oscilação amortecida"),
        (0.020, "Fronteira: sem amortecimento"),
        (0.021, "Oscilação crescente"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), sharex=True)
    rows = []
    handles = []

    for idx, (ax, (h, title)) in enumerate(zip(axes.flat, regimes)):
        ef, _ = forward_euler(grad, fast, h, n)
        eb, _ = backward_euler(grad, hess, fast, h, n)
        _fixed_length(ef, n, "Estabilidade EF")
        _fixed_length(eb, n, "Estabilidade EB")

        mu_ef = 1.0 - h * lam_max
        mu_eb = 1.0 / (1.0 + h * lam_max)
        for method, X, mu, style, marker in (
            ("EF", ef, mu_ef, "-", "o"),
            ("EB", eb, mu_eb, "--", "s"),
        ):
            component = X @ fast
            line, = ax.plot(
                np.arange(n + 1), component,
                linestyle=style, marker=marker, linewidth=1.65,
                markersize=3.0, markevery=1,
                label=f"Euler {('forward' if method == 'EF' else 'backward')}",
            )
            if idx == 0:
                handles.append(line)
            for k, value in enumerate(component):
                rows.append({
                    "metodo": method, "kappa": 100, "h": h,
                    "iteracao": k, "componente": float(value),
                    "previsto": float(mu ** k),
                    "erro": float(abs(value - mu ** k)),
                    "mu": mu, "limite_ef": 2.0 / lam_max,
                })

        ax.set_title(
            f"({chr(97 + idx)}) {title}\n"
            rf"$h={h:.3f}$  |  $\mu_{{\mathrm{{EF}}}}={mu_ef:.2f}$  "
            rf"|  $\mu_{{\mathrm{{EB}}}}={mu_eb:.2f}$",
            fontsize=10.5, pad=11,
        )
        ax.axhline(0, color="0.35", linewidth=0.75, zorder=0)
        ax.grid(alpha=0.15, linewidth=0.75)
        ax.set_xlim(-0.5, n + 0.5)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.spines[["top", "right"]].set_visible(False)


    fig.suptitle("Regimes de estabilidade dos métodos de Euler", fontsize=14, y=0.985)
    fig.text(0.5, 0.935,
             r"$\kappa=100$, $\lambda_{\max}=100$, $x_0=v_{\max}$; "
             r"$z_k=v_{\max}^{\mathsf{T}}x_k$; 20 iterações",
             ha="center", va="center", fontsize=10)
    fig.legend(handles, ["Euler forward (EF)", "Euler backward (EB)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2,
               frameon=False, fontsize=10)
    fig.supxlabel("Iteração $k$", fontsize=11, y=0.035)
    fig.supylabel(r"Componente $z_k$ no modo rápido", fontsize=11, x=0.018)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.12, top=0.78,
                        wspace=0.18, hspace=0.41)
    save_plot(fig, "01_estabilidade")
    save_csv("01_estabilidade.csv", rows)
    return max(r["erro"] for r in rows)


def experiment_geometry():
    """Análise 2: compara trajetórias EF/EB com o fluxo contínuo exato.

    Mesmo x0, mesmo passo h e horizonte T nos dois painéis. As direções
    espectrais são retas invariantes, não vetores com orientação preferida.
    """
    A, eigenvalues, V, grad, hess = setup(100)
    v_slow, v_fast = V[:, 0], V[:, -1]
    x0 = v_slow + v_fast
    h, n = 0.018, 200
    T = n * h

    grid = np.linspace(-1.6, 0.6, 300)
    X, Y = np.meshgrid(grid, grid)
    Z = 0.5 * (A[0, 0] * X**2 + 2 * A[0, 1] * X * Y + A[1, 1] * Y**2)

    t_ref = np.linspace(0.0, T, 500)
    flow_ref = exact_flow(A, x0, t_ref)
    xT = exact_flow(A, x0, [T])[0]

    fig, axes = plt.subplots(
        1, 2, figsize=(13, 7.5), sharex=True, sharey=True
    )
    slow_color = "#07847b"
    fast_color = "#ca6a18"
    disc_color = "#33ba18"
    cont_color = "#bb2a34"

    handles = []
    rows = []
    final_errors = {}

    for idx, (ax, method, title) in enumerate(zip(
        axes, ("EF", "EB"),
        ("(a) Euler forward — zigue-zague", "(b) Euler backward — sem alternância"),
    )):
        if method == "EF":
            traj, _ = forward_euler(grad, x0, h, n)
        else:
            traj, _ = backward_euler(grad, hess, x0, h, n)
        _fixed_length(traj, n, f"Geometria {method}")
        final_errors[method] = float(np.linalg.norm(traj[-1] - xT))

        ax.contour(
            X, Y, Z, levels=np.geomspace(0.3, 100, 11),
            colors="0.68", linewidths=0.75, alpha=0.72, zorder=1,
        )

        for vector, color, label in (
            (v_slow, slow_color, rf"Modo lento ($\lambda={eigenvalues[0]:g}$)"),
            (v_fast, fast_color, rf"Modo rápido ($\lambda={eigenvalues[-1]:g}$)"),
        ):
            endpoints = 1.85 * np.vstack((-vector, vector))
            line, = ax.plot(
                endpoints[:, 0], endpoints[:, 1],
                color=color, linestyle=(0, (4, 3)), linewidth=1.2,
                alpha=0.9, zorder=2, label=label,
            )
            if idx == 0:
                handles.append(line)

        continuous, = ax.plot(
            flow_ref[:, 0], flow_ref[:, 1],
            color=cont_color, linewidth=1.65, linestyle="-",
            alpha=0.95, zorder=4, label="Fluxo contínuo exato",
        )
        discrete, = ax.plot(
            traj[:, 0], traj[:, 1],
            "o-", color=disc_color, markersize=2.6, linewidth=1.25,
            zorder=3, label="Trajetória numérica",
        )
        initial, = ax.plot(
            x0[0], x0[1], "*", color="black",
            markersize=8, zorder=6, label="Condição inicial",
        )
        minimum, = ax.plot(
            0, 0, "x", color="black", markeredgewidth=1.7,
            markersize=8.5, zorder=6, label="Mínimo",
        )
        if idx == 0:
            handles.extend((continuous, discrete, initial, minimum))

        ax.set_title(title, pad=10, fontsize=11)
        ax.set(xlim=(-1.6, 0.6), ylim=(-1.6, 0.6))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.10, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)

        for k, point in enumerate(traj):
            rows.append({
                "metodo": method,
                "tipo": "discreto",
                "iteracao": int(k),
                "tempo": float(k * h),
                "x1": float(point[0]),
                "x2": float(point[1]),
            })

    for j, point in enumerate(flow_ref):
        rows.append({
            "metodo": "CONTINUO",
            "tipo": "continuo_exato",
            "iteracao": int(j),
            "tempo": float(t_ref[j]),
            "x1": float(point[0]),
            "x2": float(point[1]),
        })

    fig.suptitle(
        r"Geometria das trajetórias e modos espectrais ($\kappa=100$)",
        fontsize=15, y=0.975,
    )

    fig.text(
        0.5, 0.928,
        r"Mesmo $x_0=v_{\min}+v_{\max}$, $h=0{,}018$, "
        rf"$N={n}$, $T={T:.3f}$; referência: $x(t)=e^{{-At}}x_0$",
        ha="center", va="center", fontsize=10.6,
    )

    fig.text(
        0.5, 0.892,
        rf"Erro final em $T$: EF = {final_errors['EF']:.3e}   |   "
        rf"EB = {final_errors['EB']:.3e}",
        ha="center", va="center", fontsize=9.8, color="0.35",
    )

    fig.supxlabel(r"Coordenada $x_1$", fontsize=11, y=0.135)
    fig.supylabel(r"Coordenada $x_2$", fontsize=11, x=0.03)

    fig.legend(
        handles, [line.get_label() for line in handles],
        loc="lower center", bbox_to_anchor=(0.5, 0.03),
        ncol=3, fontsize=9.1, frameon=False,
        columnspacing=1.5, handlelength=2.5,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.93,
        bottom=0.22,
        top=0.79,
        wspace=0.18,
    )
    save_plot(fig, "02_geometria")
    save_csv("02_geometria.csv", rows)
    return final_errors


def experiment_accuracy():
    """Análise 3: erro temporal em T fixo, h decrescente e N crescente.

    Eixo inferior: passo h, decrescente da esquerda para a direita.
    Eixo superior: N=T/h, crescente da esquerda para a direita.
    As ordens são obtidas pelo ajuste de log E em função de log h.
    """
    A, _, V, grad, hess = setup(100)
    x0 = V @ np.array([1.0, 1.0])
    T = 0.05
    n_values = np.array((20, 40, 80, 160, 320, 640, 1280), dtype=int)
    h_values = T / n_values
    target = exact_flow(A, x0, [T])[0]

    rows = []
    slopes = {}
    errors_by_method = {}
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    for method, marker, linestyle, color in (
        ("EF", "o", "-", "tab:blue"),
        ("EB", "s", "--", "tab:orange"),
    ):
        errors = []
        for n, h in zip(n_values, h_values):
            if method == "EF":
                traj, _ = forward_euler(grad, x0, h, int(n))
            else:
                traj, _ = backward_euler(grad, hess, x0, h, int(n))
            _fixed_length(traj, int(n), f"Precisão {method}")

            error = float(np.linalg.norm(traj[-1] - target))
            errors.append(error)
            rows.append({
                "metodo": method, "kappa": 100, "T": T,
                "h": float(h), "N": int(n), "erro_final": error,
            })

        errors = np.asarray(errors, dtype=float)
        if not np.all(np.isfinite(errors)) or np.any(errors <= 0):
            raise RuntimeError(
                f"Erros não positivos ou não finitos impedem o ajuste log-log: {method}."
            )
        # Com T fixo, E(h,T) ~ C h^p e a inclinação de log E x log h é +p.
        slopes[method] = float(np.polyfit(np.log(h_values), np.log(errors), 1)[0])
        errors_by_method[method] = errors

        ax.loglog(
            h_values, errors, linestyle=linestyle, marker=marker,
            color=color, linewidth=1.8, markersize=6.5,
            markerfacecolor="white" if method == "EB" else color,
            markeredgewidth=1.5, zorder=4 if method == "EF" else 3,
            label="Euler forward" if method == "EF" else "Euler backward",
        )

    # Referência de primeira ordem E_ref(h) = C*h.
    # Determinamos C no passo mais refinado, usando a média geométrica
    # dos erros de EF e EB. Neste ensaio, a reta fica entre as duas curvas.
    # A expressão se prolonga até (h,E)=(0,0), mas a origem não pode
    # ser representada em eixos logarítmicos.
    h_reference = np.array([h_values.min(), h_values.max()], dtype=float)
    c_reference = (
        np.sqrt(errors_by_method["EF"][-1] * errors_by_method["EB"][-1])
        / h_values[-1]
    )
    reference_error = c_reference * h_reference
    ax.loglog(
        h_reference, reference_error,
        color="0.38", linestyle=(0, (2, 2)), linewidth=1.6,
        alpha=0.95, zorder=2,
        label=r"Referência $E_{\mathrm{ref}}(h)=Ch$",
    )

    # O passo diminui da esquerda para a direita.
    ax.set_xlim(h_values.max() * 1.16, h_values.min() / 1.16)
    ax.set_ylim(
        reference_error.min() * 0.72,
        max(errors_by_method["EF"].max(), errors_by_method["EB"].max()) * 1.6,
    )
    ax.set_xticks(h_values)
    ax.set_xticklabels([f"{h:.3g}" for h in h_values])
    ax.tick_params(axis="x", which="major", labelsize=8.5)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel(r"Passo $h$ (decrescente $\rightarrow$), com $T=0{,}05$ fixo",
                  fontsize=11, labelpad=8)
    ax.set_ylabel(r"Erro final $\|x_N-x(T)\|_2$", fontsize=11)
    ax.grid(which="major", color="0.70", alpha=0.38, linewidth=0.8)
    ax.grid(which="minor", color="0.83", alpha=0.20, linewidth=0.6)
    ax.spines[["right", "top"]].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.96,
              edgecolor="0.83", fontsize=9.5)

    ax.text(
        0.03, 0.055,
        r"Ordem estimada ($E \propto h^p$)" "\n"
        rf"EF: $p={slopes['EF']:.2f}$   |   EB: $p={slopes['EB']:.2f}$",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white",
              "edgecolor": "0.85", "alpha": 0.95},
    )

    # Exatamente as mesmas posições horizontais; o rótulo superior é N=T/h.
    upper_ax = ax.twiny()
    upper_ax.set_xscale("log")
    upper_ax.set_xlim(ax.get_xlim())
    upper_ax.set_xticks(h_values)
    upper_ax.set_xticklabels([str(int(n)) for n in n_values])
    upper_ax.set_xlabel(r"Número de passos $N=T/h$ (crescente $\rightarrow$)",
                        fontsize=10, labelpad=10)
    upper_ax.tick_params(axis="x", which="major", labelsize=9,
                         length=4, pad=5, colors="0.3")
    upper_ax.tick_params(axis="x", which="minor", length=0)
    upper_ax.spines[["right", "left", "bottom"]].set_visible(False)
    upper_ax.spines["top"].set_color("0.7")

    fig.suptitle("Precisão temporal: convergência dos métodos de Euler",
                 fontsize=14, y=0.98)
    fig.text(
        0.55, 0.025,
        r"$E_{\mathrm{ref}}(0)=0$: a reta passa por $(0,0)$, "
        r"não representável na escala log-log.",
        fontsize=8.5, ha="center", color="0.38",
    )
    fig.subplots_adjust(left=0.13, right=0.975, bottom=0.19, top=0.75)
    save_plot(fig, "03_erro_temporal")
    save_csv("03_erro_temporal.csv", rows)
    return slopes


def experiment_stiffness():
    """Análise 4: iterações até ||grad f||<=1e-6, não custo unitário.

    EF: h=1.8/kappa; EB: h=1. São estratégias de passo diferentes.
    Dados de tempo e operações ficam no CSV, não em gráficos adicionais.
    """
    kappas = (10, 100, 1000, 10000)
    repetitions, tol = 3, 1e-6
    # Uma única condição inicial no espaço físico para TODOS os kappa.
    # A base é fixa, mas np.linalg.eigh pode trocar o sinal dos autovetores;
    # por isso não reconstruímos x0 com V a cada valor de kappa.
    _, vals_ref, V_ref, _, _ = setup(100)
    v_min_ref, v_max_ref = V_ref[:, 0], V_ref[:, -1]
    x0_comum = v_min_ref + v_max_ref
    z0_lento = float(v_min_ref @ x0_comum)  # = 1
    assert np.isclose(z0_lento, 1.0)

    rows = []
    for kappa in kappas:
        A, vals, V, grad, hess = setup(kappa)
        x0 = x0_comum.copy()
        for method, h in (("EF", 1.8 / vals[-1]), ("EB", 1.0)):
            times, counts = [], []
            for _ in range(repetitions):
                tic = perf_counter()
                if method == "EF":
                    traj, info = forward_euler(
                        grad, x0, h, 100000, tol_grad=tol
                    )
                else:
                    traj, info = backward_euler(
                        grad, hess, x0, h, 100000, tol_grad=tol
                    )
                times.append(perf_counter() - tic)
                if info["status"] != "Converged":
                    raise RuntimeError(
                        f"{method} não convergiu para kappa={kappa}: "
                        f"{info['status']}"
                    )
                counts.append(info["iterations"])
            if len(set(counts)) != 1:
                raise RuntimeError(f"Contagem inconsistente entre repetições: {method}, kappa={kappa}.")

            row = {
                "metodo": method, "kappa": kappa, "h": h,
                "tol_grad": tol, "status": info["status"],
                "iteracoes": info["iterations"],
                "tempo_simulado_final": h * info["iterations"],
                "grad_final": float(np.linalg.norm(grad(traj[-1]))),
                "f_final": float(quadratic_value(traj[-1], A)),
                "grad_evals": info["grad_evals"],
                "hess_evals": info["hess_evals"],
                "linear_solves": info["linear_solves"],
                "inner_iterations": info["inner_iterations"],
                "max_implicit_residual": info["max_implicit_residual"],
                "tempo_mediano_s": float(np.median(times)),
                "repeticoes": repetitions,
            }
            rows.append(row)
            print(f"  κ={kappa:5} | {method} | h={h:.5g} | "
                  f"{info['iterations']:6} passos | {np.median(times):.4f}s")

    # Preserva o CSV histórico e o nome da figura usados pelo projeto.
    save_csv("04_rigidez_iteracoes.csv", rows)

    # A figura agora evidencia a causa da rigidez: o modo lento isolado
    # permite passos grandes, mas o modo rápido limita o passo de EF.
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.4, 5.4))
    trajectory_rows = []

    k_left = np.arange(13)
    for h, color in ((0.5,"#1f77b4" ), (1.0,"#33ba18"), (1.8, "#168a70" )):
        # Mesmo x0 do painel (b). Aqui evoluímos somente sua projeção
        # no modo lento: o passo NÃO é aplicado ao sistema completo.
        z = np.abs(z0_lento * (1.0 - h * vals_ref[0]) ** k_left)
        ax_a.plot(k_left, z, "o-", color=color, linewidth=1.9,
                  markersize=4, label=rf"$h={h:g}$")
        trajectory_rows.extend({
            "painel": "modo_lento_isolado", "metodo": "EF",
            "kappa": "", "h": h, "iteracao": int(k),
            "componente_lenta_abs": float(value)
        } for k, value in zip(k_left, z))

    ax_a.set(title=r"(a) Projeção no modo lento ($\lambda=1$)",
             xlabel="Iteração $k$",
             ylabel=r"Distância modal $|z_k^{(\mathrm{lento})}|$")
    ax_a.set_xlim(-0.3, 12.3)
    ax_a.set_ylim(-0.035, 1.07)
    ax_a.axhline(0, color="0.25", linewidth=1.2, label="Solução: distância zero")
    ax_a.legend(loc="upper right", fontsize=9)
    # Ponto inicial comum aos dois painéis. A ordenada representa a
    # magnitude da PROJEÇÃO lenta de x0, não o vetor x0 no plano físico.
    ax_a.plot(0, abs(z0_lento), "ko", markersize=6.8, zorder=10)
    ax_a.text(0.72, 1.024, r"$x_0=v_{\min}+v_{\max}$",
              fontsize=10, color="black", va="center", ha="left")
    ax_a.grid(alpha=0.18)

    # Calcula apenas a projeção espectral, cuja recorrência é exata
    # para EF em uma quadrática SPD; não é uma aproximação adicional.

    k_right = np.arange(2001)
    colors = ("#33ba18", "#1f77b4", "#168a70", "#bb2a34")
    for kappa, color in zip(kappas, colors):
        h = 1.8 / kappa
        mu_slow = 1.0 - h
        # Projeção do MESMO x0 físico nos autovetores de cada A_kappa.
        _, _, V_kappa, _, _ = setup(kappa)
        z0_slow_kappa = float(V_kappa[:, 0] @ x0_comum)
        if not np.isclose(abs(z0_slow_kappa), 1.0, atol=1e-9):
            raise RuntimeError("A família quadrática mudou as direções espectrais.")
        z = np.abs(z0_slow_kappa * mu_slow ** k_right)

        # Seleção de pontos apenas para visualização.
        # Mantém mais pontos na região de decaimento rápido.
        limite_inicial = min(2000, int(np.ceil(6.0 / h)))
        passo_marcacao = 25 # min(20, max(1, int(np.ceil(0.3 / h))))

        indices = np.unique(np.concatenate([
            np.arange(0, limite_inicial + 1, passo_marcacao),
            np.linspace(limite_inicial, 2000, 25, dtype=int),
            [2000],
        ]))

        # Linha completa + marcadores espaçados, para não poluir o gráfico.
        limite_inicial = min(2000, int(np.ceil(6.0 / h)))
        passo_marcacao = min(80, max(1, int(np.ceil(0.3 / h))))

        indices = np.unique(np.concatenate([
            np.arange(0, limite_inicial + 1, passo_marcacao),
            np.linspace(limite_inicial, 2000, 25, dtype=int),
            [2000],
        ]))

        ax_b.plot(
            k_right,
            z,
            "o-",
            color=color,
            linewidth=1,
            markersize=2.0,
            markevery=indices,
            label=rf"EF: $\kappa={kappa}$, $h={h:.2g}$",
        )

    # EB: fator do modo lento 1/(1+h)=1/2, independente de kappa.
    k_eb = np.arange(21)
    z_eb = np.abs(z0_lento * 0.5 ** k_eb)
    ax_b.plot(
        k_eb,
        z_eb,
        "o--",
        color="black",
        linewidth=1,
        markersize=2.0,
        label="EB: $h=1$ (referência)",
    )
    trajectory_rows.extend({
        "painel": "modo_lento_restrito", "metodo": "EB",
        "kappa": 100, "h": 1.0, "iteracao": int(k),
        "componente_lenta_abs": float(value)
    } for k, value in zip(k_eb, z_eb))

    ax_b.axhline(0, color="0.25", linewidth=1.2)
    ax_b.plot(0, abs(z0_lento), "ko", markersize=6.8, zorder=10)
    ax_b.text(125, 1.024, r"$x_0=v_{\min}+v_{\max}$",
              fontsize=10, color="black", va="center", ha="left")
    ax_b.set(title="(b) Modo lento no sistema completo",
             xlabel="Iteração $k$",
             ylabel=r"Distância modal $|z_k^{(\mathrm{lento})}|$")
    ax_b.set_xlim(-25, 2025)
    ax_b.set_ylim(-0.035, 1.07)
    ax_b.grid(alpha=0.18)
    ax_b.legend(loc="upper right", fontsize=8.5)
    # ax_b.text(0.03, 0.08,
    #          r"EF: estabilidade do modo rápido exige $h<2/\kappa$",
    #          transform=ax_b.transAxes, fontsize=9,
    #          bbox=dict(facecolor="white", edgecolor="0.85", alpha=0.92))

    fig.suptitle("Rigidez: o modo rápido restringe o avanço do modo lento",
                 fontsize=13.5, y=0.98)
    fig.text(0.5, 0.015,
             r"(a) Projeção lenta isolada; (b) EF usa $h=1{,}8/\kappa$ "
             r"no sistema completo. Linha $y=0$: solução.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.055, 1, 0.94])
    save_plot(fig, "04_rigidez_iteracoes")
    save_csv("04_rigidez_trajetorias.csv", trajectory_rows)
    return rows


# ---------------------------------------------------------------------------
# ROTINAS AUXILIARES (não integram o conjunto das quatro figuras principais).
# ---------------------------------------------------------------------------

def experiment_factors():
    """Fatores teóricos, disponíveis apenas mediante --auxiliares."""
    s = np.linspace(0, 3, 401)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.axhspan(-1, 1, color="0.85", alpha=0.4,
               label=r"Interior: $|\mu|<1$; bordas excluídas")
    ax.plot(s, 1 - s, linewidth=2, label="Euler forward")
    ax.plot(s, 1 / (1 + s), linewidth=2, label="Euler backward")
    ax.axvline(2, color="black", linestyle=":", label=r"$s=2$: fronteira EF")
    ax.axhline(-1, color="0.5", linestyle="--", linewidth=0.8)
    ax.set(xlabel=r"$s=h\lambda_{\max}$",
           ylabel=r"Fator de amplificação $\mu$",
           title="Fatores espectrais para o modo rápido")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "aux_fatores_espectrais", auxiliary=True)


def experiment_modes():
    """Evolução modal, disponível apenas mediante --auxiliares."""
    A, vals, V, grad, hess = setup(100)
    x0 = V @ np.array([1.0, 1.0])
    h, n = 0.015, 20
    t = h * np.arange(n + 1)
    tref = np.linspace(0, n * h, 501)
    ef, _ = forward_euler(grad, x0, h, n)
    eb, _ = backward_euler(grad, hess, x0, h, n)
    _fixed_length(ef, n, "Modos EF")
    _fixed_length(eb, n, "Modos EB")
    ref = exact_flow(A, x0, tref)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for j, ax in enumerate(axes):
        ax.plot(tref, ref @ V[:, j], "k-", linewidth=2,
                label="Fluxo contínuo exato")
        ax.plot(t, ef @ V[:, j], "o--", markersize=3, label="Euler forward")
        ax.plot(t, eb @ V[:, j], "s-.", markersize=3, label="Euler backward")
        ax.set(title=f"Modo {'lento' if j == 0 else 'rápido'}: "
                     rf"$\lambda={vals[j]:g}$",
               xlabel="Tempo $t$", ylabel="Componente espectral")
        ax.axhline(0, color="0.5", linewidth=0.7)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(r"Evolução modal: mesmo $h=0{,}015$ e horizonte temporal")
    fig.tight_layout()
    save_plot(fig, "aux_modos_temporais", auxiliary=True)


def main(include_auxiliary=False):
    for folder in (FIGURES, TABLES, LOGS):
        folder.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })

    print("[1/4] Regimes de estabilidade")
    max_error = experiment_stability()
    print("[2/4] Geometria e modos espectrais")
    experiment_geometry()
    print("[3/4] Precisão temporal")
    slopes = experiment_accuracy()
    print("[4/4] Rigidez e convergência")
    rows = experiment_stiffness()

    # Retira os arquivos antigos da pasta principal sem excluir resultados.

    if include_auxiliary:
        print("[Auxiliar] Fatores de amplificação")
        experiment_factors()
        print("[Auxiliar] Modos temporais")
        experiment_modes()

    info = [
        "Experimento 1 - quatro análises principais",
        "A=Q diag(1,kappa) Q^T, Q: rotação pi/4.",
        "EF: x_(k+1)=x_k-h A x_k.",
        "EB: (I+h A)x_(k+1)=x_k, resolvido pelo algoritmo compartilhado.",
        "Figuras principais: 01_estabilidade, 02_geometria,",
        "03_erro_temporal, 04_rigidez_iteracoes (PNG 300 dpi).",
        "Análise 1: kappa=100, x0=autovetor rápido, N=20.",
        "Análise 1: h=0.005, 0.015, 0.020, 0.021; sem parada antecipada e sem zoom.",
        "Análise 2: dois painéis com mesmo x0=v_min+v_max, h=0.018, N=36, T=0.648.",
        "Análise 2: eixos espectrais e fluxo contínuo exato x(t)=exp(-At)x0; CSV 02_geometria.csv.",
        "Análise 3: kappa=100, T=0.05 fixo; N variável; erro em x(T).",
        "Análise 3: eixo inferior h decrescente; superior N=T/h crescente; referência E_ref(h)=C*h, passando por (0,0) em escala linear.",
        f"Inclinação observada (EF): {slopes['EF']:.5f}",
        f"Inclinação observada (EB): {slopes['EB']:.5f}",
        "Análise 4: kappa=10,100,1000,10000; ||grad f||<=1e-6.",
        "Análise 4: mesma condição inicial física x0=v_min+v_max para ambos os painéis e todos os kappa (base de referencia kappa=100).",
        "Análise 4(a): integra apenas a coordenada modal lenta dessa condição inicial; passos grandes não são aplicados ao sistema completo.",
        "Análise 4(b): modo lento sob restrição do modo rápido; EF h=1.8/kappa; EB h=1.",
        "CSV 04_rigidez_iteracoes preserva contagens e tempos; 04_rigidez_trajetorias contém os dados das curvas.",
        "Resultados são gravados em results/exp1_quadratic, sem arquivar versões anteriores.",
        "Tempos ilustrativos: mediana de 3 repetições no mesmo ambiente.",
        "Não tratar operações de EF e EB como custos unitários equivalentes.",
        "Auxiliares opcionais (fora das quatro figuras principais): "
        + ("geradas" if include_auxiliary else "não geradas"),
        f"Erro máximo na verificação espectral: {max_error:.3e}",
        f"Casos no benchmark: {len(rows)}",
    ]
    (LOGS / "metodologia.txt").write_text("\n".join(info) + "\n", encoding="utf-8")
    print(f"Concluído. Resultados em: {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auxiliares", action="store_true",
        help="gera também os gráficos auxiliares de fatores e modos",
    )
    args = parser.parse_args()
    main(include_auxiliary=args.auxiliares)
