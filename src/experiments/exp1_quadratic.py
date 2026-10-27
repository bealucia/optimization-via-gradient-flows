
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.problems.quadratic import (
    quadratic_matrix,
    quadratic_gradient,
    quadratic_hessian
)

from src.methods.euler import (
    forward_euler,
    backward_euler
)


# =====================================================
# DIRETORIOS
# =====================================================

ROOT = Path(__file__).resolve().parents[2]

OUT = ROOT / "results" / "exp1_quadratic"

FIGURES = OUT / "figures"
TABLES = OUT / "tables"


# =====================================================
# EXPERIMENTO
# =====================================================

def main():

    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    # 1. Problema quadratico

    kappa = 100

    A = quadratic_matrix(kappa)

    eigvals, eigvecs = np.linalg.eigh(A)

    L = eigvals[-1]

    # Condicao inicial: autovetor do modo rapido
    x0 = eigvecs[:, -1]

    grad = lambda x: quadratic_gradient(x, A)
    hess = lambda x: quadratic_hessian(x, A)

    # 2. Parametros do experimento

    steps = 20

    step_sizes = [
        0.005,
        0.015,
        0.020,
        0.021
    ]

    rows = []

    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 7)
    )

    k = np.arange(steps + 1)

    # 3. Comparacao dos metodos

    for ax, h in zip(axes.flat, step_sizes):

        ef = forward_euler(
            grad, x0, h, steps
        )

        eb = backward_euler(
            grad, hess, x0, h, steps
        )

        mu_ef = 1 - h * L

        mu_eb = 1 / (1 + h * L)

        for name, traj, mu, style in (
            ("EF", ef, mu_ef, "o-"),
            ("EB", eb, mu_eb, "s--")
        ):

            # Componente na direcao do modo rapido
            component = traj @ x0

            # Previsao espectral exata
            theoretical = mu ** k

            error = np.max(
                np.abs(component - theoretical)
            )

            rows.append({
                "method": name,
                "kappa": kappa,
                "h": h,
                "lambda_max": L,
                "mu": mu,
                "last_component": component[-1],
                "maximum_spectral_error": error
            })

            ax.plot(
                k,
                component,
                style,
                markersize=3,
                label=name
            )

        ax.set_title(
            f"h={h:.3f} | "
            f"mu EF={mu_ef:.2f} | "
            f"mu EB={mu_eb:.2f}"
        )

        ax.axhline(
            0,
            linewidth=0.8,
            color="gray"
        )

        ax.set_xlabel("Iteração k")

        ax.set_ylabel(
            "Componente no modo rápido"
        )

        ax.legend()
        ax.grid(alpha=0.2)

    # 4. Salvar gráfico PNG

    fig.suptitle(
        "Estabilidade espectral: "
        "Euler forward versus backward "
        "(kappa=100)"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES / "stability_initial.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # 5. Salvar tabela CSV

    csv_path = TABLES / "stability_initial.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as fp:

        writer = csv.DictWriter(
            fp,
            fieldnames=list(rows[0])
        )

        writer.writeheader()
        writer.writerows(rows)

    # 6. Resultados no terminal

    print(
        f"Autovalores: {eigvals}; "
        f"limite de EF: 2/L={2/L:.6f}"
    )

    for row in rows:

        print(
            f"{row['method']} "
            f"h={row['h']:.3f}, "
            f"mu={row['mu']:.4f}, "
            f"erro="
            f"{row['maximum_spectral_error']:.2e}"
        )

    print(
        f"Arquivos salvos em: {OUT}"
    )


if __name__ == "__main__":
    main()
