"""Experimento complementar: fluxo gradiente versus fluxo de Newton numa sela.

As referências contínuas são ANALÍTICAS. Compara os métodos em t_k=k h,
distinguindo aproximação a ponto crítico de minimização. A função é
ilimitada inferiormente; a origem NÃO é mínimo.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.problems import saddle
from src.methods.euler import forward_euler
from src.methods.newton import newton_damped
from src.output import folders, csv_write, json_write, png


def _assert_close(observed, expected, label, *, atol=2e-11):
    if not np.allclose(observed, expected, rtol=1e-11, atol=atol):
        raise AssertionError(f'Identidade teórica falhou: {label}')


def experiment_saddle(*, root=None, piloto=False):
    out = folders('exp4_newton_saddle', root=root)
    x0 = np.array([0.5, 1.0])  # f(x0)<0: Newton eleva f até zero.
    T, h = (2.0, 0.1) if piloto else (3.0, 0.1)
    N = int(round(T/h)); t = h*np.arange(N+1)
    fine_t = np.linspace(0.0, T, 801)

    gf_ref = saddle.exact_gradient_flow(x0, t)
    nf_ref = saddle.exact_newton_flow(x0, t)
    gf_dense = saddle.exact_gradient_flow(x0, fine_t)
    nf_dense = saddle.exact_newton_flow(x0, fine_t)
    ef, _ = forward_euler(saddle.gradient, x0, h, N)
    nd = newton_damped(saddle.gradient, saddle.hessian, x0, h, N)

    _assert_close(ef, np.column_stack((x0[0]*(1-h)**np.arange(N+1),
                                      x0[1]*(1+h)**np.arange(N+1))), 'EF sela')
    _assert_close(nd, (1-h)**np.arange(N+1)[:, None]*x0, 'Newton amortecido')
    _assert_close(np.array([saddle.gradient(x) for x in nf_ref]),
                  np.exp(-t)[:, None]*saddle.gradient(x0), 'decaimento gradiente Newton')
    assert saddle.value(x0) < 0
    assert saddle.value(nf_ref[-1]) > saddle.value(x0)

    traces = {
        'GF_exato': (t, gf_ref), 'Newton_fluxo_exato': (t, nf_ref),
        'EF_gradiente': (t, ef), 'Newton_amortecido': (t, nd),
    }
    rows = []
    for name, (times, X) in traces.items():
        for step, (ti, xi) in enumerate(zip(times, X)):
            rows.append(dict(metodo=name, iteracao=step, t=float(ti), h=h,
                             x=float(xi[0]), y=float(xi[1]),
                             f=saddle.value(xi),
                             norma_gradiente=float(np.linalg.norm(saddle.gradient(xi))),
                             distancia_ponto_critico=float(np.linalg.norm(xi)),
                             classificacao='ponto de sela; não é minimizador'))
    csv_write(out['tables']/'01_trajetorias_sela.csv', rows)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    ax[0].plot(gf_dense[:,0], gf_dense[:,1], color='#282828', lw=2.1, label='Fluxo gradiente exato')
    ax[0].plot(nf_dense[:,0], nf_dense[:,1], color='#aa6720', lw=2.1, label='Fluxo de Newton exato')
    ax[0].plot(ef[:,0], ef[:,1], 'o--', color='#2371ad', ms=3.0, lw=1.0, label='EF do fluxo gradiente')
    ax[0].plot(nd[:,0], nd[:,1], 's:', color='#b34c31', ms=2.9, lw=1.2, label='Newton amortecido')
    ax[0].plot(0, 0, 'kx', ms=8, mew=1.6)
    ax[0].set(xlabel='$x$', ylabel='$y$', title='Trajetórias no plano: origem é sela')
    ax[0].grid(alpha=.2); ax[0].legend(fontsize=7.5)
    for name, (tt, X) in traces.items():
        label = name.replace('_', ' ')
        ax[1].plot(tt, [saddle.value(x) for x in X], lw=1.6, label=label)
    ax[1].axhline(0, color='0.5', lw=.8)
    ax[1].set(xlabel='$t=kh$', ylabel='$f(x)$', title='Newton pode aumentar a função objetivo')
    ax[1].grid(alpha=.2); ax[1].legend(fontsize=7.5)
    fig.tight_layout()
    png(fig, out['figures']/'01_sela_fluxos_e_discretizacoes.png')

    steps = (0.5, 1.0, 1.5, 2.0, 2.1)
    n_steps = 18 if piloto else 32
    fig2, ax2 = plt.subplots(figsize=(8.1, 4.8))
    rows_steps = []
    for hh in steps:
        X = newton_damped(saddle.gradient, saddle.hessian, x0, hh, n_steps)
        factor = abs(1-hh)
        theory = float(np.linalg.norm(x0))*factor**np.arange(n_steps+1)
        observed = np.linalg.norm(X, axis=1)
        _assert_close(observed, theory, f'Newton h={hh}', atol=4e-11)
        status = ('convergente' if factor<1 else
                  'fronteira oscilatória' if factor==1 else 'divergente')
        for k, (actual, expected) in enumerate(zip(observed, theory)):
            rows_steps.append(dict(h=hh, iteracao=k, t=hh*k, fator=1-hh,
                                   modulo_fator=factor, status_teorico=status,
                                   norma_ponto_critico=float(actual),
                                   norma_prevista=float(expected)))
        ax2.semilogy(np.arange(n_steps+1), observed, marker='o', markevery=5,
                     ms=3, lw=1.4, label=f'$h={hh:g}$ ({status})')
    ax2.set(xlabel='Iteração $k$ (não é tempo comum)',
            ylabel='Distância ao ponto crítico (sela)',
            title=r'Estabilidade de Newton amortecido: fator $|1-h|$')
    ax2.grid(alpha=.18, which='both'); ax2.legend(fontsize=8)
    fig2.tight_layout()
    png(fig2, out['figures']/'02_estabilidade_newton_amortecido.png')
    csv_write(out['tables']/'02_estabilidade_newton.csv', rows_steps)
    json_write(out['logs']/'config.json', dict(x0=x0.tolist(), T=T, h=h,
        n_steps=N, steps_newton=list(steps), n_steps_estabilidade=n_steps,
        referencias='soluções analíticas exatas',
        observacao='A origem é uma sela e f é ilimitada inferiormente; não interpretar como minimização.'))
    return rows, rows_steps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--piloto', action='store_true')
    args = parser.parse_args()
    trajectories, steps = experiment_saddle(piloto=args.piloto)
    print(f'Experimento complementar Newton: {len(trajectories)} posições, '
          f'{len(steps)} medições de estabilidade')


if __name__ == '__main__':
    main()
