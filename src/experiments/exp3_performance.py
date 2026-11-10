"""Módulo 3 — custos da estabilidade em mínimos quadrados sintéticos.

Controle cruzado de condicionamento (κ2(A^T A)), dimensão e condição inicial:
``extremos`` excita os dois modos extremos; ``multimodal`` excita todos.

Implementações comparadas:
EF-direto    gradiente calculado por A.T @ (A @ x - b)
EF-Gram      G=A.T@A e c=A.T@b preparados uma vez, g=G@x-c
EB-Cholesky/LU/QR/SVD    quatro resoluções algebricamente equivalentes.

Medidas: o tempo de preparar o método e o tempo do laço até a mesma tolerância.
Todas as medições são empíricas, dependem de máquina/BLAS e não atestam
superioridade universal. As falhas e casos Max Steps permanecem nos CSVs.

Executar: python -m src.experiments.exp3_performance --piloto
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO
import os
import platform
import sys
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy

from src.methods.linear_implicit import prepare_implicit, iterate_until_tolerance
from src.output import folders, csv_write, json_write, png
from src.problems.least_squares import make_problem

BACKENDS = ('Cholesky', 'LU', 'QR', 'SVD')
METHODS = ('EF-direto', 'EF-Gram') + BACKENDS
PLOT_LABEL = {k: ('EB-' + k if k in BACKENDS else k) for k in METHODS}


def _dispersion(numbers):
    a = np.asarray(numbers, dtype=float)
    med = float(np.median(a))
    return med, float(np.median(np.abs(a-med))), float(a.min()), float(a.max())


def _prepared_run(problem, method, h, tol, max_steps):
    """Retorna métricas de UMA execução, incluindo a preparação medida.

    A geração da instância (A,b,x*) não entra em nenhum tempo. O cálculo de
    gradiente para o teste de parada é incluído em cada laço.
    """
    if method == 'EF-direto':
        setup_seconds = 0.0
        gradient = problem.gradient
        def step(x, g):
            return x - h*g
    elif method == 'EF-Gram':
        t0 = perf_counter()
        G = problem.A.T @ problem.A
        c = problem.A.T @ problem.b
        setup_seconds = perf_counter()-t0
        def gradient(x):
            return G @ x - c
        def step(x, g):
            return x - h*g
    elif method in BACKENDS:
        solver = prepare_implicit(problem.A, problem.b, h, method)
        setup_seconds = solver.factor_seconds
        gradient = solver.monitoring_gradient or problem.gradient
        def step(x, g):
            return solver.step(x)
    else:
        raise ValueError('Método não reconhecido: ' + str(method))
    result = iterate_until_tolerance(gradient, step, problem.x0,
                                     tol_grad=tol, max_steps=max_steps)
    sample = {
        'preparacao_s': float(setup_seconds),
        'laco_s': float(result['loop_seconds']),
        'passos_s': float(result['step_seconds']),
        'total_s': float(setup_seconds + result['loop_seconds']),
        'iteracoes': int(result['iterations']),
        'status': result['status'],
        'avaliacoes_gradiente': int(result['n_grad']),
        # Relatório final pelo gradiente original, independente do backend.
        'gradiente_original_final': float(np.linalg.norm(problem.gradient(result['x_final']))),
        'erro_parametros': float(np.linalg.norm(result['x_final']-problem.x_star)),
        'erro': '',
    }
    return sample


def _max_implicit_residual(problem, method, h, n_steps):
    """Auditoria fora das medidas de tempo: máximo resíduo em TODOS os passos.

    Uma segunda trajetória determinística é construída exclusivamente para
    diagnóstico; seu custo não contamina o benchmark. O denominador é uma
    escala para o sistema implícito B y = x + h A.T b.
    """
    solver = prepare_implicit(problem.A, problem.b, h, method)
    x = problem.x0.copy()
    rhs_constant = h*(problem.A.T @ problem.b)
    Bnorm = 1.0+h*problem.kappa_hessian
    largest = 0.0
    for _ in range(n_steps):
        y = solver.step(x)
        residual = np.linalg.norm(y-x+h*problem.gradient(y))
        scale = Bnorm*np.linalg.norm(y) + np.linalg.norm(x+rhs_constant)
        largest = max(largest, float(residual/max(scale, np.finfo(float).tiny)))
        x = y
    return largest


def _aggregate(raw, *, scenario, dimension, kappa, method, h, repeats, tol, problem):
    times = [r for r in raw if r['cenario']==scenario and r['dimension']==dimension
             and r['kappa_hessian']==kappa and r['metodo']==method]
    assert len(times) == repeats
    statuses = {s['status'] for s in times}
    iterations = {s['iteracoes'] for s in times}
    if len(statuses) != 1 or len(iterations) != 1:
        raise RuntimeError(f'Execuções não reproduzíveis: {scenario}, {dimension}, {kappa}, {method}')
    status = times[-1]['status']
    def group(field):
        return _dispersion([row[field] for row in times]) if status != 'Solver Failure' else ('','','','')
    prep, prep_mad, _, _ = group('preparacao_s')
    loop, loop_mad, _, _ = group('laco_s')
    steps, _, _, _ = group('passos_s')
    total, total_mad, total_min, total_max = group('total_s')
    max_residual = ''
    if method in BACKENDS and status != 'Solver Failure':
        max_residual = _max_implicit_residual(problem, method, h, times[-1]['iteracoes'])
        if max_residual > 1e-8:
            raise RuntimeError(f'Resíduo implícito alto: {scenario}/{dimension}/{kappa}/{method}: {max_residual}')
    return dict(cenario=scenario, dimension=dimension, m=problem.A.shape[0],
        kappa_hessian=kappa, cond_A=float(np.sqrt(kappa)), metodo=method,
        h=h, tol_grad=tol, repeticoes=repeats, status=status,
        iteracoes=times[-1]['iteracoes'],
        gradiente_original_final=times[-1]['gradiente_original_final'],
        erro_parametros=times[-1]['erro_parametros'],
        norma_ruido=problem.noise_norm,
        residuo_relativo_max_todos_passos=max_residual,
        preparacao_mediana_s=prep, preparacao_mad_s=prep_mad,
        laco_mediana_s=loop, laco_mad_s=loop_mad,
        passos_mediana_s=steps, total_mediana_s=total,
        total_mad_s=total_mad, total_min_s=total_min, total_max_s=total_max,
        observacao=times[-1]['erro'])


def benchmark_instance(problem, kappa, dimension, tol, max_steps, repeats):
    """Repetições intercaladas por método, com aquecimento não medido."""
    methods = (('EF-direto',1.8/kappa), ('EF-Gram',1.8/kappa)) + tuple((key,1.0) for key in BACKENDS)
    for method, h in methods:
        try:
            _prepared_run(problem, method, h, tol, max_steps)
        except (np.linalg.LinAlgError, RuntimeError, FloatingPointError, ValueError):
            # Falhas também serão registradas nas repetições medidas.
            pass
    raw = []
    for repetition in range(1, repeats+1):
        # Rotação determinística da ordem: minimiza tendência sistemática.
        order = methods[(repetition-1)%len(methods):]+methods[:(repetition-1)%len(methods)]
        for execution_order,(method,h) in enumerate(order, start=1):
            try:
                sample = _prepared_run(problem, method, h, tol, max_steps)
            except (np.linalg.LinAlgError, RuntimeError, FloatingPointError, ValueError) as exc:
                sample = dict(preparacao_s='', laco_s='', passos_s='', total_s='',
                    iteracoes=0, status='Solver Failure', avaliacoes_gradiente=0,
                    gradiente_original_final='', erro_parametros='',
                    erro=f'{type(exc).__name__}: {exc}')
            raw.append(dict(cenario=problem.init_mode, dimension=dimension,
                m=problem.A.shape[0], kappa_hessian=kappa, metodo=method,
                h=h, repeticao=repetition, ordem_na_rodada=execution_order, **sample))
    summaries = [_aggregate(raw, scenario=problem.init_mode, dimension=dimension,
        kappa=kappa, method=method, h=h, repeats=repeats, tol=tol, problem=problem)
        for method,h in methods]
    return summaries, raw


def _data(rows, method, scenario, fixed_key, fixed_value, x_key):
    return sorted((r for r in rows if r['metodo']==method and r['cenario']==scenario
                   and r[fixed_key]==fixed_value), key=lambda r:r[x_key])


def _condition_figure(rows, out, dims, kappas, scenarios):
    scenario = 'extremos' if 'extremos' in scenarios else scenarios[0]
    n=max(dims)
    fig,ax=plt.subplots(figsize=(8.5,5.1))
    for method in ('EF-direto','EF-Gram','Cholesky'):
        seq=_data(rows,method,scenario,'dimension',n,'kappa_hessian')
        failed=[r for r in seq if r['status']!='Converged']
        if any(r['status']=='Converged' for r in seq):
            ax.loglog([r['kappa_hessian'] for r in seq],
                      [r['iteracoes'] if r['status']=='Converged' else np.nan for r in seq],
                      'o-', linewidth=1.8, label=PLOT_LABEL[method])
        if failed:
            ax.scatter([r['kappa_hessian'] for r in failed],
                       [max(1,r['iteracoes']) for r in failed], marker='x', s=80,
                       label=f'{PLOT_LABEL[method]}: não convergiu')
    ax.set(xlabel=r'Número de condição $\kappa_2(A^\top A)$',
        ylabel='Iterações realizadas (×: sem convergência)',
        title=rf'Condicionamento — cenário extremos ($n={n}$)')
    ax.grid(which='both',alpha=.2);ax.legend(fontsize=9)
    fig.tight_layout();png(fig,out['figures']/'01_condicionamento_iteracoes.png')


def _factor_figure(rows,out,dims,kappas,scenarios):
    scenario = 'extremos' if 'extremos' in scenarios else scenarios[0]
    chosen=[r for r in rows if r['cenario']==scenario and r['dimension']==max(dims)
            and r['kappa_hessian']==max(kappas) and r['metodo'] in BACKENDS]
    selected=sorted(chosen,key=lambda r:BACKENDS.index(r['metodo']))
    fig,ax=plt.subplots(figsize=(8,4.8))
    x=np.arange(len(selected))
    good=[r['status']=='Converged' for r in selected]
    prep=np.array([r['preparacao_mediana_s']*1e3 if good[i] else 0 for i,r in enumerate(selected)])
    loop=np.array([r['laco_mediana_s']*1e3 if good[i] else 0 for i,r in enumerate(selected)])
    ax.bar(x,prep,label='Preparação',alpha=.8)
    ax.bar(x,loop,bottom=prep,label='Laço',alpha=.75)
    for i,r in enumerate(selected):
        if not good[i]:ax.text(i,.05,'NC',ha='center',transform=ax.get_xaxis_transform())
    ax.set_xticks(x,[r['metodo'] for r in selected])
    ax.set(ylabel='Soma das medianas (ms)',
           title=rf'Custos por solver ($n={max(dims)}$, $\kappa={max(kappas)}$)')
    ax.grid(axis='y',alpha=.2);ax.legend()
    fig.tight_layout();png(fig,out['figures']/'02_decomposicao_custo_solver.png')


def _dimension_figure(rows,out,dims,kappas,scenarios):
    scenario = 'extremos' if 'extremos' in scenarios else scenarios[0]
    k=max(kappas)
    fig,ax=plt.subplots(figsize=(8.5,5.1))
    for method in ('EF-Gram','Cholesky','LU','QR','SVD'):
        seq=_data(rows,method,scenario,'kappa_hessian',k,'dimension')
        bad=[r for r in seq if r['status']!='Converged']
        if any(r['status']=='Converged' for r in seq):
            ax.loglog([r['dimension'] for r in seq],
                      [1e3*r['total_mediana_s'] if r['status']=='Converged' else np.nan
                       for r in seq], 'o-', label=PLOT_LABEL[method])
        if bad:
            ax.scatter([r['dimension'] for r in bad],
                       [1e3*r['total_mediana_s'] if isinstance(r['total_mediana_s'],float)
                        else 1e-6 for r in bad],marker='x',s=70,
                       label=PLOT_LABEL[method]+' (NC)')
    ax.set(xlabel='Dimensão n', ylabel='Tempo total mediano (ms; ×: sem convergência)',
            title=rf'Dimensão — cenário extremos ($\kappa={k}$)')
    ax.grid(which='both',alpha=.2);ax.legend(ncol=2,fontsize=9)
    fig.tight_layout();png(fig,out['figures']/'03_dimensao_tempo.png')


def _advantage_figure(rows,out,dims,kappas,scenarios):
    from matplotlib.colors import TwoSlopeNorm
    comparisons=[]
    fig,axes=plt.subplots(1,len(scenarios),figsize=(5.9*len(scenarios),4.5),squeeze=False)
    for ax,scenario in zip(axes[0],scenarios):
        ratios=np.full((len(dims),len(kappas)),np.nan)
        for i,n in enumerate(dims):
            for j,k in enumerate(kappas):
                ef=next(r for r in rows if r['cenario']==scenario and r['dimension']==n
                        and r['kappa_hessian']==k and r['metodo']=='EF-Gram')
                eb=next(r for r in rows if r['cenario']==scenario and r['dimension']==n
                        and r['kappa_hessian']==k and r['metodo']=='Cholesky')
                valid=ef['status']=='Converged' and eb['status']=='Converged'
                ratio=ef['total_mediana_s']/eb['total_mediana_s'] if valid else None
                comparisons.append(dict(cenario=scenario,dimension=n,kappa_hessian=k,
                    status_EF=ef['status'],status_EB=eb['status'],
                    iteracoes_EF=ef['iteracoes'],iteracoes_EB=eb['iteracoes'],
                    T_EF_mediana_s=ef['total_mediana_s'],
                    T_EB_Cholesky_mediana_s=eb['total_mediana_s'],
                    razao_EF_div_EB=ratio if valid else ''))
                if valid:ratios[i,j]=np.log2(ratio)
        if np.isfinite(ratios).any():
            m=max(1.0,float(np.nanmax(np.abs(ratios))))
            im=ax.imshow(ratios,aspect='auto',cmap='coolwarm',
                         norm=TwoSlopeNorm(vmin=-m,vcenter=0,vmax=m))
            fig.colorbar(im,ax=ax,shrink=.8,pad=.02)
        ax.set_xticks(range(len(kappas)),[str(k) for k in kappas])
        ax.set_yticks(range(len(dims)),[str(n) for n in dims])
        ax.set(xlabel=r'$\kappa_2(A^\top A)$',ylabel='Dimensão n',title=scenario)
        for i in range(len(dims)):
            for j in range(len(kappas)):
                ax.text(j,i,('NC' if not np.isfinite(ratios[i,j])
                             else f'{2**ratios[i,j]:.2f}×'),
                        ha='center',va='center',fontsize=9)
    fig.suptitle('Razão de tempos: EF-Gram / EB-Cholesky (>1 favorece EB)')
    fig.tight_layout();png(fig,out['figures']/'04_vantagem_tempo_total.png')
    csv_write(out['tables']/'03_comparacao_tempo_ef_eb.csv',comparisons)
    return comparisons


def _environment():
    buf=StringIO()
    with redirect_stdout(buf):
        np.__config__.show()
    try:
        from threadpoolctl import threadpool_info
        pools=threadpool_info()
    except ImportError:
        pools='threadpoolctl indisponível'
    return dict(python=sys.version.split()[0],numpy=np.__version__,
       scipy=scipy.__version__,matplotlib=matplotlib.__version__,
       plataforma=platform.platform(),cpu_count=os.cpu_count(),
       openblas_threads=os.environ.get('OPENBLAS_NUM_THREADS'),
       omp_threads=os.environ.get('OMP_NUM_THREADS'),
       mkl_threads=os.environ.get('MKL_NUM_THREADS'),
       biblioteca_numerica=buf.getvalue(),pools=pools)


def main(dimensions=(20,60,120),kappas=(10,100,1000),repeats=5,
         tol=1e-6,max_steps=100000, scenarios=('extremos','multimodal'),
         output_root=None):
    if repeats<2:raise ValueError('repeats >= 2')
    if not dimensions or not kappas:raise ValueError('Grade vazia')
    dimensions=tuple(sorted(set(int(n) for n in dimensions)))
    kappas=tuple(sorted(set(float(k) for k in kappas)))
    if any(n<2 for n in dimensions) or any(k<1 or not np.isfinite(k) for k in kappas):
        raise ValueError('n >=2 e kappa >=1 finitos')
    scenarios=tuple(dict.fromkeys(scenarios))
    if not scenarios or any(c not in ('extremos','multimodal') for c in scenarios):
        raise ValueError('Cenários inválidos')
    out=folders('exp3_performance',root=output_root)
    rows=[];raw=[]
    for scenario in scenarios:
        for n in dimensions:
            for kap in kappas:
                problem=make_problem(n=n,kappa=kap,seed=2026,init_mode=scenario)
                if np.linalg.norm(problem.gradient(problem.x_star))>1e-8:
                    raise RuntimeError('Minimizador analítico inconsistente')
                summary,samples=benchmark_instance(problem,kap,n,tol,max_steps,repeats)
                rows.extend(summary);raw.extend(samples)
                for r in summary:
                    print(f'{scenario:9} n={n:>3} κ={kap:>6g} {r["metodo"]:<11} '
                          f'k={r["iteracoes"]:>6} {r["status"]}')
    csv_write(out['tables']/'01_benchmark_completo.csv',rows)
    csv_write(out['tables']/'02_tempos_repeticoes.csv',raw)
    json_write(out['logs']/'config.json',dict(dimensions=list(dimensions),
        kappas=list(kappas),repeats=repeats,scenarios=list(scenarios),
        seed=2026,tol_grad=tol,max_steps=max_steps,oversampling=2,
        noise_scale=.05,EF_h='1.8/kappa',EB_h=1.0,ambiente=_environment(),
        notas=[
          'Dados sintéticos inconsistentes com minimizador conhecido.',
          'Cenários têm a mesma distância inicial sqrt(2); não equivalem a dados reais.',
          'EF-direto usa A.T(Ax-b) sem preparação; EF-Gram usa G,c com preparação temporizada.',
          'EB Cholesky/LU reutilizam também G,c para verificar o gradiente.',
          'QR e SVD utilizam A para gradiente de parada: estratégias completas distintas.',
          'Cada repetição mede preparação mais laço; dados gerados fora do cronômetro.',
          'As repetições são intercaladas por método, após aquecimento por caso.',
          'Resíduo implícito MÁXIMO auditado por trajetória adicional fora do cronômetro.',
          'Falha do solver e orçamento de passos esgotado não equivalem a instabilidade matemática.',
          'Razões de tempos somente quando EF e EB atingiram a MESMA tolerância.',
          'Tempos de máquina são exploratórios e devem ser repetidos no computador do pesquisador.',
          'Comparação de otimização, não de aproximação de x(T) para T fixo.']))
    _condition_figure(rows,out,dimensions,kappas,scenarios)
    _factor_figure(rows,out,dimensions,kappas,scenarios)
    _dimension_figure(rows,out,dimensions,kappas,scenarios)
    comparisons=_advantage_figure(rows,out,dimensions,kappas,scenarios)
    return rows,comparisons


def _values_arg(value,kind):
    try:
        result=tuple(kind(s.strip()) for s in value.split(',') if s.strip())
        if not result:raise ValueError('lista vazia')
        return result
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--piloto',action='store_true',help='grade menor e duas repetições')
    parser.add_argument('--dimensoes',type=lambda s:_values_arg(s,int))
    parser.add_argument('--kappas',type=lambda s:_values_arg(s,float))
    parser.add_argument('--repeticoes',type=int)
    parser.add_argument('--max-passos',type=int,default=100000)
    parser.add_argument('--cenario',choices=('ambos','extremos','multimodal'),default='ambos')
    args=parser.parse_args()
    dims=args.dimensoes if args.dimensoes is not None else ((8,16) if args.piloto else (20,60,120))
    kappas=args.kappas if args.kappas is not None else ((10,100) if args.piloto else (10,100,1000))
    reps=args.repeticoes if args.repeticoes is not None else (2 if args.piloto else 5)
    scenarios=('extremos','multimodal') if args.cenario=='ambos' else (args.cenario,)
    main(dimensions=dims,kappas=kappas,repeats=reps,max_steps=args.max_passos,
         scenarios=scenarios)
