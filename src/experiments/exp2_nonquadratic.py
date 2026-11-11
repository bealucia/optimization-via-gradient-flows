"""Módulo 2: duas investigações não quadráticas.

1. Problema separável: convergência, contração e dissipação.
2. Problema acoplado: trajetórias sob variação de h e sensibilidade inicial.
Análises antigas de robustez, precisão e resíduos permanecem auxiliares.
Comparações por iteração e erro temporal no mesmo t respondem perguntas distintas.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.problems.strongly_convex import StronglyConvex
from src.problems.coupled_nonquadratic import CoupledNonQuadratic
from src.methods.euler import forward_euler, backward_euler
from src.output import folders, png, csv_write, json_write
from scipy.integrate import solve_ivp


def experiment_convergence(kappas=(10, 100), tol=1e-6, max_steps=12000,
                           mostrar_limite_auxiliar=True):
    out = folders('exp2_nonquadratic')
    x0 = np.array([1., 1.])
    rows, traces = [], []
    fig, axes = plt.subplots(1, len(kappas), figsize=(11, 4.5), squeeze=False)
    demo_traj = None
    demo_problem = None
    for ax, kappa in zip(axes[0], kappas):
        problem = StronglyConvex(float(kappa))
        h_exp = 1.8/problem.L  # Sob o limite suficiente h<2/L
        variants = (('EF', h_exp, 'EF, $h=1{,}8/L$'),
                    ('EB', h_exp, 'EB, mesmo $h$'),
                    ('EB', 1., 'EB, $h=1$'))
        for method, h, label in variants:
            trial = _trajectory(problem, method, x0, h, n_steps=max_steps,
                                tol_grad=tol)
            traj, info, norms, values, failure = (trial['X'], trial['info'],
                trial['g'], trial['f'], trial['falha'])
            row = dict(kappa=kappa, metodo=method, h=h,
                       caso=label, status=trial['status'],
                       iteracoes=info.get('iterations', ''), falha=failure,
                       grad_final='' if traj is None else float(norms[-1]),
                       f_final='' if traj is None else float(values[-1]),
                       numero_aumentos_f='' if traj is None else int(np.count_nonzero(np.diff(values)>1e-12)),
                       grad_evals=info.get('grad_evals', ''),
                       hess_evals=info.get('hess_evals', ''),
                       linear_solves=info.get('linear_solves', ''),
                       inner_iterations=info.get('inner_iterations', ''),
                       max_implicit_residual=info.get('max_implicit_residual', ''))
            rows.append(row)
            if traj is None:
                continue
            if info['status'] != 'Converged':
                print('Aviso: não convergiu:', row)
            ax.semilogy(np.arange(len(norms)), np.maximum(norms, 1e-16),
                        linewidth=1.8, label=label)
            for j, (grad, fval) in enumerate(zip(norms, values)):
                traces.append(dict(kappa=kappa, metodo=method, caso=label,
                                   h=h, iteracao=j, grad_norm=float(grad),
                                   f=float(fval), distancia=float(np.linalg.norm(traj[j]))))
            if kappa == max(kappas) and method == 'EB' and h == 1:
                demo_traj, demo_problem = traj, problem
        ax.axhline(tol, color='0.35', ls=':', lw=1.2, label=r'Tolerância $10^{-6}$')
        ax.set(xlabel='Iteração externa $k$', title=rf'$\kappa={kappa}$')
        ax.grid(which='both', alpha=.22)
        ax.legend(fontsize=8)
    axes[0,0].set_ylabel(r'Norma do gradiente $\|\nabla f(x_k)\|_2$')
    fig.suptitle('Função não quadrática: convergência sob passos distintos')
    fig.tight_layout()
    png(fig, out['figures'] / '01_separavel_convergencia.png')

    if demo_traj is not None and mostrar_limite_auxiliar:
        # Contração teórica válida para todo passo proximal EXATO de uma função
        # m-fortemente convexa, aqui comparada com EB Newton tolerante.
        h = 1.
        q = 1/(1+h*demo_problem.m)
        k = np.arange(len(demo_traj))
        observed = np.linalg.norm(demo_traj, axis=1)
        bound = np.linalg.norm(demo_traj[0])*q**k
        fig, ax = plt.subplots(figsize=(7.6, 4.7))
        ax.semilogy(k, observed, 'o-', ms=3.5, label='EB: distância observada')
        ax.semilogy(k, bound, '--', label=rf'Limite teórico: $(1+hm)^{{-k}}$, $m={demo_problem.m:g}$')
        ax.set(xlabel='Iteração externa $k$',
               ylabel=r'Distância $\|x_k-x^*\|_2$',
               title='Contração de Euler backward e limite de convexidade forte')
        ax.grid(which='both', alpha=.24)
        ax.legend()
        fig.tight_layout()
        png(fig, out['figures'] / 'auxiliares/limite_contracao.png')
        # Alertar, e não afirmar teorema numérico, se erro de solver interferir.
        if np.any(observed > bound + 1e-8):
            print('Aviso: estimativa de contração não satisfeita dentro da margem numérica.')

    csv_write(out['tables'] / '01_separavel_convergencia_resumo.csv', rows)
    csv_write(out['tables'] / '01_separavel_convergencia_trajetorias.csv', traces)
    json_write(out['logs'] / '01_separavel_convergencia_config.json', dict(kappas=list(kappas), x0=x0.tolist(),
               tol_grad=tol, max_steps=max_steps, funcao='strongly_convex',
               epsilon=.25, m=.75,
               observacao='Compara otimização por iteração; horizontes temporais diferem.' ))
    print('Módulo 2 concluído:', out['figures'].parent)
    return rows




METHOD_NAMES = {'EF': 'Euler forward', 'EB': 'Euler backward'}


def _trajectory(problem, method, x0, h, *, n_steps, tol_grad=None):
    """Uma chamada controlada; exceções entram no CSV com status explícito."""
    try:
        if method == 'EF':
            X, info = forward_euler(problem.gradient, np.asarray(x0, dtype=float),
                                    float(h), int(n_steps), tol_grad=tol_grad)
        elif method == 'EB':
            X, info = backward_euler(problem.gradient, problem.hessian,
                                     np.asarray(x0, dtype=float), float(h),
                                     int(n_steps), tol_grad=tol_grad,
                                     inner_tol=1e-12, max_newton=15)
        else:
            raise ValueError(f'Método desconhecido: {method}')
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 2 or not np.isfinite(X).all():
            raise FloatingPointError('Trajetória inválida')
        if tol_grad is None and len(X) != n_steps + 1:
            raise RuntimeError('Comprimento incompatível com passos fixos')
        f = np.asarray([problem.value(x) for x in X], dtype=float)
        g = np.asarray([np.linalg.norm(problem.gradient(x)) for x in X], dtype=float)
        return {'status': str(info.get('status')), 'X': X, 'f': f, 'g': g,
                'info': info, 'falha': ''}
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError,
            OverflowError) as exc:
        inner_failure = (method == 'EB' and
                         (str(exc).startswith('EB:') or isinstance(exc, np.linalg.LinAlgError)))
        return {'status': 'Falha solver interno' if inner_failure else 'Falha numérica',
                'X': None, 'f': None, 'g': None,
                'info': {}, 'falha': f'{type(exc).__name__}: {exc}'}


def _summary(problem, method, label, h, trial, *, kappa, x0_name, regime,
             tol_grad=None):
    """Métricas com mesma chave para todos os estados, inclusive falhas."""
    f = trial['f']
    info = trial['info']
    return dict(ensaio=regime, kappa=float(kappa), x0_nome=x0_name,
                metodo=method, estrategia=label, h=float(h),
                limite_suficiente_ef=2/problem.L,
                tol_grad='' if tol_grad is None else tol_grad,
                status=trial['status'], falha=trial['falha'],
                iteracoes=info.get('iterations', ''),
                grad_final='' if trial['g'] is None else float(trial['g'][-1]),
                f_inicial='' if f is None else float(f[0]),
                f_final='' if f is None else float(f[-1]),
                numero_aumentos_f='' if f is None else int(np.count_nonzero(np.diff(f) > 1e-11)),
                maior_aumento_f='' if f is None else float(max(0., np.max(np.diff(f)))) if len(f) > 1 else 0.,
                grad_evals=info.get('grad_evals', ''),
                hess_evals=info.get('hess_evals', ''),
                linear_solves=info.get('linear_solves', ''),
                inner_iterations=info.get('inner_iterations', ''),
                max_implicit_residual=info.get('max_implicit_residual', ''))


def experiment_dissipation(*, piloto=False):
    """Diferencia a identidade contínua de dissipação da garantia discreta."""
    out = folders('exp2_nonquadratic')
    problem = StronglyConvex(100.)
    x0 = np.array([1., 1.])
    L = problem.L
    n_steps = 20 if piloto else 40
    variations = [('EF', .5/L, 'EF: $h=0{,}5/L$'),
                  ('EF', 1.8/L, 'EF: $h=1{,}8/L$'),
                  ('EF', 2.2/L, 'EF: $h=2{,}2/L$'),
                  ('EB', 1.8/L, 'EB: $h=1{,}8/L$'),
                  ('EB', 1., 'EB: $h=1$'),
                  ('EB', 4., 'EB: $h=4$')]
    rows = []
    histories = []
    fig, ax = plt.subplots(figsize=(8.3, 5.2))
    for method, h, label in variations:
        trial = _trajectory(problem, method, x0, h, n_steps=n_steps)
        rows.append(_summary(problem, method, label, h, trial,
                             kappa=100, x0_name='(1,1)', regime='dissipacao'))
        if trial['X'] is None:
            continue
        for j, (fx, gx) in enumerate(zip(trial['f'], trial['g'])):
            histories.append(dict(kappa=100, metodo=method, estrategia=label,
                                  h=float(h), iteracao=j, tempo_simulado=j*h,
                                  f=float(fx), grad_norm=float(gx),
                                  delta_f='' if j == 0 else float(fx-trial['f'][j-1])))
        linestyle = '--' if method == 'EB' else '-'
        ax.semilogy(np.arange(len(trial['f'])), np.maximum(trial['f'], 1e-15),
                    linestyle=linestyle, linewidth=1.8, label=label)
    ax.set(xlabel='Iteração externa $k$',
           ylabel=r'$f(x_k)-f(x^*)$ (escala logarítmica)',
           title='Dissipação da função objetivo: passos explícitos e implícitos')
    ax.grid(which='both', alpha=.20)
    ax.legend(fontsize=8.7, ncol=2)
    fig.tight_layout()
    png(fig, out['figures']/'02_separavel_dissipacao.png')
    csv_write(out['tables']/'02_separavel_dissipacao_resumo.csv', rows)
    csv_write(out['tables']/'02_separavel_dissipacao_trajetorias.csv', histories)
    json_write(out['logs']/'02_separavel_dissipacao_config.json', {
        'problema':'StronglyConvex', 'kappa':100, 'm':problem.m, 'L':L,
        'x0':x0.tolist(), 'passos_fixos':n_steps,
        'estrategias':[{'metodo':m, 'h':h, 'rotulo':lab} for m,h,lab in variations],
        'nota':'EF: 0<h<2/L é suficiente para descida global; h>=2/L não implica falha universal.'})
    return rows


def experiment_robustness(*, piloto=False):
    """Varia κ, x0 e estratégia; registra convergência, Max Steps e falha."""
    out = folders('exp2_nonquadratic')
    kappas = (10, 100) if piloto else (10, 100, 1000)
    initial = {'positivo':np.array([1.,1.])}
    if not piloto:
        initial.update({'misto':np.array([-2.,.5]),
                        'negativo':np.array([1.5,-1.5])})
    tol = 1e-6
    max_steps = 4500 if piloto else 12000
    rows = []
    for kap in kappas:
        prob = StronglyConvex(float(kap))
        variants = [('EF',1.8/prob.L,'EF: estável 1,8/L'),
                    ('EF',2.2/prob.L,'EF: acima de 2/L'),
                    ('EB',1.8/prob.L,'EB: mesmo passo de EF'),
                    ('EB',1.,'EB: h=1'),
                    ('EB',4.,'EB: h=4')]
        for label,x0 in initial.items():
            for method,h,strategy in variants:
                trial = _trajectory(prob, method, x0, h, n_steps=max_steps,
                                    tol_grad=tol)
                row = _summary(prob, method, strategy, h, trial,
                               kappa=kap, x0_name=label, regime='robustez',
                               tol_grad=tol)
                row['max_steps'] = max_steps
                row['x0'] = repr(x0.tolist())
                rows.append(row)
    csv_write(out['tables']/'auxiliares/robustez_grade.csv', rows)
    json_write(out['logs']/'auxiliares/robustez_config.json', {
        'kappas':list(kappas), 'condicoes_iniciais':{k:v.tolist() for k,v in initial.items()},
        'tol_grad':tol, 'max_steps':max_steps, 'solver_EB':{'inner_tol':1e-12,
        'max_newton':15}, 'nota':'Falha e Max Steps são categorias diferentes de Converged.'})

    # Visual de estados com uma classe por combinação e anotação de razões.
    plotrows = [r for r in rows if r['x0_nome'] == 'positivo']
    methods = ['EF: estável 1,8/L','EF: acima de 2/L','EB: mesmo passo de EF',
               'EB: h=1','EB: h=4']
    labels = ('Converged','Max Steps','Falha solver interno','Falha numérica')
    mapping = {lab:i for i,lab in enumerate(labels)}
    values = np.array([[mapping.get(next(r for r in plotrows if r['kappa']==kap
                                         and r['estrategia']==method)['status'], 3)
                       for kap in kappas] for method in methods])
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(['#3c9a87','#d7b04d','#d96a57','#9c5c8d'])
    fig,ax = plt.subplots(figsize=(7.5, 4.6))
    ax.imshow(values, aspect='auto', cmap=cmap, norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4))
    ax.set_xticks(range(len(kappas)), [str(k) for k in kappas])
    ax.set_yticks(range(len(methods)), methods)
    ax.set_xlabel(r'Condicionamento nominal $\kappa$')
    ax.set_title('Robustez dos passos: condição inicial $(1,1)$')
    for i in range(len(methods)):
        for j in range(len(kappas)):
            text = ['Convergiu','Máx. passos','Falha interna','Falha num.'][values[i,j]]
            ax.text(j,i,text,ha='center',va='center',fontsize=8,
                    color='black' if values[i,j]<2 else 'white')
    fig.tight_layout()
    png(fig, out['figures']/'auxiliares/robustez_status.png')
    return rows


def experiment_accuracy(*, piloto=False):
    """Integração temporal em T fixo; referência DOP853 tolerâncias declaradas."""
    out = folders('exp2_nonquadratic')
    p = StronglyConvex(100.)
    x0 = np.array([1.,1.])
    T = .05
    Ns = (20,40,80,160) if piloto else (20,40,80,160,320,640)
    solution = solve_ivp(lambda t,x:-p.gradient(x), (0,T), x0, method='DOP853',
                         rtol=2e-13, atol=2e-15)
    if not solution.success:
        raise RuntimeError(f'Referência contínua falhou: {solution.message}')
    target = solution.y[:,-1]
    rows=[]
    rates={}
    fig,ax = plt.subplots(figsize=(8.0,5.0))
    colors = {'EF':'#2166ad','EB':'#e08020'}
    for method in ('EF','EB'):
        H=[]; E=[]
        for n in Ns:
            h=T/n
            trial=_trajectory(p, method, x0, h, n_steps=n)
            if trial['X'] is None:
                raise RuntimeError(f'Precisão falhou: {method}, N={n}: {trial["falha"]}')
            err=float(np.linalg.norm(trial['X'][-1]-target))
            if err <= 0:
                raise RuntimeError(f'Erro não positivo em {method}, N={n}')
            rows.append(dict(metodo=method, kappa=100, T=T, N=n, h=h,
                             erro_final=err, solver_status=trial['status'],
                             max_implicit_residual=trial['info'].get('max_implicit_residual',0)))
            H.append(h); E.append(err)
        slope=float(np.polyfit(np.log(H),np.log(E),1)[0])
        rates[method]=slope
        ax.loglog(H,E,'o-' if method=='EF' else 's--',color=colors[method],
                  linewidth=1.9,markersize=5,label=f'{METHOD_NAMES[method]} ($p={slope:.2f}$)')
    # E_ref=Ch (origem matemática; não representável em escala log-log).
    midpoint=len(Ns)//2
    hmid=T/Ns[midpoint]
    efs=[r['erro_final'] for r in rows if r['metodo']=='EF']
    ebs=[r['erro_final'] for r in rows if r['metodo']=='EB']
    C=float(np.sqrt(efs[midpoint]*ebs[midpoint])/hmid)
    hs=np.array([T/n for n in Ns])
    ax.loglog(hs,C*hs,':',color='0.4',linewidth=1.4,label=r'Referência $O(h)$')
    ax.invert_xaxis()
    ax.set_xticks(hs,[f'{h:.3g}' for h in hs])
    def h_to_n(h):
        h=np.asarray(h,dtype=float)
        return np.divide(T,h,out=np.full_like(h,np.inf),where=h!=0)
    def n_to_h(n):
        n=np.asarray(n,dtype=float)
        return np.divide(T,n,out=np.full_like(n,np.inf),where=n!=0)
    top=ax.secondary_xaxis('top',functions=(h_to_n,n_to_h))
    top.set_xticks(list(Ns),[str(n) for n in Ns]);top.set_xlabel('Número de passos $N$')
    ax.set(xlabel=r'Passo $h$ decrescente, $T=Nh=0{,}05$',
           ylabel=r'Erro $\|x_N-x(T)\|_2$',
           title='Precisão temporal: função não quadrática, $T$ fixo')
    ax.grid(which='both',alpha=.20)
    ax.legend(fontsize=9)
    fig.tight_layout()
    png(fig,out['figures']/'auxiliares/precisao_temporal.png')
    csv_write(out['tables']/'auxiliares/precisao_temporal.csv',rows)
    json_write(out['logs']/'auxiliares/precisao_config.json',{
        'kappa':100, 'x0':x0.tolist(), 'T':T, 'Ns':list(Ns),
        'referencia':'scipy.integrate.solve_ivp DOP853',
        'rtol':2e-13,'atol':2e-15,'xT_referencia':target.tolist(),
        'ordens_estimadas':rates,
        'nota':'Esta é integração até T fixo; não é comparação de iterações até o mínimo.'})
    return rows



def experiment_trajectory_robustness(*, piloto=False, root=None):
    """Fluxo, duas condições iniciais, contração, fidelidade e equação variacional.

    Todas as comparações temporais usam os MESMOS t_k = k h. A referência
    contínua é numérica, Radau, verificada com DOP853 independente.
    """
    out = folders('exp2_nonquadratic', root=root)
    problem = StronglyConvex(kappa=100.0)
    x0 = np.array([1.0, 1.0])
    delta0 = np.array([0.10, -0.10])
    x0_other = x0 + delta0
    h = 0.01
    N = 40 if piloto else 80
    T = N*h
    t = h*np.arange(N+1)
    q_ef = max(abs(1-h*problem.m), abs(1-h*problem.L))
    q_eb = 1/(1+h*problem.m)
    assert q_ef < 1 and q_eb < 1

    def integrate(initial, method='Radau'):
        kwargs = dict(fun=lambda tt, xx: -problem.gradient(xx),
                      t_span=(0, T), y0=initial, t_eval=t,
                      method=method, rtol=1e-11, atol=1e-13)
        if method == 'Radau':
            kwargs['jac'] = lambda tt, xx: -problem.hessian(xx)
        sol = solve_ivp(**kwargs)
        if not sol.success or sol.y.shape != (2, N+1):
            raise RuntimeError(f'Falhou integração de referência {method}: {sol.message}')
        return sol.y.T

    ref0 = integrate(x0)
    ref1 = integrate(x0_other)
    independent_ref = integrate(x0, method='DOP853')
    independent_disagreement = np.linalg.norm(ref0-independent_ref, axis=1)
    if independent_disagreement.max() > 1e-8:
        raise RuntimeError('Solvers de referência discordam acima de 1e-8')

    ef0, _ = forward_euler(problem.gradient, x0, h, N)
    ef1, _ = forward_euler(problem.gradient, x0_other, h, N)
    eb0, info0 = backward_euler(problem.gradient, problem.hessian, x0, h, N,
                                inner_tol=1e-12)
    eb1, info1 = backward_euler(problem.gradient, problem.hessian, x0_other, h, N,
                                inner_tol=1e-12)

    epsilon = 1e-4
    direction = np.array([1.0, -1.0])/np.sqrt(2.0)
    ref_pert = integrate(x0+epsilon*direction)

    def augmented_rhs(tt, z):
        xx = z[:2]
        xi = z[2:]
        return np.r_[-problem.gradient(xx), -problem.hessian(xx) @ xi]

    varsol = solve_ivp(augmented_rhs, (0, T), np.r_[x0, direction],
                       t_eval=t, method='Radau', rtol=1e-11, atol=1e-13)
    if not varsol.success:
        raise RuntimeError('Equação variacional: '+varsol.message)
    xi = varsol.y[2:,:].T
    derivative_fd = (ref_pert-ref0)/epsilon
    variational_mismatch = np.linalg.norm(derivative_fd-xi, axis=1)
    if variational_mismatch.max() > 2e-3:
        raise RuntimeError('Equação variacional não reproduz a derivada do fluxo')

    trajectories = {'Fluxo':(ref0, ref1), 'EF':(ef0,ef1), 'EB':(eb0,eb1)}
    bounds = {
        'Fluxo':np.linalg.norm(delta0)*np.exp(-problem.m*t),
        'EF':np.linalg.norm(delta0)*q_ef**np.arange(N+1),
        'EB':np.linalg.norm(delta0)*q_eb**np.arange(N+1),
    }
    rows=[]
    for name,(xa,xb) in trajectories.items():
        separations=np.linalg.norm(xa-xb,axis=1)
        if np.any(separations > bounds[name] + 5e-9):
            raise AssertionError(f'Limite de contração violado por {name}')
        for k in range(N+1):
            error_to_flow=(0.0 if name=='Fluxo' else float(np.linalg.norm(xa[k]-ref0[k])))
            rows.append(dict(metodo=name, k=k, t=float(t[k]), h=h, kappa=problem.kappa,
                distancia_entre_trajetorias=float(separations[k]),
                limite_teorico_contracao=float(bounds[name][k]),
                erro_ao_fluxo_mesmo_t=error_to_flow,
                discrepancia_referencias=float(independent_disagreement[k])))

    fig, axes = plt.subplots(1,3,figsize=(15.0,4.0))
    colors={'Fluxo':'#222222','EF':'#2371ad','EB':'#c87321'}
    for name,(xa,xb) in trajectories.items():
        dist=np.linalg.norm(xa-xb,axis=1)
        axes[0].semilogy(t,dist,color=colors[name],lw=1.9,label=name)
        axes[0].semilogy(t,bounds[name],color=colors[name],ls=':',lw=1.,alpha=.65)
    axes[0].set(xlabel='$t=kh$',ylabel=r'$\|x(t;a)-x(t;b)\|$',
                title='Sensibilidade a condição inicial')
    axes[0].grid(alpha=.18,which='both'); axes[0].legend(fontsize=8)
    for name,X in [('EF',ef0),('EB',eb0)]:
        err=np.linalg.norm(X-ref0,axis=1)
        axes[1].semilogy(t[1:],np.maximum(err[1:],1e-16),color=colors[name],lw=1.8,label=name)
    axes[1].set(xlabel='$t=kh$',ylabel=r'$\|x_k-x_{ref}(t_k)\|$',
                title='Fidelidade temporal (não otimalidade)')
    axes[1].grid(alpha=.18,which='both'); axes[1].legend(fontsize=8)
    axes[2].plot(t,np.linalg.norm(xi,axis=1),lw=2,color='#222222',label='Equação variacional')
    axes[2].plot(t,np.linalg.norm(derivative_fd,axis=1),ls='--',lw=1.6,
                 color='#7b4a9d',label='Diferença finita do fluxo')
    axes[2].set(xlabel='$t$',ylabel=r'$\|D\Phi_t(x_0)v\|$',
                title='Sensibilidade linearizada')
    axes[2].grid(alpha=.18); axes[2].legend(fontsize=7.8)
    fig.tight_layout()
    png(fig,out['figures']/'auxiliares/trajetorias_perturbacoes.png')
    csv_write(out['tables']/'auxiliares/trajetorias_perturbacoes.csv',rows)
    csv_write(out['tables']/'auxiliares/equacao_variacional.csv',[
        dict(t=float(t[k]), xi_1=float(xi[k,0]),xi_2=float(xi[k,1]),
             diferenca_finita_1=float(derivative_fd[k,0]),
             diferenca_finita_2=float(derivative_fd[k,1]),
             discrepancia=float(variational_mismatch[k]))
        for k in range(N+1)])
    json_write(out['logs']/'auxiliares/perturbacoes_config.json',dict(
        kappa=problem.kappa,m=problem.m,L=problem.L,
        x0=x0.tolist(),x0_perturbado=x0_other.tolist(),
        epsilon_diferenca_finita=epsilon,direcao=direction.tolist(),
        h=h,N=N,T=T,q_ef=q_ef,q_eb=q_eb,
        max_diferenca_referencias=float(independent_disagreement.max()),
        max_discrepancia_variacional=float(variational_mismatch.max()),
        max_residuo_EB=float(max(info0['max_implicit_residual'],info1['max_implicit_residual'])),
        solvers_referencia=['Radau','DOP853'],rtol=1e-11,atol=1e-13,
        nota='Fluxo de referência aproximado numericamente; erro no mesmo T não é progresso de otimização.'))
    return rows


def experiment_implicit_residual(*, piloto=False, root=None):
    """Certificação do erro do passo implícito por monotonicidade forte.

    Perturbações controladas de uma solução numérica de referência do passo,
    mais um passo realmente computado pelo Newton interno.
    """
    out = folders('exp2_nonquadratic',root=root)
    p=StronglyConvex(kappa=100.)
    x0=np.array([2.0,1.0])
    h=0.5
    # A referência NÃO é declarada exata: seu erro é certificado via resíduo.
    ref, _ = backward_euler(p.gradient,p.hessian,x0,h,1,
                           inner_tol=1e-13,max_newton=25)
    yref=ref[-1]
    rref=float(np.linalg.norm(yref-x0+h*p.gradient(yref)))
    v=np.array([1.0,1.0])/np.sqrt(2.0)
    magnitudes=(1e-1,1e-2,1e-3,1e-4) if piloto else (1e-1,1e-2,1e-3,1e-4,1e-5)
    candidates=[(f'perturbacao_{delta:g}', yref+delta*v,float(delta)) for delta in magnitudes]
    actual,_=backward_euler(p.gradient,p.hessian,x0,h,1,inner_tol=1e-5,max_newton=25)
    candidates.append(('Newton_interno_tol_1e-5',actual[-1],float('nan')))
    rows=[]
    for name,y,delta in candidates:
        residual=float(np.linalg.norm(y-x0+h*p.gradient(y)))
        deviation=float(np.linalg.norm(y-yref))
        bound=residual/(1+h*p.m)
        # Limite entre duas aproximações, com ambos os resíduos considerados.
        validated_bound=(residual+rref)/(1+h*p.m)
        if deviation>validated_bound+2e-12:
            raise AssertionError('Falhou certificação de resíduo implícito')
        rows.append(dict(candidato=name,perturbacao_aplicada=delta,
                         h=h,kappa=p.kappa,m=p.m,
                         residuo_equacao_implicita=residual,
                         desvio_referencia_numerica=deviation,
                         limite_para_raiz_exata=bound,
                         residuo_referencia=rref,
                         limite_comparacao_duas_aproximacoes=validated_bound,
                         limite_respeitado=True))
    csv_write(out['tables']/'auxiliares/residuo_implicito.csv',rows)
    data=[r for r in rows if r['candidato'].startswith('perturbacao_')]
    fig,ax=plt.subplots(figsize=(7.5,4.8))
    ax.loglog([r['residuo_equacao_implicita'] for r in data],
              [r['desvio_referencia_numerica'] for r in data],
              'o-',color='#2371ad',lw=1.7,label='Distância à referência numérica')
    ax.loglog([r['residuo_equacao_implicita'] for r in data],
              [r['limite_comparacao_duas_aproximacoes'] for r in data],
              's--',color='#b36b20',lw=1.7,
              label=r'$(\|r\|+\|r_{ref}\|)/(1+hm)$')
    ax.set(xlabel=r'Resíduo implícito $\|r\|$',
           ylabel='Erro / limite certificado',
           title='Robustez do passo EB à resolução interna')
    ax.grid(alpha=.2,which='both');ax.legend(fontsize=9)
    fig.tight_layout()
    png(fig,out['figures']/'auxiliares/residuo_implicito.png')
    json_write(out['logs']/'auxiliares/residuo_implicito_config.json',dict(kappa=p.kappa,m=p.m,h=h,
        x0=x0.tolist(),referencia=yref.tolist(),residuo_referencia=rref,
        nota='Referência é solução aproximada; critério validado usa soma dos resíduos.'))
    return rows



# ===========================================================================
# INVESTIGAÇÃO 2.2: FAMÍLIA NÃO QUADRÁTICA ACOPLADA
# ===========================================================================


def _coupled_reference(problem, initial, horizon, method='Radau'):
    """Solução numérica de referência, não solução analítica.

    Mantemos a interpolação densa do integrador para avaliar todos os métodos
    nos mesmos tempos, sem interpolar linearmente os iterados EF/EB.
    """
    kwargs = dict(
        fun=lambda t, x: -problem.gradient(x),
        t_span=(0., float(horizon)), y0=np.asarray(initial, dtype=float),
        method=method, rtol=1e-11, atol=1e-13, dense_output=True,
    )
    if method == 'Radau':
        kwargs['jac'] = lambda t, x: -problem.hessian(x)
    sol = solve_ivp(**kwargs)
    if not sol.success or sol.sol is None:
        raise RuntimeError(f'Falha na referência {method}: {sol.message}')
    return sol


def _coupled_reference_check(problem, initial, horizon, times):
    """Compara dois integradores independentes para a mesma condição inicial."""
    radau = _coupled_reference(problem, initial, horizon, 'Radau')
    dop = _coupled_reference(problem, initial, horizon, 'DOP853')
    discrepancy = np.linalg.norm(radau.sol(times)-dop.sol(times), axis=0)
    maximum = float(np.max(discrepancy))
    if not np.isfinite(maximum) or maximum > 1e-8:
        raise RuntimeError(f'Referências Radau/DOP853 discordam: {maximum:.3e}')
    return radau, maximum


def _coupled_steps(problem, method, x0, h, n):
    """Executa EF/EB sem parada antecipada, preservando t_k = kh."""
    trial = _trajectory(problem, method, x0, h, n_steps=n)
    if trial['X'] is None or trial['status'] != 'Fixed Steps':
        raise RuntimeError(f'{method}: falha inesperada: {trial["falha"] or trial["status"]}')
    if len(trial['X']) != n+1:
        raise RuntimeError(f'{method}: trajetória com comprimento inesperado')
    return trial['X'], trial['info']


def _slow_angle_deg(H):
    """Orientação do autovetor de menor autovalor, com sinal fixado em v[0]>0."""
    _, eigvecs = np.linalg.eigh(H)
    v = eigvecs[:, 0]
    if v[0] < 0:
        v = -v
    return float(np.degrees(np.arctan2(v[1], v[0])))


def experiment_coupled_trajectories(*, piloto=False, root=None):
    """Varia o passo sem variar a função, x0 e o tempo final T=3.

    Dois painéis por passo: em cada painel o fluxo (Radau) e EF/EB são
    comparados geometricamente. Referência DOP853 confere Radau.
    """
    out = folders('exp2_nonquadratic', root=root)
    problem = CoupledNonQuadratic(kappa=2., epsilon=.35)
    x0 = np.array([1.2, .8])
    T = 3.0
    regimes = ((.15, 20), (.60, 5))
    dense_time = np.linspace(0., T, 501)
    ref, max_disagreement = _coupled_reference_check(problem, x0, T, dense_time)
    ref_dense = ref.sol(dense_time).T

    grid = np.linspace(-.5, 1.5, 320)
    X, Y = np.meshgrid(grid, grid)
    S = X+Y
    Z = .5*(X**2 + problem.kappa*Y**2) + problem.epsilon*(np.sin(S)-S)
    levels = np.geomspace(.008, 2.5, 13)
    rows, summary, angles = [], [], []
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.9), sharex=True, sharey=True)
    colors = {'EF':'#2166ad', 'EB':'#d27a1d'}
    all_handles = None

    for j, (ax, (h, n)) in enumerate(zip(axes, regimes)):
        if not (0 < h < 2/problem.L and np.isclose(h*n, T)):
            raise ValueError('Passos precisam respeitar a faixa estável e T fixo')
        tk = h*np.arange(n+1)
        reference_nodes = ref.sol(tk).T
        ax.contour(X, Y, Z, levels=levels, colors='.74',
                   linewidths=.75, alpha=.85, zorder=1)
        continuous, = ax.plot(ref_dense[:,0], ref_dense[:,1],
                              color='#242424', lw=2.0, label='Fluxo gradiente (referência)',
                              zorder=3)
        handles = [continuous]
        for method, marker in (('EF', 'o'), ('EB', 's')):
            traj, info = _coupled_steps(problem, method, x0, h, n)
            line, = ax.plot(traj[:,0], traj[:,1], marker+'-',
                            color=colors[method], lw=1.45, ms=4.3,
                            label='Euler forward' if method=='EF' else 'Euler backward',
                            zorder=4 if method=='EF' else 5)
            handles.append(line)
            error = np.linalg.norm(traj-reference_nodes,axis=1)
            for k, (x, e) in enumerate(zip(traj, error)):
                rows.append(dict(cenario=f'h={h:g}', metodo=method,
                                 h=float(h), N=n, T=T, k=k, t=float(tk[k]),
                                 x=float(x[0]), y=float(x[1]),
                                 erro_ao_fluxo=float(e)))
            summary.append(dict(cenario=f'h={h:g}', metodo=method,
                                h=h, N=n, T=T, status=info['status'],
                                erro_final=float(error[-1]),
                                maior_erro_na_malha=float(error.max()),
                                max_residuo_implicito=float(info['max_implicit_residual'])))
        ini, = ax.plot(*x0, '*', color='black', ms=9, zorder=7,
                       label='Condição inicial')
        minimum, = ax.plot(0,0,'x',color='black',ms=8,mew=1.7,
                           label='Minimizador',zorder=7)
        handles.extend((ini,minimum))
        if j == 0:
            all_handles = handles
        ax.set(title=rf'({chr(97+j)}) Passo $h={h:g}$, $N={n}$',
               xlim=(-.5,1.5),ylim=(-.5,1.5))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=.12)
        ax.spines[['top','right']].set_visible(False)

    for i,(tt, x) in enumerate(zip(dense_time,ref_dense)):
        angle=_slow_angle_deg(problem.hessian(x))
        angles.append(dict(t=float(tt), x=float(x[0]), y=float(x[1]),
                           hess_misto=float(problem.hessian(x)[0,1]),
                           angulo_modo_lento_graus=angle))

    fig.suptitle('Função acoplada: trajetórias contínuas e discretas',
                 fontsize=14, y=.965)
    fig.text(.5,.907,rf'$\kappa={problem.kappa:g}$, '
             rf'$\varepsilon={problem.epsilon:g}$, ' r'$x_0=(1{,}2;\ 0{,}8)$, '
             rf'$T={T:g}$; ambos os passos satisfazem $h<2/L$',
             ha='center',fontsize=10)
    fig.supxlabel(r'Coordenada $x_1$',y=.142)
    fig.supylabel(r'Coordenada $x_2$',x=.027)
    fig.legend(all_handles,[h.get_label() for h in all_handles],
               ncol=3,loc='lower center',bbox_to_anchor=(.5,.023),
               frameon=False,fontsize=9)
    fig.subplots_adjust(left=.095,right=.95,top=.805,bottom=.245,wspace=.19)
    png(fig,out['figures']/'03_acoplada_trajetorias.png')

    csv_write(out['tables']/'03_acoplada_trajetorias.csv',rows)
    csv_write(out['tables']/'03_acoplada_resumo.csv',summary)
    csv_write(out['tables']/'03_acoplada_hessiana_ao_longo_fluxo.csv',angles)
    json_write(out['logs']/'03_acoplada_trajetorias_config.json',dict(
        problema='CoupledNonQuadratic',kappa=problem.kappa,
        epsilon=problem.epsilon,m=problem.m,L=problem.L,
        x0=x0.tolist(),T=T,regimes=[{'h':h,'N':n} for h,n in regimes],
        solvers_referencia=['Radau','DOP853'],rtol=1e-11,atol=1e-13,
        max_discrepancia_referencias=max_disagreement,
        angulo_inicial_graus=angles[0]['angulo_modo_lento_graus'],
        angulo_final_graus=angles[-1]['angulo_modo_lento_graus'],
        nota='Erros calculados nos mesmos instantes t_k; passos diferentes, mesmo T. '
             'Referência contínua é numérica, não exata.'))
    return summary


def experiment_coupled_sensitivity(*, piloto=False, root=None):
    """Duas condições iniciais + derivada cruzada do fluxo.

    A quantidade D Phi_t(x0)[e1]_2 detecta efeito de x0,1 sobre x2(t).
    Para o problema separável essa derivada cruzada é identicamente zero.
    """
    out = folders('exp2_nonquadratic', root=root)
    problem = CoupledNonQuadratic(kappa=2., epsilon=.35)
    a=np.array([1.2,.8])
    delta=np.array([.08,-.06])
    b=a+delta
    e1=np.array([1.,0.])
    eps_fd=1e-5
    h=.30
    n=10
    T=n*h
    t=h*np.arange(n+1)
    t_dense=np.linspace(0,T,301)

    sol_a, difference_a = _coupled_reference_check(problem,a,T,t_dense)
    sol_b, difference_b = _coupled_reference_check(problem,b,T,t_dense)
    sol_fd = _coupled_reference(problem,a+eps_fd*e1,T)
    ref_a=sol_a.sol(t).T
    ref_b=sol_b.sol(t).T
    refs={'Fluxo':(ref_a,ref_b)}
    residuals=[]
    for method in ('EF','EB'):
        Xa, info_a=_coupled_steps(problem,method,a,h,n)
        Xb, info_b=_coupled_steps(problem,method,b,h,n)
        refs[method]=(Xa,Xb)
        residuals.append(float(max(info_a['max_implicit_residual'],
                                   info_b['max_implicit_residual'])))

    def variational_rhs(tt,z):
        x=z[:2]
        xi=z[2:]
        return np.r_[-problem.gradient(x),-problem.hessian(x)@xi]

    variational=solve_ivp(variational_rhs,(0.,T),np.r_[a,e1],
                          method='Radau',dense_output=True,
                          rtol=1e-11,atol=1e-13)
    if not variational.success or variational.sol is None:
        raise RuntimeError('Falha da equação variacional: '+variational.message)
    xi=variational.sol(t_dense)[2:,:].T
    fd=(sol_fd.sol(t_dense).T-sol_a.sol(t_dense).T)/eps_fd
    discrepancy=np.linalg.norm(xi-fd,axis=1)
    if np.max(discrepancy)>2e-4:
        raise RuntimeError('Sensibilidade por equação variacional não confere com diferença finita')

    k=np.arange(n+1)
    qef=max(abs(1-h*problem.m),abs(1-h*problem.L))
    qeb=1/(1+h*problem.m)
    bounds={'Fluxo':np.linalg.norm(delta)*np.exp(-problem.m*t),
            'EF':np.linalg.norm(delta)*qef**k,
            'EB':np.linalg.norm(delta)*qeb**k}
    colors={'Fluxo':'#222222','EF':'#2166ad','EB':'#d27a1d'}
    rows=[]
    for method,(Xa,Xb) in refs.items():
        d=np.linalg.norm(Xa-Xb,axis=1)
        if np.any(d>bounds[method]+5e-9):
            raise AssertionError(f'Limite global de contração não confirmado: {method}')
        for j in range(n+1):
            rows.append(dict(metodo=method,k=int(j),t=float(t[j]),h=h,
                             distancia_entre_trajetorias=float(d[j]),
                             limite_teorico=float(bounds[method][j]),
                             erro_ao_fluxo_a=(0. if method=='Fluxo' else
                                float(np.linalg.norm(Xa[j]-ref_a[j])))))
    crossing=[dict(t=float(tt), derivada_cruzada_variacional=float(x[1]),
                   derivada_cruzada_diferenca_finita=float(y[1]),
                   discrepancia=float(e))
              for tt,x,y,e in zip(t_dense,xi,fd,discrepancy)]

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12.,4.9))
    for method,(Xa,Xb) in refs.items():
        ax1.semilogy(t,np.linalg.norm(Xa-Xb,axis=1),
                     'o-' if method!='Fluxo' else '-',
                     ms=4,lw=1.8,color=colors[method],label=method)
    ax1.semilogy(t,bounds['Fluxo'],':',color='.43',lw=1.3,
                 label=r'Limite contínuo $e^{-mt}\|a-b\|$')
    ax1.set(xlabel=r'Tempo $t=kh$',
            ylabel=r'$\|\Phi_t(a)-\Phi_t(b)\|$ (ou iterados)',
            title='(a) Sensibilidade às condições iniciais')
    ax1.grid(alpha=.20,which='both')
    ax1.legend(fontsize=8.5)

    ax2.plot(t_dense,xi[:,1],color='#673aa6',lw=1.9,
             label=r'Equação variacional $\partial x_2(t)/\partial x_{0,1}$')
    ax2.plot(t_dense[::15],fd[::15,1],'o',color='#be7127',ms=3.4,
             label='Diferença finita do fluxo')
    ax2.axhline(0.,color='.4',lw=1.2,ls=':',
                label='Caso separável: derivada cruzada zero')
    ax2.set(xlabel='Tempo $t$',ylabel='Sensibilidade cruzada',
            title='(b) Acoplamento de perturbações')
    ax2.grid(alpha=.20)
    ax2.legend(fontsize=8.0)

    fig.suptitle('Função acoplada: contração e influência entre coordenadas',
                 fontsize=13.4,y=.99)
    fig.subplots_adjust(left=.10,right=.97,bottom=.16,top=.84,wspace=.30)
    png(fig,out['figures']/'04_acoplada_sensibilidade.png')
    csv_write(out['tables']/'04_acoplada_contracao.csv',rows)
    csv_write(out['tables']/'04_acoplada_derivada_cruzada.csv',crossing)
    json_write(out['logs']/'04_acoplada_sensibilidade_config.json',dict(
        problema='CoupledNonQuadratic',kappa=problem.kappa,
        epsilon=problem.epsilon,m=problem.m,L=problem.L,
        a=a.tolist(),b=b.tolist(),h=h,N=n,T=T,
        direcao_diferenca_finita=e1.tolist(),eps_fd=eps_fd,
        qef=qef,qeb=qeb,max_erro_variacional=float(np.max(discrepancy)),
        max_sensibilidade_cruzada=float(np.max(np.abs(xi[:,1]))),
        max_discrepancia_referencias=float(max(difference_a,difference_b)),
        max_residuo_EB=float(max(residuals)),
        solvers_referencia=['Radau','DOP853'],rtol=1e-11,atol=1e-13,
        nota='Sensibilidade cruzada e contração são propriedades distintas; '
             'curva contínua é referência numérica de alta precisão.'))
    return rows


def main(analise='principais', piloto=False):
    """Duas investigações principais; diagnósticos antigos seguem opcionais.

    principais: separável (convergência/dissipação); acoplada
                (trajetórias/sensibilidade).
    todos: inclui também robustez, precisão e diagnósticos anteriores.
    """
    essential=('convergencia','dissipacao','acoplada_trajetorias','acoplada_sensibilidade')
    optional=('robustez','precisao','trajetorias','residuo')
    selected = (essential if analise=='principais' else
                ('acoplada_trajetorias','acoplada_sensibilidade') if analise=='acoplada' else
                ('convergencia','dissipacao') if analise=='separavel' else
                ('trajetorias','residuo') if analise=='aprofundamento' else
                essential+optional if analise=='todos' else (analise,))
    runs={
        'convergencia':lambda:experiment_convergence(
            kappas=(10,) if piloto else (10,100),
            max_steps=4500 if piloto else 12000,
            mostrar_limite_auxiliar=(analise in ('convergencia','todos'))),
        'dissipacao':lambda:experiment_dissipation(piloto=piloto),
        'acoplada_trajetorias':lambda:experiment_coupled_trajectories(piloto=piloto),
        'acoplada_sensibilidade':lambda:experiment_coupled_sensitivity(piloto=piloto),
        'robustez':lambda:experiment_robustness(piloto=piloto),
        'precisao':lambda:experiment_accuracy(piloto=piloto),
        'trajetorias':lambda:experiment_trajectory_robustness(piloto=piloto),
        'residuo':lambda:experiment_implicit_residual(piloto=piloto),
    }
    for name in selected:
        result = runs[name]()
        print(f'Módulo 2 — {name}: {len(result)} registros')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analise', choices=(
        'principais','todos','separavel','acoplada',
        'convergencia','dissipacao','acoplada_trajetorias',
        'acoplada_sensibilidade','robustez','precisao',
        'aprofundamento','trajetorias','residuo'),default='principais')
    parser.add_argument('--piloto', action='store_true',
                        help='reduz apenas os ensaios extensos da função separável')
    args=parser.parse_args()
    main(args.analise,args.piloto)
