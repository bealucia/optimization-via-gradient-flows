"""Verificações independentes da função acoplada e das duas novas análises.

Referências: diferenças finitas, limites espectrais, integração independente
Radau/DOP853 e diretórios temporários, sem necessidade de dados pré-gerados.
"""

import csv
import json

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.problems.coupled_nonquadratic import CoupledNonQuadratic
from src.problems.strongly_convex import StronglyConvex
from src.methods.euler import forward_euler, backward_euler
from src.experiments.exp2_nonquadratic import (
    experiment_coupled_trajectories,
    experiment_coupled_sensitivity,
    _coupled_reference_check,
)


@pytest.mark.parametrize('kappa,epsilon', [(1, .1), (2, .35), (10, .25)])
def test_coupled_gradient_hessian_by_finite_differences(kappa, epsilon):
    p = CoupledNonQuadratic(kappa, epsilon)
    x = np.array([.7, -.35])
    step = 2e-6
    basis = np.eye(2)
    for j in range(2):
        d = basis[j]
        g_fd = (p.value(x+step*d)-p.value(x-step*d))/(2*step)
        H_fd = (p.gradient(x+step*d)-p.gradient(x-step*d))/(2*step)
        np.testing.assert_allclose(p.gradient(x)[j], g_fd, atol=5e-9)
        np.testing.assert_allclose(p.hessian(x)[:,j], H_fd, atol=5e-9)
    np.testing.assert_allclose(p.hessian(x),p.hessian(x).T,atol=1e-15)


@pytest.mark.parametrize('kappa,eps', [(2, .35), (10, .25), (100, .40)])
def test_hessian_uniform_bounds_and_minimizer(kappa, eps):
    p = CoupledNonQuadratic(kappa, eps)
    np.testing.assert_allclose(p.gradient(p.minimizer), np.zeros(2))
    assert np.isclose(p.value(p.minimizer), 0)
    np.testing.assert_allclose(np.linalg.eigvalsh(p.hessian(p.minimizer)), [1,kappa])
    for x in ([0,0], [1.2,.8], [-2,.3], [-1., 4.], [10,-15]):
        eigs=np.linalg.eigvalsh(p.hessian(x))
        assert eigs[0] >= p.m-1e-12
        assert eigs[-1] <= p.L+1e-12
        assert p.value(x) >= .5*p.m*np.dot(x,x)-1e-12


@pytest.mark.parametrize('args', [dict(kappa=.5),dict(kappa=float('nan')),
                                  dict(epsilon=0),dict(epsilon=.5),
                                  dict(epsilon=float('inf'))])
def test_invalid_parameters_rejected(args):
    with pytest.raises(ValueError):
        CoupledNonQuadratic(**args)


def test_true_nonlinear_coupling_and_rotating_directions():
    p=CoupledNonQuadratic(2.,.35)
    H0=p.hessian(np.zeros(2))
    H1=p.hessian([1.2,.8])
    assert H0[0,1]==0
    assert abs(H1[0,1])>.25
    _,V=np.linalg.eigh(H1)
    assert abs(V[1,0])>0.1
    # No problema separável, o termo misto é zero em todo ponto.
    sep=StronglyConvex(2.,.25)
    assert sep.hessian([1.2,.8])[0,1]==0


def test_ef_eb_dissipation_and_discrete_bounds():
    p=CoupledNonQuadratic(2.,.35)
    a=np.array([1.2,.8]);b=a+np.array([.08,-.06])
    h=.3;n=10
    qef=max(abs(1-h*p.m),abs(1-h*p.L))
    qeb=1/(1+h*p.m)
    assert 0<qef<1 and 0<qeb<1
    for integrator, bound in (
        (lambda x:forward_euler(p.gradient,x,h,n)[0],qef),
        (lambda x:backward_euler(p.gradient,p.hessian,x,h,n)[0],qeb),
    ):
        X=integrator(a);Y=integrator(b)
        f=np.array([p.value(v) for v in X])
        assert np.max(np.diff(f))<=1e-11
        ds=np.linalg.norm(X-Y,axis=1)
        assert np.all(ds <= np.linalg.norm(a-b)*bound**np.arange(n+1)+5e-10)


def test_reference_solvers_agree_and_cross_sensitivity_nonzero():
    p=CoupledNonQuadratic(2.,.35)
    a=np.array([1.2,.8]);T=2.
    t=np.linspace(0,T,41)
    sol, disagreement=_coupled_reference_check(p,a,T,t)
    assert disagreement<1e-8
    eps=1e-5
    b=a+eps*np.array([1.,0.])
    pert=solve_ivp(lambda _,x:-p.gradient(x),(0,T),b,
                   method='DOP853',t_eval=t,rtol=1e-11,atol=1e-13)
    assert pert.success
    cross=(pert.y[1]-sol.sol(t)[1])/eps
    assert abs(cross[10])>0.03


def test_coupled_analyses_are_reproducible_without_previous_results(tmp_path):
    a=experiment_coupled_trajectories(root=tmp_path,piloto=True)
    b=experiment_coupled_sensitivity(root=tmp_path,piloto=True)
    assert len(a)==4 and len(b)==33
    base=tmp_path/'exp2_nonquadratic'
    for file in (
        'figures/08_acoplada_trajetorias.png',
        'figures/09_acoplada_sensibilidade.png',
        'tables/08_acoplada_trajetorias.csv',
        'tables/08_acoplada_resumo.csv',
        'tables/08_acoplada_hessiana_ao_longo_fluxo.csv',
        'tables/09_acoplada_contracao.csv',
        'tables/09_acoplada_derivada_cruzada.csv',
        'logs/08_acoplada_trajetorias_config.json',
        'logs/09_acoplada_sensibilidade_config.json',
    ):
        path=base/file
        assert path.exists() and path.stat().st_size>100
    with (base/'tables/08_acoplada_resumo.csv').open(encoding='utf8') as fp:
        summary=list(csv.DictReader(fp))
    assert {float(row['h']) for row in summary}=={.15,.6}
    assert all(float(row['T'])==3 for row in summary)
    assert all(float(row['erro_final'])>=0 for row in summary)
    with (base/'logs/09_acoplada_sensibilidade_config.json').open(encoding='utf8') as fp:
        cfg=json.load(fp)
    assert cfg['max_erro_variacional']<2e-4
    assert cfg['max_sensibilidade_cruzada']>0.05
    assert cfg['max_discrepancia_referencias']<1e-8
