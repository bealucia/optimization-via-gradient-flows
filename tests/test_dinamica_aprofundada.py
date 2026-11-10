"""Testes das proposições novas e das saídas correspondentes.

Referências independentes: fórmulas fechadas, problemas lineares e limites analíticos.
"""
import numpy as np
import pytest

from src.problems.strongly_convex import StronglyConvex
from src.problems.quadratic import quadratic_matrix
from src.problems import saddle
from src.methods.euler import forward_euler, backward_euler
from src.methods.newton import newton_damped
from src.experiments.exp2_nonquadratic import (
    experiment_trajectory_robustness, experiment_implicit_residual)
from src.experiments.exp4_newton_saddle import experiment_saddle


def test_saddle_is_not_minimizer_and_newton_flow_is_exact():
    x0 = np.array([0.5, 1.0])
    times = np.array([0., .5, 1., 2.])
    assert saddle.value(np.zeros(2)) == 0
    assert saddle.value(np.array([0., 2.])) < saddle.value(np.zeros(2))
    np.testing.assert_allclose(np.linalg.eigvalsh(saddle.hessian(x0)), [-1,1])
    continuous = saddle.exact_newton_flow(x0, times)
    np.testing.assert_allclose(continuous, np.exp(-times)[:,None]*x0[None,:])
    np.testing.assert_allclose(np.array([saddle.gradient(x) for x in continuous]),
                               np.exp(-times)[:,None]*saddle.gradient(x0)[None,:])
    assert saddle.value(continuous[-1]) > saddle.value(x0)


def test_saddle_gradient_flow_is_unstable_and_newton_is_attractive():
    x0 = np.array([0.5,1.0])
    reference = saddle.exact_gradient_flow(x0,[0.,1.,2.])
    assert abs(reference[-1,1]) > abs(reference[0,1])
    N=9
    h=.2
    X,_=forward_euler(saddle.gradient,x0,h,N)
    Y=newton_damped(saddle.gradient,saddle.hessian,x0,h,N)
    k=np.arange(N+1)
    np.testing.assert_allclose(X, np.column_stack((x0[0]*(1-h)**k,x0[1]*(1+h)**k)))
    np.testing.assert_allclose(Y, (1-h)**k[:,None]*x0[None,:])


@pytest.mark.parametrize('h,contractive',[(.5,True),(1.,True),(1.5,True),(2.,False),(2.1,False)])
def test_newton_amortecido_fator_e_fronteira(h,contractive):
    x0=np.array([.5,1.])
    y=newton_damped(saddle.gradient,saddle.hessian,x0,h,6)
    expected=np.linalg.norm(x0)*abs(1-h)**6
    np.testing.assert_allclose(np.linalg.norm(y[-1]),expected,rtol=1e-12,atol=1e-12)
    assert (expected<np.linalg.norm(x0)) == contractive


def test_newton_rejects_singular_hessian():
    with pytest.raises(np.linalg.LinAlgError):
        newton_damped(lambda x:np.array([1.,1.]),
                      lambda x:np.diag([1.,0.]),np.array([2.,3.]),1.,1)


def test_one_step_residual_bound_for_nonlinear_proximal():
    p=StronglyConvex(100)
    x0=np.array([2.,1.]);h=.5
    X,_=backward_euler(p.gradient,p.hessian,x0,h,1,inner_tol=1e-13)
    y_exact_ref=X[-1]
    Y,_=backward_euler(p.gradient,p.hessian,x0,h,1,inner_tol=1e-4)
    y_rough=Y[-1]
    ra=np.linalg.norm(y_rough-x0+h*p.gradient(y_rough))
    rr=np.linalg.norm(y_exact_ref-x0+h*p.gradient(y_exact_ref))
    assert np.linalg.norm(y_rough-y_exact_ref)<=(ra+rr)/(1+h*p.m)+1e-12


def test_ef_eb_as_exacts_flows_of_modified_quadratic():
    A=quadratic_matrix(20.)
    vals,V=np.linalg.eigh(A)
    h=.02 # 0<h<1/L
    for scheme in ('EF','EB'):
        if scheme=='EF':
            rates=-np.log(1-h*vals)/h
            step=np.eye(2)-h*A
        else:
            rates=np.log1p(h*vals)/h
            step=np.linalg.solve(np.eye(2)+h*A,np.eye(2))
        exact_modified=V @ np.diag(np.exp(-h*rates)) @ V.T
        np.testing.assert_allclose(exact_modified,step,atol=1e-12)
        if scheme=='EF':
            assert np.all(rates>vals)
        else:
            assert np.all(rates<vals)


def test_experimentos_novos_gravam_saida_em_pasta_temporaria(tmp_path):
    a=experiment_trajectory_robustness(piloto=True,root=tmp_path)
    b=experiment_implicit_residual(piloto=True,root=tmp_path)
    c,d=experiment_saddle(piloto=True,root=tmp_path)
    assert len(a)>100 and len(b)==5
    assert len(c)>50 and len(d)>80
    for path in [
        tmp_path/'exp2_nonquadratic/figures/06_trajetorias_perturbacoes.png',
        tmp_path/'exp2_nonquadratic/figures/07_residuo_implicito.png',
        tmp_path/'exp2_nonquadratic/tables/06_equacao_variacional.csv',
        tmp_path/'exp4_newton_saddle/figures/01_sela_fluxos_e_discretizacoes.png',
        tmp_path/'exp4_newton_saddle/tables/02_estabilidade_newton.csv',
    ]:
        assert path.is_file() and path.stat().st_size>100


def test_desigualdade_variacional_discreta_no_proximal():
    p=StronglyConvex(25.)
    x0=np.array([1.7,-0.4]);h=.25
    X,_=backward_euler(p.gradient,p.hessian,x0,h,4,inner_tol=1e-13)
    z=np.array([.1,.2]) # z arbitrário, não minimizador
    for prev,nxt in zip(X[:-1],X[1:]):
        evi=((np.linalg.norm(nxt-z)**2 - np.linalg.norm(prev-z)**2)/(2*h)
             + p.value(nxt)-p.value(z)
             + np.linalg.norm(nxt-prev)**2/(2*h))
        assert evi <= 2e-11


def test_taxas_globais_contrativas_ef_eb_com_mesmo_h():
    p=StronglyConvex(20.)
    h=.04
    a=np.array([1.1,-.7]);b=np.array([1.0,-.65])
    xa,_=forward_euler(p.gradient,a,h,12)
    xb,_=forward_euler(p.gradient,b,h,12)
    ya,_=backward_euler(p.gradient,p.hessian,a,h,12)
    yb,_=backward_euler(p.gradient,p.hessian,b,h,12)
    q_exp=max(abs(1-h*p.m),abs(1-h*p.L))
    q_imp=1/(1+h*p.m)
    initial=np.linalg.norm(a-b)
    for k in range(13):
        assert np.linalg.norm(xa[k]-xb[k]) <= initial*q_exp**k+1e-11
        assert np.linalg.norm(ya[k]-yb[k]) <= initial*q_imp**k+1e-11
