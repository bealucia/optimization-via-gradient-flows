"""Contrato do benchmark: repetições, custos e consistência sem ranking de desempenho."""
import numpy as np

from src.problems.least_squares import make_problem
from src.experiments.exp3_performance import measure_case


def test_benchmark_records_repeats_and_consistent_total():
    p = make_problem(8,10,seed=4)
    summary, samples = measure_case(p,10,8,'Cholesky',1.,1e-6,1000,3)
    assert len(samples)==3
    assert summary['status']=='Converged'
    assert summary['iteracoes']==20
    assert summary['repeticoes']==3
    assert np.isfinite(summary['total_mediana_s'])
    assert summary['total_mediana_s'] >= 0
    for sample in samples:
        assert sample['status']=='Converged'
        assert sample['iteracoes']==20
        assert np.isclose(sample['total_s'],sample['fatoracao_s']+sample['lacos_s'],rtol=1e-12,atol=1e-12)
        assert sample['residuo_relativo_ultimo_passo'] < 1e-10


def test_benchmark_ef_uses_same_target_and_initial_problem():
    p = make_problem(8,10,seed=4)
    summary, samples = measure_case(p,10,8,'EF',1.8/10,1e-6,1000,2)
    assert summary['status']=='Converged'
    assert summary['grad_final'] <= 1e-6
    assert summary['fatoracao_mediana_s']==0
    assert len(samples)==2
