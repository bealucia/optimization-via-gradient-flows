"""Contrato das medições: custos, repetições, resíduos e status sem ranking de tempos."""
import numpy as np

from src.problems.least_squares import make_problem
from src.experiments.exp3_performance import benchmark_instance, _prepared_run


def test_paired_strategies_match_minimizer_and_tolerance():
    problem = make_problem(8, 10, seed=4)
    summaries, raw = benchmark_instance(problem, 10, 8, 1e-6, 1000, 2)
    assert len(summaries) == 6
    assert len(raw) == 12
    assert {r['metodo'] for r in summaries} == {
        'EF-direto', 'EF-Gram', 'Cholesky', 'LU', 'QR', 'SVD'}
    assert all(row['status'] == 'Converged' for row in summaries)
    assert all(row['gradiente_original_final'] <= 1e-6 + 1e-10 for row in summaries)
    assert len({r['iteracoes'] for r in summaries if r['metodo'].startswith('EF')}) == 1
    assert len({r['iteracoes'] for r in summaries if r['metodo'] in ('Cholesky', 'LU', 'QR', 'SVD')}) == 1
    for row in raw:
        assert np.isclose(row['total_s'], row['preparacao_s'] + row['laco_s'],
                          rtol=1e-12, atol=1e-12)
        assert row['repeticao'] in (1, 2)
        assert 1 <= row['ordem_na_rodada'] <= 6
    for summary in summaries:
        if summary['metodo'] in ('Cholesky', 'LU', 'QR', 'SVD'):
            assert summary['residuo_relativo_max_todos_passos'] < 1e-10
        else:
            assert summary['residuo_relativo_max_todos_passos'] == ''


def test_explicit_preprocessing_is_measured_and_direct_has_none():
    p = make_problem(10, 20, seed=9)
    direct = _prepared_run(p, 'EF-direto', 1.8/20, 1e-6, 3000)
    gram = _prepared_run(p, 'EF-Gram', 1.8/20, 1e-6, 3000)
    assert direct['preparacao_s'] == 0
    assert gram['preparacao_s'] >= 0
    assert direct['iteracoes'] == gram['iteracoes']
    assert direct['status'] == gram['status'] == 'Converged'
    np.testing.assert_allclose(direct['gradiente_original_final'],
                               gram['gradiente_original_final'], atol=1e-10)


def test_max_steps_is_not_called_divergence():
    p = make_problem(8, 100, seed=4)
    sample = _prepared_run(p, 'EF-Gram', 1.8/100, tol=1e-12, max_steps=2)
    assert sample['status'] == 'Max Steps'
    assert sample['iteracoes'] == 2
