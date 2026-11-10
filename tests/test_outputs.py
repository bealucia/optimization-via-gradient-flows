"""Integração experimental e arquivos de saída, sempre em diretórios temporários."""
import csv

from src.output import folders
from src.experiments import exp1_quadratic, exp2_nonquadratic, exp3_performance


def test_exp1_essential_figures_and_tables(monkeypatch, tmp_path):
    out = folders('exp1_quadratic',root=tmp_path)
    monkeypatch.setattr(exp1_quadratic, 'FIGURES',out['figures'])
    monkeypatch.setattr(exp1_quadratic, 'TABLES',out['tables'])
    monkeypatch.setattr(exp1_quadratic, 'LOGS',out['logs'])
    monkeypatch.setattr(exp1_quadratic, 'AUX_FIGURES',out['figures']/'auxiliares')
    error=exp1_quadratic.experiment_stability()
    slopes=exp1_quadratic.experiment_accuracy()
    assert error < 1e-9
    assert 0.85 < slopes['EF'] < 1.15
    assert 0.85 < slopes['EB'] < 1.15
    assert (out['figures']/'01_estabilidade.png').is_file()
    assert (out['figures']/'03_erro_temporal.png').is_file()
    assert (out['tables']/'03_erro_temporal.csv').is_file()


def test_exp2_single_program_writes_chosen_analysis(monkeypatch,tmp_path):
    monkeypatch.setattr(exp2_nonquadratic,'folders',
                        lambda name: folders(name,root=tmp_path))
    exp2_nonquadratic.main('convergencia', piloto=True)
    out=tmp_path/'exp2_nonquadratic'
    assert (out/'figures'/'01_convergencia_nao_quadratica.png').is_file()
    assert (out/'figures'/'02_limite_contracao.png').is_file()
    with (out/'tables'/'01_resumo_convergencia.csv').open(encoding='utf-8') as fp:
        rows=list(csv.DictReader(fp))
    assert len(rows)==3
    assert any(r['metodo']=='EB' for r in rows)


def test_exp3_all_scenarios_preserved_and_no_hidden_missing(monkeypatch,tmp_path):
    rows,comparison=exp3_performance.main(
        dimensions=(6,), kappas=(10,), repeats=2, max_steps=200,
        scenarios=('extremos','multimodal'), output_root=tmp_path)
    assert len(rows)==12
    assert len(comparison)==2
    assert all(r['status']=='Converged' for r in rows)
    out=tmp_path/'exp3_performance'
    assert len(list((out/'figures').glob('*.png')))==4
    assert (out/'tables'/'02_tempos_repeticoes.csv').is_file()
    assert (out/'logs'/'config.json').is_file()


def test_exp3_nonconvergence_is_explicit_in_tables(monkeypatch, tmp_path):
    rows, comparisons = exp3_performance.main(
        dimensions=(6,), kappas=(100,), repeats=2, max_steps=1,
        scenarios=('extremos',), output_root=tmp_path)
    assert len(rows) == 6
    assert len(comparisons) == 1
    assert all(row['status'] == 'Max Steps' for row in rows)
    assert comparisons[0]['razao_EF_div_EB'] == ''
    assert (tmp_path/'exp3_performance'/'figures'/'04_vantagem_tempo_total.png').is_file()
