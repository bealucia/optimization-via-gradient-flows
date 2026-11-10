"""Persistência de resultados experimentais; sem inferência científica.

As rotinas escrevem somente em ``results/<experimento>/``. Não movem,
arquivam nem apagam outros resultados. Todas as figuras são PNG a 300 dpi.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def folders(experiment: str, *, root: Path | None = None) -> dict[str, Path]:
    if not experiment or '/' in experiment or '\\' in experiment or experiment in ('.', '..'):
        raise ValueError('Nome inválido de experimento')
    base = Path(root) if root is not None else Path(__file__).resolve().parents[1] / 'results'
    target = base / experiment
    paths = {kind: target / kind for kind in ('figures', 'tables', 'logs')}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def csv_write(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0])
    if any(set(row) != set(keys) for row in rows):
        raise ValueError(f'Colunas inconsistentes no CSV {path}')
    with path.open('w', encoding='utf-8', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_write(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def png(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(path, dpi=300, bbox_inches='tight')
    finally:
        from matplotlib import pyplot as plt
        plt.close(fig)
