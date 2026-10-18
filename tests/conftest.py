"""Gemeinsame Fixtures: handgerechnete Mini-Zellen für Reuegrad und Entscheidungshilfe (DEMO-PLAYBOOK §1b)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def cell(task, params, values, refs=None, extra=None):
    return {"task": task, "params": params, "values": values, "refs": refs or {}, "extra": extra or {}}


@pytest.fixture
def cells_g():
    """Aufgabe G (Fehler, drei Verfahren) mit zwei Signalen. Handrechnung der Reuegrade (Fehler / bester Fehler):
    Blocks: (1, 2, 4) -> (1, 2, 4); Bumps: (3, 1,5; 6) -> (2, 1, 4)."""
    s, w, f = "Wavelet-SURE (D8, weich)", "Wavelet universal (hart)", "Fourier universal (hart)"
    return [cell("G", {"signal": "Blocks"}, {s: 1.0, w: 2.0, f: 4.0}), cell("G", {"signal": "Bumps"}, {s: 3.0, w: 1.5, f: 6.0})]


@pytest.fixture
def cells_d():
    """Aufgabe D (Erfolgsanteile in %, drei Verfahren) mit drei Szenarien. Reuegrad = bester minus eigener:
    (weiß, -6): (60, 80, 100) -> (40, 20, 0); (weiß, 0): (100, 100, 100) -> (0, 0, 0); (AR, -6): (5, 100, 90) -> (95, 0, 10)."""
    g, p, f = "Fisher-g (weißer Hintergrund)", "Fisher-g nach AR(2)-Vorfilterung", "Harmonischer F-Test (NW = 4)"
    return [cell("D", {"bg": "weiß", "snr": -6.0}, {g: 60.0, p: 80.0, f: 100.0}), cell("D", {"bg": "weiß", "snr": 0.0}, {g: 100.0, p: 100.0, f: 100.0}),
            cell("D", {"bg": "AR(1) 0,8", "snr": -6.0}, {g: 5.0, p: 100.0, f: 90.0})]


@pytest.fixture
def mini_results(tmp_path, cells_g, cells_d):
    """Kleine Ergebnistabelle im Format von spv_results.json (für Entscheidungshilfe und Tabellenfunktionen)."""
    path = tmp_path / "mini_results.json"
    path.write_text(json.dumps({"version": 1, "cells": cells_g + cells_d}), encoding="utf-8")
    return str(path)
