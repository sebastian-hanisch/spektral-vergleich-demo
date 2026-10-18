"""Entscheidungshilfe gegen Handrechnung (Mini-Tabelle aus conftest.py)."""

import math

import pytest

import spv_advisor as ADV
import spv_constants as C

SURE, UNIV, FOU = C.METHODS["G"]
FG, FP, FF = C.METHODS["D"]


def test_distance_rules():
    assert ADV._distance("D", {"bg": "weiß", "snr": -6.0}, {"bg": "weiß", "snr": -6.0}) == 0.0
    assert ADV._distance("D", {"bg": "weiß", "snr": -6.0}, {"bg": "AR(1) 0,8", "snr": -6.0}) == math.inf
    assert ADV._distance("D", {"bg": "weiß", "snr": -3.0}, {"bg": "weiß", "snr": 0.0}) == pytest.approx(3.0 / 1.0)       # |(-3) - 0| / (|0| + 1)
    assert ADV._distance("D", {"bg": "weiß", "snr": 10.0}, {"bg": "weiß", "snr": 20.0}) == pytest.approx(10.0 / 21.0)
    assert ADV._distance("F", {"signal": "Chirp", "snr": 10.0}, {"signal": "Chirp", "snr": None}) == math.inf
    assert ADV._distance("F", {"signal": "Chirp", "snr": None}, {"signal": "Chirp", "snr": None}) == 0.0


def test_nearest_cell_exact_and_near(mini_results):
    cell, exact = ADV.nearest_cell("D", {"bg": "weiß", "snr": 0.0}, mini_results)
    assert exact and cell["params"] == {"bg": "weiß", "snr": 0.0}
    cell, exact = ADV.nearest_cell("D", {"bg": "weiß", "snr": -4.0}, mini_results)        # Abstand zu -6: 2/7 = 0,286; zu 0: 4/1 = 4
    assert not exact and cell["params"]["snr"] == -6.0
    cell, exact = ADV.nearest_cell("D", {"bg": "AR(1) 0,8", "snr": 3.0}, mini_results)
    assert not exact and cell["params"] == {"bg": "AR(1) 0,8", "snr": -6.0}


def test_nearest_cell_needs_a_matching_category(mini_results):
    with pytest.raises(ValueError):
        ADV.nearest_cell("D", {"bg": "Unsinn", "snr": 0.0}, mini_results)


def test_worst_cell(cells_d):
    reg, where = ADV.worst_cell("D", FG, cells_d)
    assert reg == 95.0 and where["params"] == {"bg": "AR(1) 0,8", "snr": -6.0}
    reg, where = ADV.worst_cell("D", FF, cells_d)
    assert reg == 10.0 and where["params"]["bg"] == "AR(1) 0,8"


def test_advice_for_a_known_type_recommends_the_local_winner(mini_results):
    adv = ADV.advise("D", {"bg": "AR(1) 0,8", "snr": -6.0}, True, mini_results)
    assert adv.recommendation == FP and adv.exact and adv.known and "nächstgelegenen Szenario" in adv.basis and adv.note == ""
    assert adv.cell["params"] == {"bg": "AR(1) 0,8", "snr": -6.0}
    row = {r["method"]: r for r in adv.rows}
    assert row[FG]["value"] == 5.0 and row[FG]["regret"] == 95.0 and row[FG]["max_regret"] == 95.0 and row[FG]["mean_regret"] == pytest.approx(45.0) and row[FG]["worst_cell"] == "AR(1) 0,8, Linie -6 dB"
    assert row[FP]["regret"] == 0.0 and row[FP]["max_regret"] == 20.0 and row[FF]["max_regret"] == 10.0


def test_advice_for_an_unknown_type_is_the_minimax_choice(mini_results):
    adv = ADV.advise("D", {"bg": "AR(1) 0,8", "snr": -6.0}, False, mini_results)
    assert adv.recommendation == FF and "kleinster größter Reuegrad über alle 3 Szenarien" in adv.basis and not adv.known


def test_advice_between_scenarios_carries_a_note(mini_results):
    adv = ADV.advise("D", {"bg": "weiß", "snr": -4.0}, True, mini_results)
    assert not adv.exact and "zwischen den gemessenen Szenarien" in adv.note and adv.cell["params"]["snr"] == -6.0 and adv.recommendation == FF


def test_advice_rows_follow_the_method_order(mini_results):
    adv = ADV.advise("G", {"signal": "Blocks"}, True, mini_results)
    assert adv.recommendation == SURE and [r["method"] for r in adv.rows] == [SURE, UNIV, FOU]
    assert adv.rows[0]["worst_cell"] == "Bumps" and adv.rows[0]["max_regret"] == 2.0 and adv.rows[2]["max_regret"] == 4.0


def test_worst_cell_returns_the_first_of_equal_cells():
    from conftest import cell
    cells = [cell("D", {"bg": "weiß", "snr": s}, {FG: 50.0, FP: 100.0, FF: 100.0}) for s in (-6.0, -3.0, 0.0)]
    reg, where = ADV.worst_cell("D", FG, cells)
    assert reg == 50.0 and where["params"]["snr"] == -6.0                              # bei Gleichstand die erste Zelle
