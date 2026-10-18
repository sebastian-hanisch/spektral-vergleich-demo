"""Reuegrad, Mittel, Maximum, Siege, Minimax und Szenarienmix gegen Handrechnung (Mini-Zellen aus conftest.py)."""

import math

import pytest

import spv_constants as C
import spv_regret as R
from conftest import cell

SURE, UNIV, FOU = C.METHODS["G"]
FG, FP, FF = C.METHODS["D"]


def test_regret_ratio_is_error_over_best_error():
    assert R.regret_ratio({"a": 2.0, "b": 4.0, "c": 3.0}) == {"a": 1.0, "b": 2.0, "c": 1.5}


def test_regret_ratio_with_a_zero_best_error():
    assert R.regret_ratio({"a": 0.0, "b": 0.0, "c": 3.0}) == {"a": 1.0, "b": 1.0, "c": math.inf}


def test_regret_points_is_best_minus_own():
    assert R.regret_points({"a": 60.0, "b": 80.0, "c": 100.0}) == {"a": 40.0, "b": 20.0, "c": 0.0}


def test_cell_regret_by_task_kind(cells_g, cells_d):
    assert R.cell_regret("G", cells_g[0]) == {SURE: 1.0, UNIV: 2.0, FOU: 4.0}
    assert R.cell_regret("G", cells_g[1]) == {SURE: 2.0, UNIV: 1.0, FOU: 4.0}
    assert R.cell_regret("D", cells_d[0]) == {FG: 40.0, FP: 20.0, FF: 0.0}
    assert R.cell_regret("D", cells_d[2]) == {FG: 95.0, FP: 0.0, FF: 10.0}


def test_cell_regret_ignores_references_and_missing_methods():
    c = cell("G", {"signal": "Blocks"}, {SURE: 1.0, UNIV: 2.0}, refs={"verrauscht": 0.1})
    assert R.cell_regret("G", c) == {SURE: 1.0, UNIV: 2.0}


def test_winners_include_all_ties(cells_g, cells_d):
    assert R.winners("G", cells_g[0]) == [SURE] and R.winners("G", cells_g[1]) == [UNIV]
    assert R.winners("D", cells_d[0]) == [FF] and R.winners("D", cells_d[1]) == [FG, FP, FF]


def test_summarize_with_equal_weights(cells_g):
    s = R.summarize("G", cells_g)
    # Mittel: SURE (1 + 2)/2, universal (2 + 1)/2, Fourier (4 + 4)/2; Maximum 2, 2, 4; Siege 1, 1, 0
    assert {k: v["mean"] for k, v in s.items()} == {SURE: 1.5, UNIV: 1.5, FOU: 4.0}
    assert {k: v["max"] for k, v in s.items()} == {SURE: 2.0, UNIV: 2.0, FOU: 4.0}
    assert {k: v["wins"] for k, v in s.items()} == {SURE: 1, UNIV: 1, FOU: 0}
    assert {k: v["wins_weighted"] for k, v in s.items()} == {SURE: 0.5, UNIV: 0.5, FOU: 0.0} and all(v["n"] == 2 for v in s.values())


def test_summarize_weights_are_normalised(cells_g):
    s = R.summarize("G", cells_g, [3.0, 1.0])
    assert s[SURE]["mean"] == pytest.approx((3 * 1.0 + 2.0) / 4) and s[UNIV]["mean"] == pytest.approx((3 * 2.0 + 1.0) / 4) and s[FOU]["mean"] == pytest.approx(4.0)
    assert s[SURE]["wins_weighted"] == pytest.approx(0.75) and s[UNIV]["wins_weighted"] == pytest.approx(0.25)
    assert R.summarize("G", cells_g, [30.0, 10.0]) == s


def test_summarize_zero_weight_cell_does_not_count_for_the_maximum(cells_g):
    s = R.summarize("G", cells_g, [1.0, 0.0])
    assert s[SURE]["max"] == 1.0 and s[UNIV]["max"] == 2.0 and s[SURE]["mean"] == 1.0
    assert s[SURE]["wins"] == 1 and s[UNIV]["wins"] == 1 and s[SURE]["wins_weighted"] == 1.0 and s[UNIV]["wins_weighted"] == 0.0        # Siege zählen alle Zellen, der Siegesanteil nur gewichtete


def test_summarize_rate_task(cells_d):
    s = R.summarize("D", cells_d)
    assert {k: v["mean"] for k, v in s.items()} == pytest.approx({FG: (40 + 0 + 95) / 3, FP: (20 + 0 + 0) / 3, FF: (0 + 0 + 10) / 3})
    assert {k: v["max"] for k, v in s.items()} == {FG: 95.0, FP: 20.0, FF: 10.0} and {k: v["wins"] for k, v in s.items()} == {FG: 1, FP: 2, FF: 2}


def test_summarize_rejects_bad_input(cells_g):
    for bad in ([1.0], [1.0, 2.0, 3.0], [-1.0, 2.0], [0.0, 0.0]):
        with pytest.raises(ValueError):
            R.summarize("G", cells_g, bad)
    with pytest.raises(ValueError):
        R.summarize("G", [])


def test_minimax_choice_and_ties(cells_g, cells_d):
    assert R.minimax_choice(R.summarize("G", cells_g)) == SURE            # Maximum 2 bei SURE und universal, Mittel gleich: die Reihenfolge entscheidet
    assert R.minimax_choice(R.summarize("D", cells_d)) == FF               # Maximum 10 gegen 20 und 95
    tie = {"a": {"max": 2.0, "mean": 1.5}, "b": {"max": 2.0, "mean": 1.2}, "c": {"max": 3.0, "mean": 1.0}}
    assert R.minimax_choice(tie) == "b"


def test_mix_weights_split_the_share_over_the_groups():
    cells = [cell("A", {"bg": b, "n": 64}, {}) for b in ("Resonanz", "Steil", "weiß", "Kerbe")]
    w = R.mix_weights(cells, 0.6)
    assert w == pytest.approx([0.2, 0.2, 0.2, 0.4]) and sum(w) == pytest.approx(1.0)
    assert R.mix_weights(cells, 1.0) == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.0]) and R.mix_weights(cells, 0.0) == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_mix_weights_reject_bad_input():
    cells = [cell("A", {"bg": b, "n": 64}, {}) for b in ("Resonanz", "Kerbe")]
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            R.mix_weights(cells, bad)
    with pytest.raises(ValueError):
        R.mix_weights(cells[:1], 0.5)
    with pytest.raises(ValueError):
        R.mix_weights(cells[1:], 0.5)


def test_cell_labels():
    assert R.cell_label("A", {"bg": "Kerbe", "n": 256}) == "Kerbe, n = 256"
    assert R.cell_label("B", {"delta": 1.5, "snr": 10.0}) == "Abstand 1,5 Bins, 10 dB"
    assert R.cell_label("C", {"n": 64, "snr": 0.0}) == "n = 64, 0 dB"
    assert R.cell_label("D", {"bg": "weiß", "snr": -6.0}) == "weiß, Linie -6 dB"
    assert R.cell_label("E", {"line": 1.3, "snr": -3.0}) == "Linie bei 1,3, -3 dB"
    assert R.cell_label("F", {"signal": "Chirp", "snr": None}) == "Chirp, rauschfrei" and R.cell_label("F", {"signal": "Sinus-FM", "snr": 20.0}) == "Sinus-FM, 20 dB"
    assert R.cell_label("G", {"signal": "Bumps"}) == "Bumps"
    with pytest.raises(ValueError):
        R.cell_label("X", {})
