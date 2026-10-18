"""Figuren: Aufbau (Spuren, Achsen, Beschriftungen) für jede Aufgabe und die Experimente; Zahlen in den Heatmaps gegen Handrechnung."""

import math

import numpy as np
import plotly.graph_objects as go
import pytest

import spv_advisor as ADV
import spv_constants as C
import spv_evaluation as ev
import spv_regret as R
import spv_results as RES
import spv_visualization as V
from sweeps_cache import get

S = ev.Settings


def trace_names(fig):
    return [t.name for t in fig.data]


def test_color_of_and_xlabel():
    assert V.color_of("A", "Hann-Periodogramm") == C.PALETTE[0] and V.color_of("A", "AR (Burg, BIC)") == C.PALETTE[3] and V.color_of("A", "verrauscht") == C.COLOR_NOISY
    assert V.color_of("B", "ESPRIT (p = 4)") == C.PALETTE[5] and len({V.color_of("B", m) for m in C.METHODS["B"]}) == 7
    assert V.xlabel(None) == "rauschfrei" and V.xlabel("weiß") == "weiß" and V.xlabel(1.5) == "1,5" and V.xlabel(-6.0) == "−6" and V.xlabel(64) == "64" and V.xlabel(0.75) == "0,75"


@pytest.mark.parametrize("task", C.TASKS)
def test_live_figures_for_every_task(task):
    a = ev.analyse(S(task=task))
    fig = V.build_live(a)
    assert isinstance(fig, go.Figure) and len(fig.data) >= len(a.curves)
    assert fig.layout.xaxis.fixedrange and fig.layout.yaxis.fixedrange and fig.layout.xaxis.title.text and fig.layout.yaxis.title.text
    names = trace_names(fig)
    if task in ("A", "F", "G"):
        assert names[0] in ("wahres Spektrum", "wahr")
    if task == "G":
        assert names[:3] == ["wahr", "verrauscht", "Wavelet-SURE (D8, weich)"] and names.count("verrauscht") == 1
    vals = V.build_values(a)
    assert list(vals.data[0].x) == list(a.values) and list(vals.data[0].y) == pytest.approx(list(a.values.values()))


def test_live_figure_marks_the_true_lines_and_the_esprit_estimates():
    a = ev.analyse(S(task="B"))
    fig = V.build_live(a)
    assert len(fig.layout.shapes) == 2 and "ESPRIT-Schätzungen" in trace_names(fig) and list(fig.layout.yaxis.range) == [-60, 3]
    c = V.build_live(ev.analyse(S(task="C")))
    assert len(c.layout.shapes) == 1 and trace_names(c) == ["Hann-Periodogramm"]
    d = V.build_live(ev.analyse(S(task="D")))
    assert len(d.layout.shapes) == 1 and V.build_live(ev.analyse(S(task="F"))).layout.shapes[0].x0 == 60


def test_values_figure_for_c_carries_the_bound():
    fig = V.build_values(ev.analyse(S(task="C")))
    assert len(fig.layout.shapes) == 1 and fig.layout.shapes[0].y0 == pytest.approx(0.0217958, abs=1e-6) and fig.layout.annotations[0].text == "Cramér-Rao-Schranke"
    assert len(V.build_values(ev.analyse(S(task="A"))).layout.shapes) == 0


def test_d_stats_figure_is_statistic_over_calibrated_threshold():
    a = ev.analyse(S(task="D"))
    fig = V.build_d_stats(a)
    stats, cal = a.extra["stats"], a.extra["calibrated"]
    assert list(fig.data[0].y) == pytest.approx([stats[i] / cal[i] for i in range(3)])
    thr_g, thr_f = a.extra["nominal"]
    assert list(fig.data[1].y) == pytest.approx([thr_g / cal[0], thr_f / cal[2]]) and list(fig.data[1].x) == [C.METHODS["D"][0], C.METHODS["D"][2]]
    assert fig.layout.shapes[0].y0 == 1.0


def test_heatmap_numbers_by_hand(cells_g, cells_d):
    fig = V.build_heatmap("G", cells_g)
    h = fig.data[0]
    assert list(h.y) == list(C.METHODS["G"]) and list(h.x) == ["Blocks", "Bumps"]
    assert np.array(h.z) == pytest.approx(np.log10([[1, 2], [2, 1], [4, 4]])) and [list(r) for r in h.text] == [["1,0", "2,0"], ["2,0", "1,0"], ["4,0", "4,0"]]
    fig = V.build_heatmap("D", cells_d)
    assert np.array(fig.data[0].z) == pytest.approx(np.array([[40, 0, 95], [20, 0, 0], [0, 0, 10]])) and [list(r) for r in fig.data[0].text][0] == ["40", "0", "95"]


@pytest.mark.parametrize("task", C.TASKS)
def test_heatmap_for_every_task(task):
    cells = RES.cells(task)
    fig = V.build_heatmap(task, cells)
    methods = [m for m in C.METHODS[task] if all(m in c["values"] for c in cells)]
    assert list(fig.data[0].y) == methods and len(fig.data[0].x) == len(cells) and np.array(fig.data[0].z).shape == (len(methods), len(cells))
    assert np.all(np.isfinite(np.array(fig.data[0].z, float)))


def test_sweep_traces_and_labels(cells_d):
    fig = V.build_sweep("D", cells_d, "snr", [-6.0, 0.0], "bg", ["weiß", "AR(1) 0,8"])
    assert len(fig.data) == 6 and [list(t.x) for t in fig.data[:3]] == [["−6", "0"]] * 3 and [list(t.x) for t in fig.data[3:]] == [["−6"]] * 3
    assert list(fig.data[2].y) == [100.0, 100.0] and list(fig.data[0].y) == [60.0, 100.0] and fig.layout.yaxis.title.text == "%"
    assert [a.text for a in fig.layout.annotations] == ["weiß", "AR(1) 0,8"] and fig.layout.xaxis.type == "category" and list(fig.layout.yaxis.range) == [-2, 102]
    assert [t.showlegend for t in fig.data[:3]] == [True] * 3 and [t.showlegend for t in fig.data[3:6]] == [False] * 3


def test_sweep_for_errors_is_logarithmic_and_has_the_nominal_curves_for_d():
    cells = RES.cells("A")
    fig = V.build_sweep("A", cells, "bg", list(C.BACKGROUNDS_A), "n", list(C.N_A))
    assert len(fig.data) == 4 * 3 and fig.layout.yaxis.type == "log" and list(fig.data[0].x) == [V.xlabel(b) for b in C.BACKGROUNDS_A]
    d = V.build_sweep("D", RES.cells("D"), "snr", list(C.D_SNRS), "bg", list(C.D_BGS), nominal=True)
    assert len(d.data) == 3 * 2 + 2 * 2
    dotted = [t for t in d.data if t.line.dash == "dot"]
    assert len(dotted) == 4 and dotted[0].name == "Fisher-g (weißer Hintergrund) (Theorieschwelle)" and list(dotted[0].y)[:1] == pytest.approx([RES.find("D", bg="weiß", snr=-12.0)["extra"]["nominal_rate"]["Fisher-g (weißer Hintergrund)"]])


def test_sweep_without_split_for_g_and_missing_points_for_e():
    g = V.build_sweep("G", RES.cells("G"), "signal", list(C.G_SIGNALS))
    assert len(g.data) == 3 and list(g.data[0].x) == list(C.G_SIGNALS) and g.layout.legend.y == -0.75
    e = V.build_sweep("E", RES.cells("E"), "snr", sorted(C.E_SNRS), "line", list(C.E_LINES))
    interp = [t for t in e.data if t.name == C.E_INTERP]
    assert len(interp) == 1 and len(interp[0].x) == 3                                  # nur in dem Feld der Linie unter der Nyquist-Grenze
    f = V.build_sweep("F", RES.cells("F"), "snr", list(C.F_SNRS), "signal", list(C.F_SIGNALS))
    assert list(f.data[0].x) == ["rauschfrei", "20", "10", "0"] and f.layout.yaxis.type == "log"


def test_mix_figure_follows_the_curves():
    curves = ev.mix_curves(25)
    fig = V.build_mix(curves, 0.71)
    assert len(fig.data) == 8 and list(fig.data[0].x) == [0, 25, 50, 75, 100] and fig.layout.yaxis.type == "log"
    assert [t.showlegend for t in fig.data[1::2]] == [False] * 4 and [s.x0 for s in fig.layout.shapes] == [71.0, 71.0]
    assert list(fig.data[6].y) == pytest.approx(curves["mean"]["AR (Burg, BIC)"]) and list(fig.data[7].y) == pytest.approx(curves["wins"]["AR (Burg, BIC)"])


def test_advice_figure(mini_results):
    adv = ADV.advise("D", {"bg": "AR(1) 0,8", "snr": -6.0}, True, mini_results)
    fig = V.build_advice(adv.rows, "D")
    assert list(fig.data[0].y) == [95.0, 0.0, 10.0] and list(fig.data[1].y) == [95.0, 20.0, 10.0] and fig.layout.yaxis.title.text == "Reuegrad (Prozentpunkte hinter dem besten)"
    adv = ADV.advise("G", {"signal": "Blocks"}, True, mini_results)
    fig = V.build_advice(adv.rows, "G")
    assert list(fig.data[0].y) == [1.0, 2.0, 4.0] and fig.layout.yaxis.title.text == "Reuegrad (1 = bester)"


def test_experiment_figures_from_the_cache():
    sp = get("spectrum")
    assert len(V.build_heatmap("A", sp["cells"]).data) == 1 and math.isfinite(sp["share"])
