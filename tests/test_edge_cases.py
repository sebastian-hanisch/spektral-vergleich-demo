"""Randfälle: Regler an den Grenzen, alle Aufgaben mit Entscheidungshilfe und PDF, leere Eingaben."""

import itertools
import math

import pytest

import spv_advisor as ADV
import spv_constants as C
import spv_evaluation as ev
from spv_pdf_export import build_pdf

S = ev.Settings


def edge_settings():
    """Alle Aufgaben an den Reglergrenzen (Minimum und Maximum je Regler, Kategorien vollständig)."""
    out = []
    out += [S(task="A", a_bg=bg, a_n=n) for bg in C.BACKGROUNDS_A for n in (C.N_A[0], C.N_A[-1])]
    out += [S(task="B", b_delta=d, b_snr=s) for d, s in itertools.product((C.B_DELTA_MIN, C.B_DELTA_MAX), (C.B_SNR_MIN, C.B_SNR_MAX))]
    out += [S(task="C", c_n=n, c_snr=s) for n, s in itertools.product(C.C_NS, (C.C_SNR_MIN, C.C_SNR_MAX))]
    out += [S(task="D", d_bg=bg, d_snr=s) for bg, s in itertools.product(C.D_BGS, (C.D_SNR_MIN, C.D_SNR_MAX))]
    out += [S(task="E", e_line=line, e_snr=s) for line, s in itertools.product(C.E_LINE_CHOICES, (C.E_SNR_MIN, C.E_SNR_MAX))]
    out += [S(task="F", f_sig=sig, f_snr=s) for sig, s in itertools.product(C.F_SIGNALS, (C.F_SNR_MIN, C.F_SNR_MAX))]
    out += [S(task="G", g_sig=sig) for sig in C.G_SIGNALS]
    return out


@pytest.mark.parametrize("s", edge_settings(), ids=lambda s: f"{s.task}-{s.params}")
def test_every_edge_setting_gives_finite_values(s):
    a = ev.analyse(s)
    assert list(a.values) == [m for m in C.METHODS[s.task] if m in a.values] and len(a.values) >= 3
    assert all(math.isfinite(v) and v >= 0 for v in a.values.values())
    if C.TASK_KIND[s.task] == C.KIND_RATE:
        assert set(a.values.values()) <= {0.0, 1.0}
    kind, text = ev.verdict_main(a)
    assert kind in ("success", "info", "warning") and a.title in text


@pytest.mark.parametrize("task", C.TASKS)
@pytest.mark.parametrize("known", [True, False])
def test_advice_for_every_task_and_both_answers(task, known):
    s = {"A": S(task="A", a_bg="Bandpass", a_n=1024), "B": S(task="B", b_delta=0.5, b_snr=0.0), "C": S(task="C", c_n=256, c_snr=-5.0), "D": S(task="D", d_snr=3.0), "E": S(task="E", e_line=0.2, e_snr=-9.0),
         "F": S(task="F", f_snr=40.0), "G": S(task="G", g_sig="HeaviSine")}[task]
    adv = ADV.advise(task, s.params, known)
    assert adv.recommendation in C.METHODS[task] and adv.known is known and len(adv.rows) == len([m for m in C.METHODS[task] if any(r["method"] == m for r in adv.rows)])
    rows = ev.advice_rows(task, adv)
    assert rows and all(set(r) >= {"Verfahren", "Reuegrad hier", "größter Reuegrad", "mittlerer Reuegrad", "am schlechtesten bei"} for r in rows)
    assert not adv.exact or adv.note == ""


def test_advice_for_the_noise_free_chirp_is_exact_and_between_noisy_cases_is_not():
    adv = ADV.advise("F", {"signal": "Chirp", "snr": None}, True)
    assert adv.exact and adv.cell["params"] == {"signal": "Chirp", "snr": None}
    adv = ADV.advise("F", {"signal": "Chirp", "snr": 5.0}, True)
    assert not adv.exact and adv.cell["params"]["snr"] in (0.0, 10.0)


def test_unknown_advice_is_independent_of_the_scenario():
    a = ADV.advise("A", {"bg": "Kerbe", "n": 64}, False)
    b = ADV.advise("A", {"bg": "Resonanz", "n": 1024}, False)
    assert a.recommendation == b.recommendation == "AR (Burg, BIC)" and [r["max_regret"] for r in a.rows] == [r["max_regret"] for r in b.rows]


@pytest.mark.parametrize("task", C.TASKS)
def test_pdf_for_every_task(task):
    a = ev.analyse(S(task=task))
    data = build_pdf(a)
    assert data[:5] == b"%PDF-" and len(data) > 1500


def test_empty_inputs():
    assert ev.table_rows("A", []) == []
    assert ev.names_label(["x"]) == "x"


def test_summarize_and_exp_spectrum_at_the_mix_ends():
    for share in (0, 100):
        r = ev.exp_spectrum(share)
        assert r["minimax"] in C.METHODS["A"] and sum(v["wins_weighted"] for v in r["summary"].values()) == pytest.approx(1.0)
