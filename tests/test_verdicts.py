"""Verdikte: jede Verzweigung der Live-Verdikte an handgesetzten Kennzahlen, die Verdikte der Experimente an der Tabelle."""

import pytest

import spv_constants as C
import spv_evaluation as ev
from ev_helpers import analysis_with
from sweeps_cache import get

AR = "AR (Burg, BIC)"
HEAD = "Szenario Test (eine Realisierung, Seed 47): "
TAIL = "Die Tabellen unten zeigen viele Läufe; eine einzelne Realisierung kann vom Mittel abweichen."


def a_values(**override):
    base = {"Hann-Periodogramm": 6.0, "Welch (Hann, 50 %, n/4)": 3.0, "Multitaper (NW = 4)": 4.0, AR: 1.0}
    base.update(override)
    return base


def test_verdict_a_ar_wins_on_an_autoregressive_background():
    kind, text = ev.verdict_main(analysis_with("A", a_values(), a_bg="Steil"))
    assert kind == "success" and text.startswith(HEAD + "kleinster Fehler AR (Burg, BIC) mit 1,00 dB, größter Hann-Periodogramm mit 6,00 dB.") and "Das AR-Modell passt (Heimvorteil)." in text and text.endswith(TAIL)


def test_verdict_a_without_home_advantage():
    kind, text = ev.verdict_main(analysis_with("A", a_values(**{AR: 5.0}), a_bg="Kerbe"))
    assert kind == "info" and "kleinster Fehler Welch (Hann, 50 %, n/4) mit 3,00 dB, größter Hann-Periodogramm mit 6,00 dB." in text and "Der Hintergrund ist kein autoregressives Spektrum." in text and "Heimvorteil" not in text


def test_verdict_a_ar_best_on_a_non_ar_background_is_still_green_but_says_no_home_advantage():
    kind, text = ev.verdict_main(analysis_with("A", a_values(), a_bg="Bandpass"))
    assert kind == "success" and "kein autoregressives Spektrum" in text


@pytest.mark.parametrize("task,names,word", [("B", C.METHODS["B"], "aufgelöst"), ("D", C.METHODS["D"], "erkannt"), ("E", C.METHODS["E"], "erkannt")])
def test_verdict_rate_tasks_all_some_none(task, names, word):
    none = ev.verdict_main(analysis_with(task, {k: 0.0 for k in names}))
    assert none[0] == "warning" and none[1] == HEAD + f"kein Verfahren hat die Linie(n) {word}. " + TAIL
    allv = ev.verdict_main(analysis_with(task, {k: 1.0 for k in names}))
    assert allv[0] == "success" and allv[1] == HEAD + f"alle Verfahren haben die Linie(n) {word}. " + TAIL
    some = ev.verdict_main(analysis_with(task, {names[0]: 1.0, names[1]: 0.0, **{k: 1.0 for k in names[2:]}}))
    assert some[0] == "info" and f"{word}: {names[0]}, " in some[1] and f"nicht {word}: {names[1]}. " in some[1]


def test_verdict_rate_task_threshold_is_one_half():
    names = C.METHODS["D"]
    kind, text = ev.verdict_main(analysis_with("D", {names[0]: 0.5, names[1]: 0.49, names[2]: 0.0}))
    assert kind == "info" and "erkannt: Fisher-g (weißer Hintergrund); nicht erkannt: Fisher-g nach AR(2)-Vorfilterung, Harmonischer F-Test (NW = 4)." in text


def test_verdict_c_names_the_bound():
    v = {k: 0.05 for k in C.METHODS["C"]}
    v["Spitze (Rechteck, 16-fach, Parabel)"], v["AR-Spitze (Burg, p = 8)"] = 0.01, 0.2
    kind, text = ev.verdict_main(analysis_with("C", v, extra={"crb": 0.0218}))
    assert kind == "info" and "kleinster Fehler Spitze (Rechteck, 16-fach, Parabel) mit 0,010 Bins (Cramér-Rao-Schranke 0,022 Bins), größter AR-Spitze (Burg, p = 8) mit 0,200 Bins." in text


def test_verdict_f_and_g():
    f = {k: 0.01 for k in C.METHODS["F"]}
    f["Neu zugeordnet"], f["Synchrosqueezing"] = 0.001, 0.05
    kind, text = ev.verdict_main(analysis_with("F", f))
    assert kind == "info" and "kleinster Fehler Neu zugeordnet mit 0,00100, größter Synchrosqueezing mit 0,05000." in text
    g = {"Wavelet-SURE (D8, weich)": 0.4, "Wavelet universal (hart)": 0.7, "Fourier universal (hart)": 1.2}
    kind, text = ev.verdict_main(analysis_with("G", g, refs={"verrauscht": 1.0}))
    assert kind == "info" and "kleinster Fehler Wavelet-SURE (D8, weich) mit 0,400 (verrauscht 1,000)." in text


def test_experiment_verdict_kinds_and_openings():
    kinds = {"spectrum": ev.verdict_spectrum, "lines": ev.verdict_lines, "detection": ev.verdict_detection, "irregular": ev.verdict_irregular, "denoise": ev.verdict_denoise}
    want = {"spectrum": "warning", "lines": "info", "detection": "warning", "irregular": "info", "denoise": "success"}
    for name, fn in kinds.items():
        kind, text = fn(get(name))
        assert kind == want[name] and text and text[0].isupper() and text.endswith("."), name


def test_spectrum_verdict_follows_the_mix():
    low = ev.verdict_spectrum(ev.exp_spectrum(0))[1]
    high = ev.verdict_spectrum(ev.exp_spectrum(100))[1]
    assert "Unter dem eingestellten Mix (0 % autoregressiv)" in low and "Unter dem eingestellten Mix (100 % autoregressiv)" in high and low != high
    assert "ab einem autoregressiven Anteil von 26 % vor Welch" in low and "15 von 15 Szenarien" in low
