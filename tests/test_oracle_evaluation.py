"""Live-Analyse, Tabellenfunktionen und Hilfen der Auswertung (spv_evaluation.py) gegen Handrechnung und feste Werte."""

import math

import numpy as np
import pytest

import spv_advisor as ADV
import spv_constants as C
import spv_evaluation as ev
import spv_results as RES
from conftest import cell
from ev_helpers import analysis_with

S = ev.Settings


def test_de_formats_numbers_german():
    assert ev.de(-3.14159, 2) == "−3,14" and ev.de(2.0, 0) == "2" and ev.de(0.5) == "0,5" and ev.de(1234.5, 1) == "1234,5"
    assert ev.de(float("nan")) == "–" and ev.de(math.inf) == "∞" and ev.de(-math.inf) == "−∞"


def test_settings_params_per_task():
    s = S(task="A", a_bg="Kerbe", a_n=64)
    assert s.params == {"bg": "Kerbe", "n": 64}
    assert S(task="B", b_delta=1.5, b_snr=20.0).params == {"delta": 1.5, "snr": 20.0}
    assert S(task="C", c_n=256, c_snr=0.0).params == {"n": 256, "snr": 0.0}
    assert S(task="D", d_bg="weiß", d_snr=-3.0).params == {"bg": "weiß", "snr": -3.0}
    assert S(task="E", e_line=0.2, e_snr=-6.0).params == {"line": 0.2, "snr": -6.0}
    assert S(task="F", f_sig="Sinus-FM", f_snr=20.0).params == {"signal": "Sinus-FM", "snr": 20.0}
    assert S(task="G", g_sig="Bumps").params == {"signal": "Bumps"}
    with pytest.raises(ValueError):
        S(task="X").params


def test_default_settings_match_the_constants():
    s = S()
    assert s.task == C.DEFAULT_TASK and s.seed == C.DEFAULT_SEED and (s.a_bg, s.a_n, s.c_n, s.d_bg, s.e_line, s.f_sig, s.g_sig) == ("Resonanz", 256, 64, "AR(1) 0,8", 1.3, "Chirp", "Blocks")
    assert (s.b_delta, s.b_snr, s.c_snr, s.d_snr, s.e_snr, s.f_snr) == (1.0, 10.0, 10.0, -6.0, 0.0, 10.0)


def test_live_a_fixed_values_and_shape():
    a = ev.analyse(S(task="A"))
    assert a.title == "Resonanz, n = 256" and a.unit == "dB" and list(a.values) == list(C.METHODS["A"])
    assert a.values == pytest.approx({"Hann-Periodogramm": 7.059, "Welch (Hann, 50 %, n/4)": 2.200, "Multitaper (NW = 4)": 2.116, "AR (Burg, BIC)": 1.149}, abs=1e-3)
    assert a.refs == pytest.approx({"AR (Burg, feste Ordnung)": 1.1495}, abs=1e-3)
    assert all(len(x) == 129 and len(y) == 129 for x, y in a.curves.values()) and len(a.truth_curve[0]) == 129 and a.truth_lines == ()
    assert ev.analyse(S(task="A", a_bg="weiß", a_n=64, seed=5)).values == pytest.approx({"Hann-Periodogramm": 4.633, "Welch (Hann, 50 %, n/4)": 1.876, "Multitaper (NW = 4)": 1.908, "AR (Burg, BIC)": 1.530}, abs=1e-3)
    assert ev.analyse(S(task="A", a_bg="weiß", a_n=64, seed=5)).refs == {}


def test_live_b_fixed_values_and_geometry():
    a = ev.analyse(S(task="B"))
    assert a.values == {"Hann-Periodogramm": 0.0, "Multitaper (NW = 2)": 0.0, "AR (Burg, p = 16)": 1.0, "AR (Burg, BIC)": 0.0, "MUSIC (p = 4)": 1.0, "ESPRIT (p = 4)": 1.0, "ESPRIT, Linienzahl per MDL": 1.0}
    assert a.truth_lines == pytest.approx((12.8, 13.8)) and a.points["ESPRIT (p = 4)"] == pytest.approx([12.766, 13.753], abs=1e-3) and a.extra == pytest.approx({"f1_bins": 12.8, "f2_bins": 13.8})
    x, y = a.curves["Hann-Periodogramm"]
    assert len(x) == len(y) == 144 and x[0] == pytest.approx(8.8125) and x[-1] == pytest.approx(17.75) and max(y) == pytest.approx(0.0)         # dB gegen das Maximum
    wide = ev.analyse(S(task="B", b_delta=4.0, b_snr=30.0))
    assert set(wide.values.values()) == {1.0} and wide.points["ESPRIT (p = 4)"] == pytest.approx([12.802, 16.8], abs=1e-3)


def test_live_c_fixed_values_and_the_bound():
    a = ev.analyse(S(task="C"))
    assert a.title == "n = 64, 10 dB" and a.extra["crb"] == pytest.approx(0.0217958, abs=1e-6) and a.extra["fb"] == pytest.approx(15.7851, abs=1e-3)
    assert a.values == pytest.approx({"Jacobsen (3 Bins)": 0.01747, "Candan (3 Bins)": 0.01766, "Spitze (Rechteck, 16-fach, Parabel)": 0.00598, "Hann-Spitze (16-fach, Parabel)": 0.02608, "MUSIC (p = 2)": 0.01441,
                                      "ESPRIT (p = 2)": 0.00709, "AR-Spitze (Burg, p = 8)": 0.03511}, abs=1e-4)
    assert all(a.values[k] == pytest.approx(abs(a.points[k] - a.extra["fb"])) for k in a.values) and a.truth_lines == (a.extra["fb"],)
    strong = ev.analyse(S(task="C", c_n=256, c_snr=30.0))
    assert max(v for k, v in strong.values.items() if k != "AR-Spitze (Burg, p = 8)") < 0.003 and strong.values["MUSIC (p = 2)"] < 1e-3


def test_live_d_fixed_values_and_statistics():
    a = ev.analyse(S(task="D"))
    assert a.values == {"Fisher-g (weißer Hintergrund)": 0.0, "Fisher-g nach AR(2)-Vorfilterung": 1.0, "Harmonischer F-Test (NW = 4)": 1.0}
    assert a.extra["stats"] == pytest.approx((0.24250, 0.27525, 44.579), abs=1e-3) and a.extra["calibrated"] == pytest.approx([0.39029, 0.10742, 17.740], abs=1e-3)
    assert a.extra["nominal"] == pytest.approx((0.10862, 13.77059), abs=1e-4) and a.extra["nominal_decision"] == (1.0, None, 1.0)
    assert a.extra["false_alarm_nominal"] == pytest.approx({"Fisher-g (weißer Hintergrund)": 0.9925, "Harmonischer F-Test (NW = 4)": 0.15})
    assert a.truth_lines == (C.D_LINE,) and len(a.curves["Periodogramm"][0]) == 63
    assert set(ev.analyse(S(task="D", d_bg="weiß", d_snr=3.0)).values.values()) == {1.0}


def test_live_e_fixed_values_and_grids():
    a = ev.analyse(S(task="E"))
    assert a.values == {"regelmäßig": 0.0, "Jitter ±0,25": 1.0, "zufällig": 1.0, "Nachtlücken": 1.0} and a.truth_lines == (1.3,)
    assert all(len(x) == 640 and x[-1] == pytest.approx(2.0) for x, _ in a.curves.values()) and a.extra["thresholds"]["zufällig"] == pytest.approx(7.8897, abs=1e-3)
    low = ev.analyse(S(task="E", e_line=0.2, e_snr=3.0))
    assert list(low.values) == list(C.METHODS["E"]) and set(low.values.values()) == {1.0}
    assert len(low.curves[C.E_INTERP][0]) == 257 and len(low.curves["zufällig"][0]) == 160


def test_live_f_fixed_values():
    a = ev.analyse(S(task="F"))
    assert a.values == pytest.approx({"Spektrogramm (Gauß)": 0.00059, "Neu zugeordnet": 0.00052, "Synchrosqueezing": 0.00061, "CWT-Kamm (Morlet)": 0.00141, "Phasenableitung (21 geglättet)": 0.00214}, abs=2e-5)
    assert a.extra["window"] == (60, 452) and len(a.truth_curve[0]) == 512 and all(len(x) == 512 for x, _ in a.curves.values())
    assert ev.analyse(S(task="F", f_sig="Sinus-FM", f_snr=40.0)).values == pytest.approx({"Spektrogramm (Gauß)": 0.00455, "Neu zugeordnet": 0.00454, "Synchrosqueezing": 0.00546, "CWT-Kamm (Morlet)": 0.00348,
                                                                                         "Phasenableitung (21 geglättet)": 0.00204}, abs=2e-5)


def test_live_g_fixed_values_and_references():
    a = ev.analyse(S(task="G"))
    assert a.values == pytest.approx({"Wavelet-SURE (D8, weich)": 0.44174, "Wavelet universal (hart)": 0.69045, "Fourier universal (hart)": 1.1932}, abs=1e-4)
    assert a.refs == pytest.approx({"Fourier beste Schwelle (Orakel)": 1.0023, "Gleitender Mittelwert (Orakel, beste Breite)": 0.8942, "verrauscht": 1.0226}, abs=1e-3)
    assert set(a.curves) == {"Wavelet-SURE (D8, weich)", "Wavelet universal (hart)", "Fourier universal (hart)", "verrauscht"} and len(a.truth_curve[0]) == 1024
    assert ev.analyse(S(task="G", g_sig="Doppler")).values["Wavelet-SURE (D8, weich)"] == pytest.approx(0.23651, abs=1e-4)


def test_analyse_is_deterministic_and_the_seed_matters():
    for t in C.TASKS:
        a, b = ev.analyse(S(task=t)), ev.analyse(S(task=t))
        assert a.values == b.values
    assert ev.analyse(S(task="A", seed=1)).values != ev.analyse(S(task="A", seed=2)).values
    with pytest.raises(KeyError):
        ev.analyse(S(task="X"))


# --- Spitzen und Namen ----------------------------------------------------------------------------------------------------------------------


def test_extremes_for_error_tasks():
    a = analysis_with("G", {"Wavelet-SURE (D8, weich)": 0.5, "Wavelet universal (hart)": 0.5, "Fourier universal (hart)": 2.0})
    best, worst, bv, wv = ev.extremes(a)
    assert best == ["Wavelet-SURE (D8, weich)", "Wavelet universal (hart)"] and worst == ["Fourier universal (hart)"] and (bv, wv) == (0.5, 2.0)


def test_extremes_for_rate_tasks_and_names_label():
    a = analysis_with("D", {"Fisher-g (weißer Hintergrund)": 0.0, "Fisher-g nach AR(2)-Vorfilterung": 1.0, "Harmonischer F-Test (NW = 4)": 1.0})
    best, worst, bv, wv = ev.extremes(a)
    assert best == ["Fisher-g nach AR(2)-Vorfilterung", "Harmonischer F-Test (NW = 4)"] and worst == ["Fisher-g (weißer Hintergrund)"] and (bv, wv) == (1.0, 0.0)
    assert ev.names_label(best) == "2 gleichauf" and ev.names_label(worst) == "Fisher-g (weißer Hintergrund)" and ev.names_label(["a", "b", "c"]) == "3 gleichauf"


# --- Tabellen -------------------------------------------------------------------------------------------------------------------------------


def test_table_rows_format_and_fill_gaps(cells_g):
    rows = ev.table_rows("G", cells_g)
    assert rows == [{"Szenario": "Blocks", "Wavelet-SURE (D8, weich)": "1,000", "Wavelet universal (hart)": "2,000", "Fourier universal (hart)": "4,000"},
                    {"Szenario": "Bumps", "Wavelet-SURE (D8, weich)": "3,000", "Wavelet universal (hart)": "1,500", "Fourier universal (hart)": "6,000"}]
    with_refs = ev.table_rows("G", [cells_g[0], cell("G", {"signal": "Bumps"}, cells_g[1]["values"], refs={"verrauscht": 1.0})], with_refs=True)
    assert with_refs[0]["verrauscht"] == "–" and with_refs[1]["verrauscht"] == "1,000" and list(with_refs[0]) == list(with_refs[1])


def test_table_rows_use_the_digits_of_the_task(cells_d):
    assert ev.table_rows("D", cells_d)[0]["Fisher-g (weißer Hintergrund)"] == "60" and ev.table_rows("D", cells_d)[2]["Szenario"] == "AR(1) 0,8, Linie −6 dB".replace("−", "-")
    assert ev.DIGITS == {"A": 2, "B": 1, "C": 2, "D": 0, "E": 0, "F": 5, "G": 3} and ev.LIVE_DIGITS["C"] == 3


def test_summary_rows(cells_g, cells_d):
    import spv_regret as R
    rows = ev.summary_rows("G", R.summarize("G", cells_g))
    assert rows[0] == {"Verfahren": "Wavelet-SURE (D8, weich)", "mittlerer Reuegrad": "1,50", "größter Reuegrad": "2,00", "Siege": "1 von 2", "Siegesanteil": "50 %"}
    assert rows[2]["mittlerer Reuegrad"] == "4,00" and rows[2]["Siege"] == "0 von 2" and rows[2]["Siegesanteil"] == "0 %"
    rate = ev.summary_rows("D", R.summarize("D", cells_d))
    assert rate[0]["größter Reuegrad"] == "95,0 P." and rate[0]["mittlerer Reuegrad"] == "45,0 P." and rate[2]["Siege"] == "2 von 3"


def test_advice_rows(mini_results):
    adv = ADV.advise("D", {"bg": "AR(1) 0,8", "snr": -6.0}, True, mini_results)
    rows = ev.advice_rows("D", adv)
    assert rows[0] == {"Verfahren": "Fisher-g (weißer Hintergrund)", "Wert (%)": "5", "Reuegrad hier": "95,0 P.", "größter Reuegrad": "95,0 P.", "mittlerer Reuegrad": "45,0 P.", "am schlechtesten bei": "AR(1) 0,8, Linie -6 dB"}
    adv = ADV.advise("G", {"signal": "Bumps"}, False, mini_results)
    assert ev.advice_rows("G", adv)[1]["Reuegrad hier"] == "1,00" and ev.advice_rows("G", adv)[1]["Wert (mittlere quadratische Abweichung)"] == "1,500"


# --- Experimente aus der Tabelle ------------------------------------------------------------------------------------------------------------


def test_exp_spectrum_counts_wins_by_background_type():
    r = ev.exp_spectrum(71)
    assert (r["n_ar"], r["n_non_ar"]) == (15, 6) and r["share"] == 0.71 and len(r["cells"]) == 21 and r["minimax"] == "AR (Burg, BIC)"
    assert r["wins_by_type"]["AR (Burg, BIC)"] == {"ar": 15, "non_ar": 1} and sum(v["ar"] + v["non_ar"] for v in r["wins_by_type"].values()) == 21
    assert r["summary"]["AR (Burg, BIC)"]["mean"] == pytest.approx(1.13, abs=0.01) and r["equal"]["Hann-Periodogramm"]["max"] == pytest.approx(28.31, abs=0.01)
    assert ev.exp_spectrum(0)["share"] == 0.0 and ev.exp_spectrum(100)["summary"]["AR (Burg, BIC)"]["mean"] == pytest.approx(1.0)


def test_mix_curves_and_the_wins_crossover():
    c = ev.mix_curves(25)
    assert c["shares"] == [0, 25, 50, 75, 100] and set(c["mean"]) == set(C.METHODS["A"])
    assert c["mean"]["AR (Burg, BIC)"][0] == pytest.approx(1.43, abs=0.01) and c["mean"]["AR (Burg, BIC)"][-1] == pytest.approx(1.0)
    assert c["wins"]["Welch (Hann, 50 %, n/4)"][0] == pytest.approx(0.5) and c["wins"]["AR (Burg, BIC)"][-1] == pytest.approx(1.0)
    assert all(len(v) == 5 for m in ("mean", "wins") for v in c[m].values())
    # Siegesanteil AR: q + (1 - q)/6, Welch: (1 - q)/2 (AR gewinnt 15 von 15 AR-Zellen und eine von sechs, Welch drei von sechs): gleich bei q = 1/4
    assert ev.wins_crossover() == 26 and ev.wins_crossover("Welch (Hann, 50 %, n/4)", "AR (Burg, BIC)") == 0 and ev.wins_crossover("Hann-Periodogramm", "AR (Burg, BIC)") == 101


def test_the_other_experiments_return_their_cells():
    lines, det, irr, den = ev.exp_lines(), ev.exp_detection(), ev.exp_irregular(), ev.exp_denoise()
    assert (len(lines["b"]), len(lines["c"]), len(det["cells"]), len(irr["e"]), len(irr["f"]), len(den["cells"])) == (8, 6, 10, 6, 8, 4)
    assert lines["summary_b"]["ESPRIT (p = 4)"]["max"] == 0.0 and den["summary"]["Wavelet-SURE (D8, weich)"]["wins"] == 4 and det["summary"]["Fisher-g nach AR(2)-Vorfilterung"]["wins"] == 10
    o = ev.overall()
    assert list(o) == list(C.TASKS) and o["A"]["AR (Burg, BIC)"]["wins"] == 16
    assert o["E"][C.E_INTERP]["n"] == 3 and o["E"]["regelmäßig"]["n"] == 6          # das Interpolieren kennt nur die drei Zellen unter der Nyquist-Grenze
    assert RES.find("A", bg="Resonanz", n=64) is not None


# --- Kurven der Live-Ansicht gegen einen unabhängigen Rechenweg (aus dem Mutationstest) ----------------------------------------------------------


def test_live_c_curve_is_the_hann_periodogram_in_db_around_the_line():
    import signal_generator as G
    import spv_scenarios as SC
    s = S(task="C", c_n=64, c_snr=10.0, seed=7)
    a = ev.analyse(s)
    x, fb = SC.one_line(64, 10.0, G.SplitMix64(7))
    sp = np.abs(np.fft.rfft(x * SC.hann(64), 16 * 64)) ** 2
    bins = np.arange(len(sp)) / 16
    sel = (bins >= fb - 6) & (bins <= fb + 6)
    want = 10 * np.log10(sp[sel])
    xs, ys = a.curves["Hann-Periodogramm"]
    assert xs == pytest.approx(bins[sel]) and ys == pytest.approx(want - want.max()) and max(ys) == pytest.approx(0.0)
    assert xs.min() >= fb - 6 and xs.max() <= fb + 6 and xs.min() < fb - 5.9 and xs.max() > fb + 5.9


def test_live_d_curve_is_the_periodogram_without_the_dc_bin():
    import signal_generator as G
    import spv_scenarios as SC
    a = ev.analyse(S(task="D", d_bg="weiß", d_snr=3.0, seed=5))
    fr, y = a.curves["Periodogramm"]
    rng = G.SplitMix64(5)
    ph = 2 * np.pi * rng.uniform(1)[0]
    x = SC.colored_noise("weiß", 1, 128, rng)[0] + SC.amplitude(3.0) * np.cos(2 * np.pi * 0.2 * np.arange(128) + ph)
    want = 10 * np.log10(np.abs(np.fft.fft(x - x.mean())) ** 2)[1:64]
    assert fr == pytest.approx(np.arange(1, 64) / 128) and y == pytest.approx(want) and len(y) == 63
    assert fr[int(np.argmax(y))] == pytest.approx(26 / 128)                       # Linie bei 0,2 = Bin 25,6, stark über dem Rauschen: Bin 26


def test_live_e_interpolation_needs_threshold_and_location():
    thr_hit_wrong_place = ev.analyse(S(task="E", e_line=0.2, e_snr=-9.0, seed=8)).values[C.E_INTERP]
    right_place_below_thr = ev.analyse(S(task="E", e_line=0.2, e_snr=-9.0, seed=11)).values[C.E_INTERP]
    assert thr_hit_wrong_place == 0.0 and right_place_below_thr == 0.0
