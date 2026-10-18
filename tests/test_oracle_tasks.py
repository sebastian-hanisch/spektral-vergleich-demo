"""Aufgaben (spv_tasks.py): Schätzer gegen scipy und Handrechnung, Zellen gegen feste Werte kleiner Läufe (Eigenschaften der Schätzer in den Quellrepos der Reihe geprüft)."""

import math

import numpy as np
import pytest
import scipy.signal as sg

import signal_generator as G
import spv_constants as C
import spv_estimators as E
import spv_scenarios as S
import spv_tasks as T

ALL_A = C.METHODS["A"]


def white(n, seed=1):
    return G.SplitMix64(seed).normal(n)


# --- A Spektrum schätzen --------------------------------------------------------------------------------------------------------------------


def test_tapers_are_orthonormal_and_cached():
    h = T.tapers(256, 4)
    assert h.shape == (7, 256) and h @ h.T == pytest.approx(np.eye(7), abs=1e-9) and T.tapers(256, 4) is h
    assert T.tapers(64, 2).shape == (3, 64)


def test_burg_fit_of_a_pure_sinusoid():
    a, s2 = T.burg_fit(np.cos(0.7 * np.arange(200)), 2)                    # x[t] = 2 cos(0,7) x[t-1] - x[t-2]
    assert a == pytest.approx([2 * math.cos(0.7), -1.0], abs=5e-3) and 0 <= s2 < 1e-4        # Vorhersagekoeffizienten: x[t] = a1 x[t-1] + a2 x[t-2]


def test_bic_order_finds_the_true_order():
    xr, _, _ = S.ar_noise(1, 1024, C.POLES["Resonanz"], None, 5)
    xw, _, _ = S.ar_noise(1, 1024, (), None, 5)
    assert T.bic_order(xr[0], 1024) == 2 and T.bic_order(xw[0], 1024) == 1


def test_a_estimates_against_scipy():
    n = 256
    x, f = white(n), np.arange(n // 2 + 1) / n
    est = T.a_estimates(x, n, f)
    assert list(est) == list(ALL_A)
    w = S.hann(n)
    _, p = sg.periodogram(x, window=w, detrend=False)
    assert est["Hann-Periodogramm"][0] == pytest.approx(p[0]) and est["Hann-Periodogramm"][1:-1] == pytest.approx(p[1:-1] / 2)       # scipy verdoppelt die inneren Bins einseitig
    seg = n // 4
    _, pw = sg.welch(x, fs=1.0, window=S.hann(seg), nperseg=seg, noverlap=seg // 2, detrend=False)
    assert est["Welch (Hann, 50 %, n/4)"][::4][1:-1] == pytest.approx(pw[1:-1] / 2) and len(est["Welch (Hann, 50 %, n/4)"]) == n // 2 + 1
    man = np.mean([np.abs(np.fft.rfft(hk * x)) ** 2 for hk in T.tapers(n, 4)], axis=0)
    assert est["Multitaper (NW = 4)"] == pytest.approx(man)


def test_a_estimates_ar_recovers_the_spectrum_of_an_ar_process():
    x, f, true = S.ar_noise(1, 4096, ((0.0, 0.8),), None, 9)
    ar = T.a_estimates(x[0], 4096, f, ("AR (Burg, BIC)",))["AR (Burg, BIC)"]
    assert list(T.a_estimates(x[0], 4096, f, ("AR (Burg, BIC)",))) == ["AR (Burg, BIC)"] and np.median(ar / true) == pytest.approx(1.0, abs=0.05)


def test_a_estimates_reference_order_adds_a_reference_only_when_asked():
    x, f, _ = S.ar_noise(1, 256, C.POLES["Resonanz"], None, 5)
    assert "AR (Burg, feste Ordnung)" not in T.a_estimates(x[0], 256, f)
    est = T.a_estimates(x[0], 256, f, ALL_A, ref_order=2)
    assert "AR (Burg, feste Ordnung)" in est and est["AR (Burg, feste Ordnung)"].shape == f.shape


def test_log_error_by_hand():
    true = np.ones(10)
    assert T.log_error(np.full(10, 10.0), true) == pytest.approx(10.0)                      # Faktor 10 = 10 dB
    est = np.ones(10)
    est[4], est[5] = 10.0, 0.1                                                              # nur die inneren Bins 4 bis 5 zählen: +10 dB und -10 dB
    est[0] = est[9] = 1e6                                                                   # Ränder ohne Wirkung
    assert T.log_error(est, true) == pytest.approx(10.0)
    est[5] = 1.0
    assert T.log_error(est, true) == pytest.approx(math.sqrt(50.0))                         # sqrt((10^2 + 0)/2)


def test_cell_a_fixed_values_and_structure():
    c = T.cell_a("weiß", 64, 2)
    assert list(c["values"]) == list(ALL_A) and c["refs"] == {} and c["extra"] == {"runs": 2}
    assert c["values"] == pytest.approx({"Hann-Periodogramm": 5.4094, "Welch (Hann, 50 %, n/4)": 0.7301, "Multitaper (NW = 4)": 0.8866, "AR (Burg, BIC)": 0.3064}, abs=1e-3)
    k = T.cell_a("Kerbe", 64, 2)
    assert k["values"] == pytest.approx({"Hann-Periodogramm": 4.6099, "Welch (Hann, 50 %, n/4)": 4.9947, "Multitaper (NW = 4)": 4.0701, "AR (Burg, BIC)": 5.8574}, abs=1e-3)
    assert k["refs"] == pytest.approx({"AR (Burg, feste Ordnung)": 4.1133}, abs=1e-3)
    assert T.cell_a("Kerbe", 64, 2) == k


# --- B Zwei Linien trennen ------------------------------------------------------------------------------------------------------------------


def two_clean_lines(delta):
    n = 64
    t = np.arange(n)
    f2 = 0.2 + delta / n
    return 30 * (np.cos(2 * np.pi * 0.2 * t) + np.cos(2 * np.pi * f2 * t + 1)) + white(n, 5), f2


def test_b_spectra_have_the_padded_grid():
    x, _ = two_clean_lines(4.0)
    spectra = T.b_spectra(x, 64)
    assert list(spectra) == ["Hann-Periodogramm", "Multitaper (NW = 2)", "AR (Burg, p = 16)", "AR (Burg, BIC)", "MUSIC (p = 4)"]
    assert all(v.shape == (513,) for v in spectra.values())


def test_b_outcomes_wide_lines_are_resolved_by_everything():
    x, f2 = two_clean_lines(4.0)
    assert T.b_outcomes(x, 64, 0.2, f2) == {name: True for name in C.METHODS["B"]}


def test_b_outcomes_lines_below_one_bin_are_not_resolved_by_the_periodogram_family():
    x, f2 = two_clean_lines(0.25)
    out = T.b_outcomes(x, 64, 0.2, f2)
    assert not out["Hann-Periodogramm"] and not out["Multitaper (NW = 2)"] and not out["AR (Burg, BIC)"]


def test_cell_b_fixed_values():
    c = T.cell_b(1.0, 10.0, 6)
    assert list(c["values"]) == list(C.METHODS["B"]) and c["extra"] == {"runs": 6}
    assert c["values"] == pytest.approx({"Hann-Periodogramm": 100 / 3, "Multitaper (NW = 2)": 0.0, "AR (Burg, p = 16)": 200 / 3, "AR (Burg, BIC)": 0.0, "MUSIC (p = 4)": 250 / 3, "ESPRIT (p = 4)": 100.0,
                                        "ESPRIT, Linienzahl per MDL": 100.0})
    assert set(T.cell_b(2.5, 20.0, 4)["values"].values()) == {0.0, 100.0} and T.cell_b(2.5, 20.0, 4)["values"]["Multitaper (NW = 2)"] == 0.0


# --- C Eine Linie genau treffen -------------------------------------------------------------------------------------------------------------


def test_c_estimates_of_a_noise_free_line():
    n, fb = 64, 16.3
    est = T.c_estimates(np.cos(2 * np.pi * fb / n * np.arange(n) + 0.4), n)
    assert set(est) == set(C.METHODS["C"])
    assert est["MUSIC (p = 2)"] == pytest.approx(fb, abs=1e-3) and est["ESPRIT (p = 2)"] == pytest.approx(fb, abs=1e-6)
    for k in ("Jacobsen (3 Bins)", "Candan (3 Bins)", "Spitze (Rechteck, 16-fach, Parabel)", "Hann-Spitze (16-fach, Parabel)"):
        assert est[k] == pytest.approx(fb, abs=0.01), k
    assert est["AR-Spitze (Burg, p = 8)"] == pytest.approx(fb, abs=0.1)


def test_cell_c_fixed_values_and_the_bound():
    c = T.cell_c(64, 10.0, 5)
    crb = math.sqrt(3.0 / (math.pi ** 2 * 10.0 * 64 * (64 ** 2 - 1.0))) * 64
    assert c["extra"]["crb"] == pytest.approx(crb) and crb == pytest.approx(0.0217958, abs=1e-6)
    assert c["values"] == pytest.approx({"Spitze (Rechteck, 16-fach, Parabel)": 0.4274, "Jacobsen (3 Bins)": 1.0139, "Candan (3 Bins)": 1.0142, "Hann-Spitze (16-fach, Parabel)": 1.2621,
                                         "MUSIC (p = 2)": 1.06, "ESPRIT (p = 2)": 0.686, "AR-Spitze (Burg, p = 8)": 3.4414}, abs=1e-3)
    assert {k: v * crb for k, v in c["values"].items()} == pytest.approx(c["extra"]["rmse"])


# --- D Eine Linie im Rauschen finden --------------------------------------------------------------------------------------------------------


def test_d_stats_of_a_pure_line_and_of_noise():
    n = 128
    line = np.sin(2 * np.pi * 16 * np.arange(n) / n)
    g, g_pre, f = T.d_stats(line, n)
    assert g == pytest.approx(1.0, abs=1e-9) and f > 1e5                          # ein einziger Bin trägt die ganze Leistung
    g, g_pre, f = T.d_stats(white(n, 3), n)
    assert 1 / 63 < g < 0.2 and 1 / 63 < g_pre < 0.2 and 0 < f < 20


def test_d_nominal_thresholds():
    g, f = T.d_nominal_thresholds(128)
    assert E.fisher_p(g, 63) == pytest.approx(0.05) and g == pytest.approx(0.10862, abs=1e-4)
    assert f == pytest.approx(6.0 * (1280.0 ** (1.0 / 6.0) - 1.0)) and f == pytest.approx(13.7706, abs=1e-3)       # 6 (1280^(1/6) - 1)


def test_cells_d_fixed_values_and_structure():
    d = T.cells_d("weiß", 6, 20)
    assert list(d) == list(C.D_SNRS)
    c = d[-12.0]
    assert list(c["values"]) == list(C.METHODS["D"]) and c["extra"]["runs"] == 6 and c["extra"]["null_runs"] == 20
    assert c["values"] == pytest.approx({"Fisher-g (weißer Hintergrund)": 100 / 3, "Fisher-g nach AR(2)-Vorfilterung": 50.0, "Harmonischer F-Test (NW = 4)": 100 / 3})
    assert c["extra"]["thresholds"] == pytest.approx([0.0927, 0.0927, 14.026], abs=1e-3)
    assert c["extra"]["nominal_thresholds"] == pytest.approx([0.10862, 13.77059], abs=1e-4)
    assert c["extra"]["false_alarm_nominal"] == pytest.approx({"Fisher-g (weißer Hintergrund)": 0.0, "Harmonischer F-Test (NW = 4)": 0.05})
    assert d[-6.0]["extra"]["nominal_rate"] == pytest.approx({"Fisher-g (weißer Hintergrund)": 200 / 3, "Harmonischer F-Test (NW = 4)": 250 / 3})
    assert d[-9.0]["values"]["Fisher-g (weißer Hintergrund)"] == pytest.approx(250 / 3)


# --- E Unregelmäßige Messzeiten -------------------------------------------------------------------------------------------------------------


def test_e_grid_depends_on_the_line_side_of_nyquist():
    hi, lo = T.e_grid(1.3), T.e_grid(0.2)
    assert hi[0] == pytest.approx(0.003125) and hi[-1] == pytest.approx(2.0) and len(hi) == 640
    assert lo[0] == pytest.approx(0.003125) and lo[-1] == pytest.approx(0.5) and len(lo) == 160


def test_e_series_times_and_values():
    t, y = T.e_series("regelmäßig", 0.2, 10.0, G.SplitMix64(2))
    assert t == pytest.approx(np.arange(64.0)) and len(y) == 64
    t, y = T.e_series("zufällig", 0.2, 10.0, G.SplitMix64(2))
    assert np.all(np.diff(t) >= 0) and t.min() >= 0 and t.max() <= 64


def test_e_interp_spectrum_of_a_regular_line():
    fg, sp = T.e_interp_spectrum(np.arange(64.0), np.cos(2 * np.pi * 0.2 * np.arange(64)))
    assert len(fg) == 257 and fg[0] == 0.0 and fg[-1] == 0.5 and sp[:8].max() == 0.0 and sp[-8:].max() == 0.0
    assert fg[np.argmax(sp)] == pytest.approx(0.2, abs=0.01)


def test_cell_e_fixed_values():
    c = T.cell_e(1.3, 0.0, 5)
    assert c["values"] == pytest.approx({"regelmäßig": 0.0, "Jitter ±0,25": 100.0, "zufällig": 100.0, "Nachtlücken": 100.0}) and list(c["extra"]["thresholds"]) == list(C.E_KINDS)
    c = T.cell_e(0.2, 0.0, 5)
    assert c["values"] == pytest.approx({"regelmäßig": 100.0, "Jitter ±0,25": 100.0, "zufällig": 100.0, "Nachtlücken": 40.0, C.E_INTERP: 60.0})
    assert c["extra"]["thresholds"]["regelmäßig"] == pytest.approx(5.1843, abs=1e-3) and c["extra"]["thresholds"][C.E_INTERP] == pytest.approx(135.668, abs=1e-2)


# --- F Zeitabhängige Frequenz ---------------------------------------------------------------------------------------------------------------


def test_ridge_window_and_rmse_by_hand():
    assert T.ridge_window(512) == (60, 452) and T.ridge_window(256) == (30, 226)
    assert T.f_rmse(np.zeros(512), np.zeros(512), 512) == 0.0
    assert T.f_rmse(np.arange(10) * 0.01, np.arange(10) * 0.01 + 0.02, 10) == pytest.approx(0.02)                # n = 10: Fenster 1 bis 9, konstanter Versatz 0,02
    est = np.zeros(512)
    est[60:452:2] = 0.1                                                                                         # jeder zweite Wert im Fenster um 0,1 daneben, Ränder ohne Wirkung
    est[:60] = 5.0
    assert T.f_rmse(est, np.zeros(512), 512) == pytest.approx(0.1 / math.sqrt(2.0), rel=0.01)


def test_f_estimates_of_a_noise_free_chirp():
    sig = S.tf_signal("Chirp", 512)
    est = T.f_estimates(sig.x, 512)
    assert list(est) == list(C.METHODS["F"])
    for k, v in est.items():
        assert len(v) == 512 and T.f_rmse(v, sig.tracks[0], 512) < 0.001, k


def test_cell_f_fixed_values():
    c = T.cell_f("Chirp", None)
    assert c["extra"] == {"runs": 1} and c["values"]["Neu zugeordnet"] < 1e-4 and c["values"]["Spektrogramm (Gauß)"] == pytest.approx(0.0003, abs=1e-4)
    s = T.cell_f("Sinus-FM", 20.0, 2)
    assert s["values"] == pytest.approx({"Spektrogramm (Gauß)": 0.0046, "Neu zugeordnet": 0.0046, "Synchrosqueezing": 0.0073, "CWT-Kamm (Morlet)": 0.0035, "Phasenableitung (21 geglättet)": 0.0021}, abs=1e-4)


# --- G Entrauschen --------------------------------------------------------------------------------------------------------------------------


def test_fourier_threshold_by_hand():
    y = white(64, 4)
    assert T.fourier_threshold(y, 0.0) == pytest.approx(y)
    only_dc_nyquist = T.fourier_threshold(y, 1e9)
    sign = (-1.0) ** np.arange(64)
    assert only_dc_nyquist == pytest.approx(y.mean() + sign * np.mean(sign * y))                  # Gleichanteil und Nyquist bleiben
    assert T.fourier_threshold(np.ones(64), 1.0) == pytest.approx(np.ones(64))


def test_g_estimates_and_references():
    n = C.N_G
    y = white(n, 6)
    est = T.g_estimates(y)
    assert list(est) == list(C.METHODS["G"]) and all(len(v) == n for v in est.values())
    truth = np.zeros(n)
    refs = T.g_references(y, truth)
    assert list(refs) == ["Fourier beste Schwelle (Orakel)", "Gleitender Mittelwert (Orakel, beste Breite)"]
    assert refs["Fourier beste Schwelle (Orakel)"] <= np.mean(T.fourier_threshold(y, 0.5) ** 2) and refs["Gleitender Mittelwert (Orakel, beste Breite)"] <= np.mean(np.convolve(y, np.ones(65) / 65, "same") ** 2) + 0.01
    assert est["Wavelet-SURE (D8, weich)"].std() < 0.5 and est["Fourier universal (hart)"].std() < 0.5       # reines Rauschen wird fast ganz entfernt


def test_cell_g_fixed_values():
    c = T.cell_g("Blocks", 3)
    assert c["values"] == pytest.approx({"Wavelet-SURE (D8, weich)": 0.3773, "Wavelet universal (hart)": 0.5224, "Fourier universal (hart)": 1.2538}, abs=1e-3)
    assert c["refs"] == pytest.approx({"verrauscht": 1.0046, "Fourier beste Schwelle (Orakel)": 0.9673, "Gleitender Mittelwert (Orakel, beste Breite)": 0.8517}, abs=1e-3)


# --- Hilfsfunktionen aus dem Mutationstest --------------------------------------------------------------------------------------------------


def test_snapshot_length_by_hand():
    assert T.snapshot_length(64) == 21 and T.snapshot_length(256) == 85 and T.snapshot_length(30) == 12 and T.snapshot_length(36) == 12 and T.snapshot_length(39) == 13


def test_esprit_resolved_rules():
    f1, f2 = 0.2, 0.2 + 1.0 / 64                                                   # halber Abstand: 1/128
    tol = 0.5 * (f2 - f1)
    assert T.esprit_resolved([f1, f2], f1, f2) and T.esprit_resolved([f1 + 0.9 * tol, f2 - 0.9 * tol], f1, f2)
    assert not T.esprit_resolved([f1 + 1.1 * tol, f2], f1, f2) and not T.esprit_resolved([f1, f2 - 1.1 * tol], f1, f2)
    assert not T.esprit_resolved([f1 - 1.1 * tol, f2], f1, f2) and not T.esprit_resolved([f1, f2 + 1.1 * tol], f1, f2)
    assert not T.esprit_resolved([f1], f1, f2) and not T.esprit_resolved([f1, f2, 0.3], f1, f2) and not T.esprit_resolved([], f1, f2)
    assert not T.esprit_resolved([f1 + 3 * tol, f2 + 3 * tol], f1, f2)                # beide Schätzungen weit daneben
