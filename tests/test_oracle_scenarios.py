"""Szenarien gegen Handrechnung: Amplitude, Fenster, FIR-Gang, Hintergründe (Kerbe, Bandpass, AR), Linien, Rauschen, Testsignale."""

import math

import numpy as np
import pytest

import signal_generator as G
import spv_constants as C
import spv_scenarios as S


def test_amplitude_from_snr():
    assert S.amplitude(0.0) == pytest.approx(math.sqrt(2.0))                 # SNR = A^2 / 2 = 1
    assert S.amplitude(10.0) == pytest.approx(math.sqrt(20.0)) and S.amplitude(-10.0) == pytest.approx(math.sqrt(0.2))


def test_hann_window_by_hand():
    w = S.hann(4)                                                           # 0,5 - 0,5 cos(2 pi (m + 1/2) / 4)
    c = 0.5 - 0.5 * math.cos(math.pi / 4)
    assert w == pytest.approx([c, 1.0 - c, 1.0 - c, c]) and c == pytest.approx(0.1464466)
    assert S.hann(64).sum() == pytest.approx(32.0) and S.hann(64).max() < 1.0


def test_fir_response_of_the_notch():
    h = C.NOTCH                                                              # 1 + 0,9409 z^-2: bei f = 0,25 gilt exp(-i 2 pi 0,25 2) = -1
    assert S.fir_response(h, 0.25)[0] == pytest.approx(1.0 - 0.9409)
    assert S.fir_response(h, 0.0)[0] == pytest.approx(1.9409) and S.fir_response(h, 0.5)[0] == pytest.approx(1.9409)
    assert S.fir_response(h, [0.0, 0.25]).shape == (2,)


def test_bandpass_taps_are_normalised_symmetric_and_selective():
    h = S.bandpass_taps()
    assert len(h) == C.BAND_TAPS and np.sum(h ** 2) == pytest.approx(1.0) and np.allclose(h, h[::-1])
    assert abs(S.fir_response(h, 0.15)[0]) ** 2 == pytest.approx(6.4, rel=0.02)         # Durchlass (Parseval: Mittel 5, Welligkeit)
    assert abs(S.fir_response(h, 0.4)[0]) ** 2 < 1e-6 and abs(S.fir_response(h, 0.02)[0]) ** 2 < 0.2


def test_fir_noise_shape_variance_and_true_spectrum():
    x, f, true = S.fir_noise(200, 256, C.NOTCH, 5)
    assert x.shape == (200, 256) and f.shape == (129,) and f[0] == 0.0 and f[-1] == 0.5 and f[64] == 0.25
    assert x.var() == pytest.approx(1.0 + 0.9409 ** 2, rel=0.05)           # Rauschleistung = Summe h^2 bei weißem Eingang
    assert true[0] == pytest.approx(1.9409 ** 2) and true[64] == pytest.approx(0.0591 ** 2)
    x2, _, _ = S.fir_noise(200, 256, C.NOTCH, 5)
    assert np.array_equal(x, x2)
    assert not np.array_equal(x, S.fir_noise(200, 256, C.NOTCH, 6)[0])


def test_ar_noise_white_and_with_extra_white_noise():
    x, f, true = S.ar_noise(100, 256, (), None, 5)
    assert x.var() == pytest.approx(1.0, rel=0.05) and true == pytest.approx(np.ones(129))
    x, f, true = S.ar_noise(100, 256, (), 10.0, 5)                          # weiß mit Leistung 1 plus weißes Rauschen 10 dB darunter: 1,1
    assert x.var() == pytest.approx(1.1, rel=0.05) and true == pytest.approx(np.full(129, 1.1))


def test_ar_noise_true_spectrum_matches_the_generator():
    poles = C.POLES["Resonanz"]
    x, f, true = S.ar_noise(100, 256, poles, None, 5)
    ar = G.ar_from_poles(poles)
    assert true == pytest.approx(G.true_psd(f, 1.0, poles, G.ar_unit_sigma(ar), 0.0))
    assert x.var() == pytest.approx(1.0, rel=0.1) and true.argmax() == round(0.2 * 256)      # Resonanz bei 0,2


def test_background_dispatch():
    for bg, taps in (("Kerbe", C.NOTCH), ("Bandpass", S.bandpass_taps())):
        x, f, true = S.background(bg, 3, 64, 11)
        x2, f2, true2 = S.fir_noise(3, 64, taps, 11)
        assert np.array_equal(x, x2) and np.array_equal(true, true2)
    for bg in C.AR_TYPE:
        x, f, true = S.background(bg, 3, 64, 11)
        x2, f2, true2 = S.ar_noise(3, 64, C.POLES[bg], C.EXTRA_DB.get(bg), 11)
        assert np.array_equal(x, x2) and np.array_equal(true, true2)
    assert S.background("Resonanz + weißes Rauschen", 1, 64, 11)[2].min() > S.background("Resonanz", 1, 64, 11)[2].min()


def test_two_lines_have_the_requested_distance_and_noise():
    x, f2 = S.two_lines(64, 0.2, 4.0, 30.0, G.SplitMix64(7))
    assert f2 == pytest.approx(0.2 + 4.0 / 64) and len(x) == 64
    sp = np.abs(np.fft.rfft(x * S.hann(64), 64 * 16)) ** 2
    bins = np.arange(len(sp)) / 16
    peaks = [i for i in range(1, len(sp) - 1) if sp[i] > sp[i - 1] and sp[i] > sp[i + 1]]
    top = sorted(bins[sorted(peaks, key=lambda i: -sp[i])[:2]])
    assert top == pytest.approx([12.8, 16.8], abs=0.3)                       # 0,2 n = 12,8 und 12,8 + 4
    x0, _ = S.two_lines(4096, 0.2, 4.0, -60.0, G.SplitMix64(7))
    assert x0.var() == pytest.approx(1.0, rel=0.1)                           # fast nur Rauschen mit Varianz 1


def test_one_line_frequency_range_and_power():
    fbs = [S.one_line(64, 0.0, G.SplitMix64(s))[1] for s in range(200)]
    assert 0.15 * 64 <= min(fbs) and max(fbs) <= 0.35 * 64 and np.mean(fbs) == pytest.approx(0.25 * 64, abs=1.0)
    x, fb = S.one_line(1024, 60.0, G.SplitMix64(3))
    assert x.var() == pytest.approx(1e6, rel=0.05) and len(x) == 1024


def test_colored_noise_variance_and_correlation():
    rng = G.SplitMix64(3)
    ar = S.colored_noise("AR(1) 0,8", 100, 128, rng)
    assert ar.shape == (100, 128) and ar.var() == pytest.approx(1.0, rel=0.1)
    assert np.mean(ar[:, 1:] * ar[:, :-1]) / ar.var() == pytest.approx(0.8, abs=0.05)
    white = S.colored_noise("weiß", 100, 128, rng)
    assert white.var() == pytest.approx(1.0, rel=0.05) and np.mean(white[:, 1:] * white[:, :-1]) / white.var() == pytest.approx(0.0, abs=0.05)


def test_time_frequency_signals():
    c = S.tf_signal("Chirp", 512)
    assert len(c.x) == 512 and c.tracks[0][0] == pytest.approx(C.CHIRP_F0) and c.tracks[0][-1] == pytest.approx(C.CHIRP_F1)
    assert c.tracks[0][256] == pytest.approx(C.CHIRP_F0 + (C.CHIRP_F1 - C.CHIRP_F0) * 256 / 511)
    s = S.tf_signal("Sinus-FM", 512)
    assert s.tracks[0].min() == pytest.approx(C.SINFM_FC - C.SINFM_DF) and s.tracks[0].max() == pytest.approx(C.SINFM_FC + C.SINFM_DF) and s.tracks[0].mean() == pytest.approx(C.SINFM_FC, abs=0.01)
    assert len(S.tf_signal("Chirp").x) == C.N_F
    with pytest.raises(ValueError):
        S.tf_signal("Unsinn")
