"""Szenarien der sieben Aufgaben: Hintergründe mit wahrem Spektrum (AR, Kerbe, Bandpass), Linien, Messzeiten, Signale. Zufall nur über SplitMix64.

Frequenzen als Bruchteil der Abtastrate (Abtastrate 1); Rauschvarianz 1, Amplitude A einer Linie mit SNR = A^2 / 2 je Abtastwert (dB).
"""

import math

import numpy as np

import signal_generator as G
import spv_constants as C

TWO_PI = 2.0 * math.pi


def amplitude(snr_db: float) -> float:
    """Amplitude einer Linie mit dem Rauschabstand snr_db je Abtastwert gegen die Rauschvarianz 1."""
    return math.sqrt(2.0 * 10.0 ** (snr_db / 10.0))


def hann(n: int) -> np.ndarray:
    """Hann-Fenster 0,5 - 0,5 cos(2 pi (m + 1/2) / n)."""
    return 0.5 - 0.5 * np.cos(TWO_PI * (np.arange(n) + 0.5) / n)


def fir_response(h, f) -> np.ndarray:
    """Frequenzgang H(f) = sum_k h_k exp(-i 2 pi f k)."""
    h = np.asarray(h, float)
    return np.sum(h[:, None] * np.exp(-2j * np.pi * np.outer(np.arange(len(h)), np.atleast_1d(f))), axis=0)


def bandpass_taps(lo: float = C.BAND_LO, hi: float = C.BAND_HI, length: int = C.BAND_TAPS, beta: float = C.BAND_BETA) -> np.ndarray:
    """FIR-Bandpass (gefenstertes Sinc, Kaiser-Fenster), auf Rauschleistung 1 normiert."""
    k = np.arange(length) - (length - 1) / 2
    h = (2 * hi * np.sinc(2 * hi * k) - 2 * lo * np.sinc(2 * lo * k)) * np.kaiser(length, beta)
    return h / np.sqrt(np.sum(h ** 2))


def fir_noise(runs: int, n: int, h, seed: int):
    """Weißes Rauschen durch das FIR-Filter h (Einschwingen verworfen): (x, Frequenzen k / n, wahres Spektrum |H|^2)."""
    h = np.asarray(h, float)
    length = len(h)
    w = G.SplitMix64(seed).normal(runs * (n + length)).reshape(runs, n + length)
    x = np.stack([np.convolve(r, h, mode="full")[length:length + n] for r in w])
    f = np.arange(n // 2 + 1) / n
    return x, f, np.abs(fir_response(h, f)) ** 2


def ar_noise(runs: int, n: int, poles, extra_db, seed: int):
    """Rauschen aus AR-Polen (Varianz 1), optional plus weißes Rauschen mit der Leistung extra_db unter dem AR-Anteil: (x, Frequenzen k / n, wahres Spektrum)."""
    ar = G.ar_from_poles(poles) if poles else None
    ar_sigma = G.ar_unit_sigma(ar) if poles else 0.0
    white = 0.0 if poles else 1.0
    if extra_db is not None:
        white = math.sqrt(white ** 2 + 10.0 ** (-extra_db / 10.0))
    x = G.make_noise_batch(runs, n, poles, ar_sigma, white, G.SplitMix64(seed))
    f = np.arange(n // 2 + 1) / n
    return x, f, G.true_psd(f, 1.0, poles, ar_sigma, white)


def background(bg: str, runs: int, n: int, seed: int):
    """Hintergrund der Aufgabe A: (x, f, wahres Spektrum)."""
    if bg == "Kerbe":
        return fir_noise(runs, n, C.NOTCH, seed)
    if bg == "Bandpass":
        return fir_noise(runs, n, bandpass_taps(), seed)
    return ar_noise(runs, n, C.POLES[bg], C.EXTRA_DB.get(bg), seed)


def two_lines(n: int, f1: float, delta_bins: float, snr_db: float, rng):
    """Zwei gleich starke Linien bei f1 und f1 + delta/n mit zufälligen Phasen plus Rauschen: (x, f2)."""
    ph = TWO_PI * rng.uniform(2)
    t = np.arange(n)
    f2 = f1 + delta_bins / n
    x = amplitude(snr_db) * (np.cos(TWO_PI * f1 * t + ph[0]) + np.cos(TWO_PI * f2 * t + ph[1])) + rng.normal(n)
    return x, f2


def one_line(n: int, snr_db: float, rng):
    """Eine Linie mit zufälliger Frequenz (0,15 bis 0,35 f_s) und Phase plus Rauschen: (x, Frequenz in Bins)."""
    u = rng.uniform(2)
    fb = (0.15 + 0.2 * u[0]) * n
    t = np.arange(n)
    return amplitude(snr_db) * np.cos(TWO_PI * fb / n * t + TWO_PI * u[1]) + rng.normal(n), fb


def colored_noise(bg: str, runs: int, n: int, rng):
    """Rauschen der Aufgabe D (weiß oder AR(1)) aus dem laufenden Strom rng: (runs, n)."""
    poles = C.D_POLES[bg]
    ar = G.ar_from_poles(poles) if poles else None
    ar_sigma = G.ar_unit_sigma(ar) if poles else 0.0
    white = 0.0 if poles else 1.0
    return G.make_noise_batch(runs, n, poles, ar_sigma, white, rng)


def tf_signal(name: str, n: int = C.N_F):
    """Testsignal der Aufgabe F (Chirp oder Sinus-FM) mit wahrer Momentanfrequenz."""
    import spv_timefreq as TF
    if name == "Chirp":
        return TF.linear_chirp(n, C.CHIRP_F0, C.CHIRP_F1)
    if name == "Sinus-FM":
        return TF.sine_fm(n, C.SINFM_FC, C.SINFM_DF, C.SINFM_FM)
    raise ValueError(f"unbekanntes Signal: {name}")
