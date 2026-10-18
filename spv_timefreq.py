"""Zeit-Frequenz-Verfahren der Vorgänger-Demos, wortgleich übernommen (Gauß-STFT, Neuzuordnung und Synchrosqueezing, Morlet-CWT mit Kamm, analytisches Signal). Reine Funktionen, nur numpy.
"""

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi
PI14 = math.pi ** -0.25
NFFT = 1024
DEFAULT_FLOOR = -60.0
W0 = 6.0


# --- aus rea_stft.py (reassignment-demo) ----------------------------------------

def gauss_windows(sigma: float):
    """(k, g, dg, tg) mit k = -K .. K."""
    K = int(math.ceil(4.0 * sigma))
    k = np.arange(-K, K + 1)
    g = np.exp(-(k.astype(float) ** 2) / (2.0 * sigma ** 2))
    return k, g, -k / sigma ** 2 * g, k * g


def stft3(z, sigma: float, nfft: int = NFFT):
    """Drei STFTs (Fenster, Ableitungsfenster, Zeitgewichtsfenster) der Reihe z; jede Form (n, nfft), komplex, Frequenz m / nfft."""
    z = np.asarray(z, complex)
    n = len(z)
    k, g, dg, tg = gauss_windows(sigma)
    K = (len(k) - 1) // 2
    if 2 * K + 1 > nfft:
        raise ValueError("Fenster länger als die FFT")
    pad = np.concatenate([np.zeros(K, complex), z, np.zeros(K, complex)])
    frames = np.lib.stride_tricks.sliding_window_view(pad, 2 * K + 1)
    out = []
    for w in (g, dg, tg):
        a = np.zeros((n, nfft), complex)
        a[:, :K + 1] = frames[:, K:] * w[K:]
        a[:, nfft - K:] = frames[:, :K] * w[:K]
        out.append(np.fft.fft(a, axis=1))
    return out


def freq_axis(nfft: int = NFFT) -> np.ndarray:
    return np.arange(nfft) / nfft


# --- aus rea_reassign.py (reassignment-demo) ----------------------------------------

def reassign_fields(z, sigma: float, nfft: int = NFFT, floor_db: float = DEFAULT_FLOOR, abs_floor: float = 0.0):
    """(X, f_hat, t_hat, Maske): STFT, neu zugeordnete Frequenz (Zyklen je Abtastwert) und Zeit je Punkt; Maske: Leistung über floor_db gegen die Spitze und Betrag über abs_floor."""
    X, Xd, Xt = stft3(z, sigma, nfft)
    w = TWO_PI * freq_axis(nfft)[None, :]
    safe = np.where(np.abs(X) > 0, X, 1.0)
    f_hat = (w - np.imag(Xd / safe)) / TWO_PI
    t_hat = np.arange(len(z))[:, None] + np.real(Xt / safe)
    p = np.abs(X) ** 2
    mask = (p > 10.0 ** (floor_db / 10.0) * p.max()) & (np.abs(X) > abs_floor)
    return X, f_hat, t_hat, mask


def _accumulate(rows, cols, weights, shape):
    out = np.bincount(rows * shape[1] + cols, weights=weights, minlength=shape[0] * shape[1])
    return out.reshape(shape)


def reassigned_from(X, t_hat, f_hat, mask, nfft: int = NFFT) -> np.ndarray:
    """Neu zugeordnetes Spektrogramm aus den Feldern von reassign_fields: die Leistung jedes Punktes wandert auf den nächsten Rasterpunkt von (t_hat, f_hat); Form (n, nfft)."""
    n = X.shape[0]
    ti = np.rint(t_hat).astype(int)
    fi = np.rint(f_hat * nfft).astype(int)
    ok = mask & (ti >= 0) & (ti < n) & (fi >= 0) & (fi < nfft)
    return _accumulate(ti[ok], fi[ok], (np.abs(X) ** 2)[ok], (n, nfft))


def squeezed_from(X, f_hat, mask, nfft: int = NFFT) -> np.ndarray:
    """Synchrosqueezing (nur in der Frequenz) aus den Feldern: die komplexen STFT-Werte werden auf dem Rasterpunkt der neu zugeordneten Frequenz gesammelt; Form (n, nfft), komplex."""
    n = X.shape[0]
    fi = np.rint(f_hat * nfft).astype(int)
    ok = mask & (fi >= 0) & (fi < nfft)
    rows = np.broadcast_to(np.arange(n)[:, None], fi.shape)
    r, c, v = rows[ok], fi[ok], X[ok]
    return _accumulate(r, c, v.real, (n, nfft)) + 1j * _accumulate(r, c, v.imag, (n, nfft))


# --- aus rea_signals.py (reassignment-demo) ----------------------------------------

def analytic(x) -> np.ndarray:
    """Analytisches Signal über die FFT (nur positive Frequenzen, doppelt gewichtet); der Realteil ist die Eingabe."""
    x = np.asarray(x, float)
    n = len(x)
    h = np.zeros(n)
    h[0] = 1.0
    h[1:(n + 1) // 2] = 2.0
    if n % 2 == 0:
        h[n // 2] = 1.0
    return np.fft.ifft(np.fft.fft(x) * h)


def noise_sigma(snr_db: float) -> float:
    """Standardabweichung reellen weißen Rauschens, dessen Leistung gegen die einer Linie der Amplitude 1 (Leistung 1/2) das Verhältnis snr_db hat."""
    return math.sqrt(0.5 / 10.0 ** (snr_db / 10.0))


@dataclass(frozen=True)
class TestSignal:
    __test__ = False                # kein pytest-Testfall

    x: np.ndarray                   # reelle Reihe
    tracks: tuple                   # wahre Momentanfrequenz je Komponente (Arrays der Länge n)
    name: str


def linear_chirp(n: int, f0: float, f1: float, phase: float = 0.0) -> TestSignal:
    """Frequenz linear von f0 (t = 0) bis f1 (t = n - 1); Phase 2 pi (f0 t + (f1 - f0) t^2 / (2 (n - 1)))."""
    t = np.arange(n)
    ph = TWO_PI * (f0 * t + (f1 - f0) * t * t / (2.0 * (n - 1))) + phase
    return TestSignal(np.cos(ph), (f0 + (f1 - f0) * t / (n - 1),), "Chirp")


def sine_fm(n: int, fc: float, df: float, fm: float) -> TestSignal:
    """Frequenz fc + df sin(2 pi fm t): Phase 2 pi fc t - (df / fm) cos(2 pi fm t)."""
    t = np.arange(n)
    ph = TWO_PI * fc * t - (df / fm) * np.cos(TWO_PI * fm * t)
    return TestSignal(np.cos(ph), (fc + df * np.sin(TWO_PI * fm * t),), "Sinus-FM")


# --- aus koh_wavelet.py (kohaerenz-demo) ----------------------------------------

def scale_of(f, w0: float = W0):
    """Skala s = w0 / (2 pi f)."""
    return w0 / (TWO_PI * np.asarray(f, dtype=float))


def freq_grid(fmin: float = 0.02, fmax: float = 0.2, voices: int = 12) -> np.ndarray:
    """Logarithmisches Frequenzgitter, aufsteigend: fmax * 2^(-k / voices), k = 0 .. K mit fmin unterschritten oder erreicht."""
    if not 0.0 < fmin < fmax or voices < 1:
        raise ValueError("Frequenzgitter braucht 0 < fmin < fmax und voices >= 1")
    k = int(math.ceil(voices * math.log2(fmax / fmin)))
    return (fmax * 2.0 ** (-np.arange(k + 1) / voices))[::-1]


def cwt(x, freqs, w0: float = W0) -> np.ndarray:
    """Transformation per FFT (nur positive Frequenzen, Nullen bis zur Zweierpotenz >= 2n angehängt): Form (…, Frequenzen, Zeit), komplex."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    size = 1 << int(math.ceil(math.log2(2 * n)))
    spec = np.fft.fft(x, size, axis=-1)
    w = TWO_PI * np.fft.fftfreq(size)
    out = np.empty(x.shape[:-1] + (len(freqs), n), dtype=complex)
    for i, f in enumerate(freqs):
        s = float(scale_of(f, w0))
        psi = np.where(w > 0, PI14 * math.sqrt(TWO_PI * s) * np.exp(-0.5 * (s * w - w0) ** 2), 0.0)
        out[..., i, :] = np.fft.ifft(spec * psi, axis=-1)[..., :n]
    return out


# --- aus cwt_morlet.py (wavelet-skalogramm-demo) ----------------------------------------

def power(W, freqs, w0: float, rectified: bool = True) -> np.ndarray:
    """Leistung |W|^2 / s (gleichgewichtet: jede Linie der Amplitude 1 gibt dieselbe Spitze) oder |W|^2 (L2); Form wie W."""
    p = np.abs(W) ** 2
    if rectified:
        p = p / scale_of(freqs, w0)[:, None]
    return p


def ridge_frequencies(p, freqs) -> np.ndarray:
    """Kamm: je Zeitpunkt die Frequenz des höchsten Gipfels (Parabelverfeinerung im Logarithmus der Frequenz); p hat die Form (..., Frequenzen, Zeit), freqs ist logarithmisch gleichmäßig.
    Am Rand des Frequenzgitters bleibt es beim Gitterpunkt."""
    p = np.asarray(p, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    count = p.shape[-2]
    ratio = freqs[1] / freqs[0]
    idx = np.argmax(p, axis=-2)
    inner = np.clip(idx, 1, count - 2)[..., None, :]
    a, b, c = (np.log(np.maximum(np.take_along_axis(p, inner + d, axis=-2)[..., 0, :], 1e-300)) for d in (-1, 0, 1))
    den = a - 2.0 * b + c
    off = np.where((idx >= 1) & (idx <= count - 2) & (den != 0), 0.5 * (a - c) / np.where(den != 0, den, 1.0), 0.0)
    return freqs[idx] * ratio ** off
