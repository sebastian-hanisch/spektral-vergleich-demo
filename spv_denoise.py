"""Entrauschen der Vorgänger-Demos, wortgleich übernommen (Daubechies-DWT, Schwellen, SURE, universale Schwelle, Testsignale von Donoho und Johnstone). Reine Funktionen, nur numpy.
"""

import math

import numpy as np

METHODS = ("universal", "SURE", "Hybrid", "fest")
RULES = ("weich", "hart")


# --- aus shr_wavelet.py (wavelet-schrumpfung-demo) ----------------------------------------

def daubechies(p: int) -> np.ndarray:
    """Skalierungsfilter h der Länge 2p (Summe sqrt 2, Summe der Quadrate 1, Extremalphase) der Daubechies-Wavelet mit p verschwindenden Momenten (p = 1: Haar), per Spektralfaktorisierung."""
    if p < 1:
        raise ValueError("p muss mindestens 1 sein")
    if p == 1:
        return np.array([1.0, 1.0]) / math.sqrt(2.0)
    coef = [math.comb(p - 1 + k, k) for k in range(p)]
    zs = []
    for y in np.roots(coef[::-1]):
        r = np.roots([1.0, -(2.0 - 4.0 * y), 1.0])
        zs.append(r[np.argmin(np.abs(r))])
    poly = np.array([1.0 + 0j])
    for _ in range(p):
        poly = np.convolve(poly, [1.0, 1.0])
    for z in zs:
        poly = np.convolve(poly, [1.0, -z])
    h = np.real(poly)
    h = h / h.sum() * math.sqrt(2.0)
    return h[::-1] if abs(h[0]) < abs(h[-1]) else h


def qmf(h) -> np.ndarray:
    """Wavelet-Filter g[l] = (-1)^l h[L - 1 - l] (Quadraturspiegelfilter)."""
    h = np.asarray(h, dtype=float)
    length = len(h)
    return np.array([(-1) ** l * h[length - 1 - l] for l in range(length)])


def check_levels(n: int, levels: int) -> None:
    if levels < 1:
        raise ValueError("Stufenzahl muss mindestens 1 sein")
    if n % (2 ** levels) != 0:
        raise ValueError("Die Länge muss durch 2^J teilbar sein")


def dwt(x, h, levels: int):
    """Periodische DWT per Pyramide: ([d_1, ..., d_J], a_J); c[k] = sum_l h[l] a[(2k + l) mod N], d[k] = sum_l g[l] a[(2k + l) mod N]; x hat die Form (..., n)."""
    a = np.asarray(x, dtype=float)
    check_levels(a.shape[-1], levels)
    h = np.asarray(h, dtype=float)
    g = qmf(h)
    length = len(h)
    details = []
    for _ in range(levels):
        size = a.shape[-1]
        idx = (2 * np.arange(size // 2)[:, None] + np.arange(length)[None, :]) % size
        window = a[..., idx]
        details.append(window @ g)
        a = window @ h
    return details, a


def idwt(details, a, h) -> np.ndarray:
    """Inverse der periodischen DWT (perfekte Rekonstruktion), Form (..., n)."""
    h = np.asarray(h, dtype=float)
    g = qmf(h)
    length = len(h)
    a = np.asarray(a, dtype=float)
    for d in reversed(details):
        d = np.asarray(d, dtype=float)
        m = d.shape[-1]
        out = np.zeros(d.shape[:-1] + (2 * m,))
        for l in range(length):
            idx = (2 * np.arange(m) + l) % (2 * m)
            out[..., idx] += a * h[l] + d * g[l]
        a = out
    return a


# --- aus shr_shrink.py (wavelet-schrumpfung-demo) ----------------------------------------

def soft(d, lam):
    """Weiche Schwelle sign(d) (|d| - lam)_+."""
    d = np.asarray(d, dtype=float)
    return np.sign(d) * np.maximum(np.abs(d) - lam, 0.0)


def hard(d, lam):
    """Harte Schwelle d 1{|d| > lam}."""
    d = np.asarray(d, dtype=float)
    return np.where(np.abs(d) > lam, d, 0.0)


def apply_rule(d, lam, rule: str):
    if rule == "weich":
        return soft(d, lam)
    if rule == "hart":
        return hard(d, lam)
    raise ValueError(f"unbekannte Schwelle: {rule}")


def universal_threshold(sigma, n: int):
    """Universale Schwelle sigma sqrt(2 ln n)."""
    return np.asarray(sigma, dtype=float) * math.sqrt(2.0 * math.log(n))


def mad_sigma(d1):
    """Rauschpegel aus dem Median der Beträge der Detailkoeffizienten der Stufe 1: median(|d|) / 0,6745 (je Reihe)."""
    return np.median(np.abs(np.asarray(d1, dtype=float)), axis=-1) / 0.6745


def sure_risk(z, lam) -> float:
    """Steins unverzerrter Risikoschätzer der weichen Schwelle für Koeffizienten mit Rauschvarianz 1: m - 2 #{|z| <= lam} + sum min(|z|, lam)^2 (eine Reihe)."""
    z = np.abs(np.asarray(z, dtype=float))
    return float(len(z) - 2 * np.sum(z <= lam) + np.sum(np.minimum(z, lam) ** 2))


def sure_threshold(z, lam_max):
    """Schwelle aus {0} ∪ {|z_k|} bis lam_max, die SURE minimiert (Rauschvarianz 1), je Reihe; z hat die Form (..., m)."""
    a = np.sort(np.abs(np.asarray(z, dtype=float)), axis=-1)
    m = a.shape[-1]
    k = np.arange(1, m + 1)
    risk = m - 2 * k + np.cumsum(a ** 2, axis=-1) + (m - k) * a ** 2
    risk = np.where(a <= np.asarray(lam_max)[..., None] if np.ndim(lam_max) else a <= lam_max, risk, np.inf)
    best = np.argmin(risk, axis=-1)
    lam = np.take_along_axis(a, best[..., None], axis=-1)[..., 0]
    return np.where(np.min(risk, axis=-1) < m, lam, 0.0)       # SURE(0) = m: nur eine echte Verbesserung hebt die Schwelle über null


def level_threshold(d, sigma, method: str, n: int, fixed: float = 0.0):
    """Schwelle für die Detailkoeffizienten einer Stufe (Form (..., m), sigma je Reihe): universal, SURE (je Stufe), Hybrid oder fest (fixed mal sigma... nein: fixed ist die Schwelle selbst)."""
    d = np.asarray(d, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    m = d.shape[-1]
    if method == "fest":
        return np.broadcast_to(np.asarray(fixed, dtype=float), sigma.shape).copy()
    universal = universal_threshold(sigma, n)
    if method == "universal":
        return universal
    z = d / sigma[..., None]
    sure = sigma * sure_threshold(z, math.sqrt(2.0 * math.log(m)))
    if method == "SURE":
        return sure
    if method == "Hybrid":
        eta = (np.sum(z ** 2, axis=-1) - m) / m
        return np.where(eta <= math.log2(m) ** 1.5 / math.sqrt(m), universal, sure)
    raise ValueError(f"unbekanntes Verfahren: {method}")


def shrink_details(details, sigma, method: str, rule: str, n: int, fixed: float = 0.0):
    """(geschwellte Details, Schwellen je Stufe); die Glättung bleibt unberührt."""
    out, thresholds = [], []
    for d in details:
        lam = level_threshold(d, sigma, method, n, fixed)
        thresholds.append(lam)
        out.append(apply_rule(d, np.asarray(lam)[..., None], rule))
    return out, thresholds


def shrink(y, h, levels: int, sigma, method: str, rule: str, fixed: float = 0.0):
    """Entrauschen: DWT, Schwelle auf den Details, Rücktransformation; ((..., n)-Schätzung, Schwellen je Stufe)."""
    y = np.asarray(y, dtype=float)
    details, smooth = dwt(y, h, levels)
    new, thresholds = shrink_details(details, sigma, method, rule, y.shape[-1], fixed)
    return idwt(new, smooth, h), thresholds


# --- aus shr_signals.py (wavelet-schrumpfung-demo) ----------------------------------------

BLOCKS_H = np.array([4, -5, 3, -4, 5, -4.2, 2.1, 4.3, -3.1, 5.1, -4.2])


BUMPS_H = np.array([4, 5, 3, 4, 5, 4.2, 2.1, 4.3, 3.1, 5.1, 4.2])


BUMPS_W = np.array([0.005, 0.005, 0.006, 0.01, 0.01, 0.03, 0.01, 0.01, 0.005, 0.008, 0.005])


KNOTS = np.array([0.1, 0.13, 0.15, 0.23, 0.25, 0.4, 0.44, 0.65, 0.76, 0.78, 0.81])


def _grid(n: int) -> np.ndarray:
    return (np.arange(n) + 0.5) / n


def blocks_raw(n: int) -> np.ndarray:
    """Summe von Stufen h_j (1 + sgn(t - t_j))/2."""
    t = _grid(n)
    return sum(h * (1.0 + np.sign(t - tj)) / 2.0 for h, tj in zip(BLOCKS_H, KNOTS))


def bumps_raw(n: int) -> np.ndarray:
    """Summe von Spitzen h_j (1 + |(t - t_j)/w_j|)^-4."""
    t = _grid(n)
    return sum(h * (1.0 + np.abs((t - tj) / w)) ** -4 for h, tj, w in zip(BUMPS_H, KNOTS, BUMPS_W))


def heavisine_raw(n: int) -> np.ndarray:
    """4 sin(4 pi t) - sgn(t - 0,3) - sgn(0,72 - t): glatt mit zwei Sprüngen."""
    t = _grid(n)
    return 4.0 * np.sin(4.0 * np.pi * t) - np.sign(t - 0.3) - np.sign(0.72 - t)


def doppler_raw(n: int) -> np.ndarray:
    """sqrt(t (1 - t)) sin(2 pi 1,05 / (t + 0,05)): schnell oszillierend am Anfang."""
    t = _grid(n)
    return np.sqrt(t * (1.0 - t)) * np.sin(2.0 * np.pi * 1.05 / (t + 0.05))


RAW = {"Blocks": blocks_raw, "Bumps": bumps_raw, "HeaviSine": heavisine_raw, "Doppler": doppler_raw}


def shape(name: str, n: int) -> np.ndarray:
    """Signalform auf die Standardabweichung 1 skaliert (der Mittelwert bleibt)."""
    if name not in RAW:
        raise ValueError(f"unbekanntes Signal: {name}")
    f = RAW[name](n)
    return f / f.std()


def signal(name: str, n: int, snr: float) -> np.ndarray:
    """Wahres Signal der Standardabweichung snr (Rauschen N(0, 1))."""
    return snr * shape(name, n)
