"""Schätzer der Vorgänger-Demos, übernommen bis auf Modulbezüge und Standardwerte (Periodogramm, Welch, Slepian-Taper und Multitaper, Burg und Ordnungswahl, MUSIC, ESPRIT, Drei-Bin-Schätzer, Fisher-g, F-Test, Vorfilterung, Lomb-Scargle, Abtastmuster).

Herkunft je Abschnitt in den Kommentaren; die Funktionen sind in den Quell-Demos mit eigenen Orakeltests gesichert und dort mutationsgetestet. Reine Funktionen, nur numpy.
"""

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


# --- aus mtp_taper.py (multitaper-demo) ----------------------------------------

_CACHE: dict = {}


def _tridiagonal(n: int, nw: float):
    t = np.arange(n)
    d = ((n - 1 - 2 * t) / 2.0) ** 2 * math.cos(2.0 * math.pi * nw / n)
    e = t[1:] * (n - t[1:]) / 2.0
    return d, e


def _sturm_count(d, e2, x):
    """Zahl der Eigenwerte kleiner als x (Sturm-Folge), für ein Feld von Werten x."""
    q = d[0] - x
    count = (q < 0).astype(int)
    for j in range(1, len(d)):
        q = np.where(q == 0.0, 1e-300, q)
        q = d[j] - x - e2[j - 1] / q
        count += q < 0
    return count


def _inverse_iteration(d, e, shift, seed_vector, steps: int = 3):
    n = len(d)
    dd = d - shift
    x = seed_vector
    for _ in range(steps):
        cp = np.empty(n)
        dp = np.empty(n)
        cp[0] = e[0] / dd[0]
        dp[0] = x[0] / dd[0]
        for t in range(1, n):
            den = dd[t] - e[t - 1] * cp[t - 1]
            cp[t] = e[t] / den if t < n - 1 else 0.0
            dp[t] = (x[t] - e[t - 1] * dp[t - 1]) / den
        y = np.empty(n)
        y[-1] = dp[-1]
        for t in range(n - 2, -1, -1):
            y[t] = dp[t] - cp[t] * y[t + 1]
        x = y / np.linalg.norm(y)
    return x


def dpss(n: int, nw: float, k: int) -> np.ndarray:
    """Die k Slepian-Taper der Länge n und Zeit-Bandbreite-Produkt nw als k x n (Energie 1, symmetrische Fassung).

    Vorzeichen nach Percival und Walden: gerade Taper haben positive Summe, ungerade beginnen mit einem positiven Buckel. k <= n.
    """
    if n < 3 or k < 1 or k > n or nw <= 0 or nw >= n / 2:
        raise ValueError("Ungültige Taper-Parameter")
    key = (n, float(nw), k)
    if key in _CACHE:
        return _CACHE[key].copy()
    d, e = _tridiagonal(n, nw)
    e2 = e * e
    off = np.r_[0.0, e] + np.r_[e, 0.0]
    lo, hi = float((d - off).min()), float((d + off).max())
    a = np.full(k, lo)
    b = np.full(k, hi)
    target = n - 1 - np.arange(k)                 # der j-größte Eigenwert hat n - 1 - j Eigenwerte unter sich
    for _ in range(60):
        m = 0.5 * (a + b)
        up = _sturm_count(d, e2, m) <= target
        a = np.where(up, m, a)
        b = np.where(up, b, m)
    lam = 0.5 * (a + b)
    out = np.empty((k, n))
    rng = np.random.default_rng(12345)
    for j in range(k):
        out[j] = _inverse_iteration(d, e, lam[j] + 1e-9 * (hi - lo), rng.normal(size=n))
    thresh = max(1e-7, 1.0 / n)
    for j in range(k):
        if j % 2 == 0:
            if out[j].sum() < 0:
                out[j] = -out[j]
        else:
            first = out[j][out[j] ** 2 > thresh][0]
            if first < 0:
                out[j] = -out[j]
    if len(_CACHE) > 16:
        _CACHE.clear()
    _CACHE[key] = out
    return out.copy()


# --- aus mtp_spectrum.py (multitaper-demo) ----------------------------------------

def eigenspectra(x, h, nfft: int | None = None) -> np.ndarray:
    """Eigenspektren Y_k(f) = sum_t h_k[t] x[t] exp(-2 pi i f t): Läufe (letzte Achse = Zeit) x k x Bins."""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    return np.fft.rfft(x[..., None, :] * h, nfft or x.shape[-1], axis=-1)


def multitaper_equal(x, h, fs: float, nfft: int | None = None) -> np.ndarray:
    """Mittel der k Eigenspektren mit gleichen Gewichten: (1 / k) sum_k |Y_k|^2 / fs."""
    return (np.abs(eigenspectra(x, h, nfft)) ** 2).mean(axis=-2) / fs


def f_statistic(x, h, nfft: int | None = None) -> np.ndarray:
    """Harmonischer F-Test (Thomson): F(f) = (k - 1) |mu|^2 sum U_k(0)^2 / sum |Y_k - mu U_k(0)|^2 mit mu(f) = sum U_k(0) Y_k / sum U_k(0)^2, U_k(0) = sum h_k.

    Unter H0 (keine Linie bei f) F(2, 2k - 2)-verteilt. Muss auf einem aufgefüllten Raster ausgewertet werden, wenn die Linie zwischen Bins liegen kann.
    """
    h = np.asarray(h, dtype=float)
    k = h.shape[0]
    u0 = h.sum(axis=1)
    y = eigenspectra(x, h, nfft)
    mu = (u0[:, None] * y).sum(axis=-2) / (u0 ** 2).sum()
    res = y - mu[..., None, :] * u0[:, None]
    return (k - 1) * np.abs(mu) ** 2 * (u0 ** 2).sum() / (np.abs(res) ** 2).sum(axis=-2)


def f_threshold(k: int, tail: float) -> float:
    """Schwelle x mit P(F(2, 2k - 2) > x) = tail: x = (k - 1) (tail^(-1 / (k - 1)) - 1)."""
    if k < 2:
        raise ValueError("Der F-Test braucht mindestens zwei Taper")
    return (k - 1) * (tail ** (-1.0 / (k - 1)) - 1.0)


def lines_resolved(spec, pad: int, f1_bins: float, f2_bins: float, depth: float = 0.5) -> bool:
    """Zwei Linien gelten als aufgelöst, wenn das Spektrum um jede der beiden Linienlagen (Fenster ± halber Abstand) ein Maximum hat und das Minimum zwischen den
    Maxima unter depth mal dem kleineren Maximum liegt (Einbruch von mindestens 3 dB bei depth = 0,5). Frequenzen in Bins, das Spektrum liegt auf dem Raster 1 / pad."""
    spec = np.asarray(spec, dtype=float)
    lo_bin, hi_bin = sorted((f1_bins, f2_bins))
    c1, c2 = int(round(lo_bin * pad)), int(round(hi_bin * pad))
    half = max(1, int(round((c2 - c1) / 2.0)))
    seg1 = spec[max(c1 - half, 0):c1 + half + 1]
    seg2 = spec[max(c2 - half, 0):c2 + half + 1]
    i1 = max(c1 - half, 0) + int(np.argmax(seg1))
    i2 = max(c2 - half, 0) + int(np.argmax(seg2))
    if i1 >= i2:
        return False
    return bool(spec[i1:i2 + 1].min() < depth * min(spec[i1], spec[i2]))


# --- aus arm_estimators.py (ar-spektrum-demo) ----------------------------------------

def burg(x, p: int):
    """Burg: Reflexionskoeffizient je Stufe aus der Summe der Vorwärts- und Rückwärtsfehlerquadrate. Rückgabe (a, Fehlervarianzen 0..p, Reflexionskoeffizienten)."""
    x = np.asarray(x, dtype=float)
    ef, eb = x.copy(), x.copy()
    a = np.zeros(0)
    err = [float((x ** 2).mean())]
    refl = []
    for m in range(1, p + 1):
        efp, ebp = ef[1:], eb[:-1]
        denom = (efp ** 2).sum() + (ebp ** 2).sum()
        k = 2.0 * (efp * ebp).sum() / denom if denom > 0 else 0.0
        a = np.r_[a - k * a[::-1], k] if m > 1 else np.array([k])
        ef, eb = efp - k * ebp, ebp - k * efp
        err.append(err[-1] * (1.0 - k * k))
        refl.append(float(k))
    return a, np.array(err), np.array(refl)


def ar_spectrum(a, sigma2: float, f, fs: float = 1.0) -> np.ndarray:
    """Leistungsdichte sigma^2 / fs / |A(f)|^2 an den Frequenzen f (Hz), zweiseitig wie `signal_generator.true_psd`."""
    w = 2.0 * np.pi * np.asarray(f, dtype=float) / fs
    den = np.ones(len(w), dtype=complex)
    for k, ak in enumerate(a, start=1):
        den = den - ak * np.exp(-1j * w * k)
    return sigma2 / fs / np.abs(den) ** 2


# --- aus arm_order.py (ar-spektrum-demo) ----------------------------------------

def criteria(n: int, err, orders=None) -> dict:
    """Kriterienwerte je Ordnung; err[i] = Fehlervarianz der Ordnung orders[i] (Standard 1..len(err)). Es muss n > p + 1 gelten.

    AIC = n ln s^2 + 2 p; AICc = n ln s^2 + 2 p n / (n - p - 1); BIC = n ln s^2 + p ln n; FPE = s^2 (n + p + 1) / (n - p - 1).
    """
    e = np.asarray(err, dtype=float)
    ps = np.arange(1, len(e) + 1) if orders is None else np.asarray(orders)
    if np.any(ps >= n - 1):
        raise ValueError("Ordnung zu groß für die Länge")
    ln = n * np.log(e)
    return {"AIC": ln + 2 * ps, "AICc": ln + 2 * ps * n / (n - ps - 1), "BIC": ln + ps * math.log(n), "FPE": e * (n + ps + 1) / (n - ps - 1)}


def select_order(values, orders=None) -> int:
    """Ordnung mit dem kleinsten Kriterienwert (bei Gleichstand die kleinere)."""
    v = np.asarray(values, dtype=float)
    ps = np.arange(1, len(v) + 1) if orders is None else np.asarray(orders)
    return int(ps[int(np.argmin(v))])


def pmax_for(n: int, cap: int = 20) -> int:
    """Größte Ordnung, die ein Kriterium noch prüft: höchstens cap und n / 3."""
    return max(1, min(cap, n // 3))


# --- aus wbl_welch.py (bartlett-welch-demo) ----------------------------------------

def n_segments(n: int, seg: int, hop: int) -> int:
    """Zahl der vollständigen Segmente der Länge seg im Abstand hop in einer Folge der Länge n."""
    if seg > n or hop < 1:
        raise ValueError("Segment länger als die Folge oder Schrittweite < 1")
    return (n - seg) // hop + 1


def hop_for(seg: int, overlap: float) -> int:
    """Schrittweite in Abtastwerten für den Überlappungsanteil overlap in [0, 1)."""
    return max(1, int(round(seg * (1.0 - overlap))))


def welch(x, seg: int, hop: int, w, fs: float, nfft: int | None = None):
    """Mittel der gefensterten Segment-Periodogramme (Leistungsdichte je Hz, zweiseitig, Bins 0 .. nfft / 2) und Zahl der Segmente.

    x darf mehrere Läufe haben (letzte Achse = Zeit). Skalierung je Segment: |X|^2 / (fs sum w^2); Rechteck ohne Überlappung ist Bartlett.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if len(w) != seg:
        raise ValueError("Fensterlänge ungleich Segmentlänge")
    n = x.shape[-1]
    k = n_segments(n, seg, hop)
    segs = np.lib.stride_tricks.sliding_window_view(x, seg, axis=-1)[..., ::hop, :]
    p = np.abs(np.fft.rfft(segs * w, nfft or seg, axis=-1)) ** 2 / (fs * (w ** 2).sum())
    return p.mean(axis=-2), k


# --- aus ure_subspace.py (unterraum-demo) ----------------------------------------

def snapshots(x, m: int) -> np.ndarray:
    """Überlappende Schnappschüsse x[t .. t + m - 1] als Zeilen: (n - m + 1) x m."""
    x = np.asarray(x, dtype=float)
    if m < 2 or m > len(x):
        raise ValueError("Matrixgröße außerhalb von 2 .. n")
    return np.lib.stride_tricks.sliding_window_view(x, m)


def covariance(x, m: int, forward_backward: bool = True) -> np.ndarray:
    """Stichproben-Kovarianzmatrix (m x m) der Schnappschüsse; mit Vorwärts-Rückwärts-Mittel (R + J R J) / 2 (J spiegelt die Reihenfolge)."""
    s = snapshots(x, m)
    r = s.T @ s / len(s)
    if forward_backward:
        r = 0.5 * (r + r[::-1, ::-1])
    return r


def eigendecomposition(r):
    """Eigenwerte absteigend und die zugehörigen Eigenvektoren (Spalten) einer symmetrischen Matrix."""
    w, v = np.linalg.eigh(r)
    return w[::-1], v[:, ::-1]


def music_spectrum(v, p: int, f) -> np.ndarray:
    """MUSIC-Pseudospektrum 1 / ||E_n^T a(f)||^2 mit dem Rauschunterraum E_n (Eigenvektoren ab Nummer p) und a(f) = exp(-2 pi i f (0 .. m - 1))."""
    m = v.shape[0]
    if not 0 < p < m:
        raise ValueError("Modellordnung außerhalb von 1 .. m - 1")
    en = v[:, p:]
    a = np.exp(-2j * np.pi * np.outer(np.arange(m), np.asarray(f, dtype=float)))
    return 1.0 / np.sum(np.abs(en.T @ a) ** 2, axis=0)


def local_maxima(spec) -> list[int]:
    """Indizes innerer Punkte, die höher als der linke und mindestens so hoch wie der rechte Nachbar sind."""
    v = np.asarray(spec, dtype=float)
    return [i for i in range(1, len(v) - 1) if v[i] > v[i - 1] and v[i] >= v[i + 1]]


def peak_frequencies(spec, f, count: int) -> list[float]:
    """Frequenzen der `count` höchsten lokalen Maxima, aufsteigend sortiert."""
    idx = sorted(local_maxima(spec), key=lambda i: -spec[i])[:count]
    return sorted(float(f[i]) for i in idx)


def music_frequencies(v, p: int, count: int, grid, fine_points: int = 601) -> list[float]:
    """MUSIC-Frequenzen: grobe Suche auf dem Gitter, dann je Gipfel eine feine Suche (fine_points Punkte, ± 1,5 Gitterschritte), damit das Gitter die Genauigkeit nicht begrenzt."""
    grid = np.asarray(grid, dtype=float)
    spec = music_spectrum(v, p, grid)
    step = float(grid[1] - grid[0])
    out = []
    for i in sorted(local_maxima(spec), key=lambda i: -spec[i])[:count]:
        fine = grid[i] + np.linspace(-1.5, 1.5, fine_points) * step
        out.append(float(fine[int(np.argmax(music_spectrum(v, p, fine)))]))
    return sorted(out)


def esprit(v, p: int) -> list[float]:
    """ESPRIT (Kleinste Quadrate): Die Eigenwerte von Phi = pinv(U_1) U_2 mit den um eine Zeile verschobenen Signalvektoren U liegen bei exp(2 pi i f); jede Linie einmal."""
    us = v[:, :p]
    phi, *_ = np.linalg.lstsq(us[:-1], us[1:], rcond=None)
    f = np.sort(np.abs(np.angle(np.linalg.eigvals(phi))) / (2.0 * np.pi))
    return [float(x) for x in f[::2]]


def estimate_count(eigenvalues, n_snapshots: int, kind: str) -> int:
    """Zahl der Signalkomponenten aus den Eigenwerten (Wax und Kailath): k minimiert -N (m - k) ln(g / a) + Strafe mit dem geometrischen Mittel g und dem arithmetischen
    Mittel a der m - k kleinsten Eigenwerte. MDL: Strafe 1/2 k (2 m - k) ln N; AIC: das Doppelte der Zielfunktion plus 2 k (2 m - k)."""
    w = np.asarray(eigenvalues, dtype=float)
    m = len(w)
    best_k, best = 0, None
    for k in range(0, m - 1):
        tail = np.maximum(w[k:], 1e-300)
        ll = -n_snapshots * (m - k) * math.log(math.exp(float(np.mean(np.log(tail)))) / float(np.mean(tail)))
        crit = ll + 0.5 * k * (2 * m - k) * math.log(n_snapshots) if kind == "MDL" else 2.0 * ll + 2.0 * k * (2 * m - k)
        if best is None or crit < best:
            best_k, best = k, crit
    return best_k


def refine_parabola(spec, i: int, step: float) -> float:
    """Frequenz des Scheitels einer Parabel durch den Logarithmus der drei Werte um das Maximum i (Schrittweite step)."""
    a, b, c = np.log(spec[i - 1]), np.log(spec[i]), np.log(spec[i + 1])
    den = a - 2.0 * b + c
    return (i + (0.5 * (a - c) / den if den != 0 else 0.0)) * step


def periodogram_frequencies(x, count: int, window=None, factor: int = 16) -> list[float]:
    """Frequenzen der `count` höchsten Periodogrammspitzen (Fenster oder Rechteck, factor-fach aufgefüllt, Parabel-Nachschärfung); die ersten und letzten 8 Punkte zählen nicht."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    xw = x if window is None else x * np.asarray(window, dtype=float)
    pg = np.abs(np.fft.rfft(xw, factor * n)) ** 2
    idx = [i for i in local_maxima(pg) if 8 <= i < len(pg) - 8]
    idx = sorted(idx, key=lambda i: -pg[i])[:count]
    return sorted(refine_parabola(pg, i, 1.0 / (factor * n)) for i in idx)


def crb_bins(n: int, snr_db: float) -> float:
    """Cramér-Rao-Schranke der Standardabweichung der Frequenz einer reellen Linie in Bins: sqrt(3 / (pi^2 snr n (n^2 - 1))) * n,
    snr = A^2 / (2 sigma^2) je Abtastwert (Rife und Boorstyn 1974, reelle Linie fern von 0 und Nyquist)."""
    snr = 10.0 ** (snr_db / 10.0)
    return math.sqrt(3.0 / (math.pi ** 2 * snr * n * (n * n - 1.0))) * n


# --- aus fnl_estimators.py (fenster-linienlage-demo) ----------------------------------------

def peak_bins(X, lo: int, hi: int) -> np.ndarray:
    """Spitzen-Bins je Zeile im Bereich [lo, hi) der letzten Achse."""
    X = np.atleast_2d(X)
    return np.argmax(np.abs(X[:, lo:hi]), axis=1) + lo


def three_bins(X, k):
    X = np.atleast_2d(X)
    rows = np.arange(X.shape[0])
    k = np.atleast_1d(k)
    return X[rows, k - 1], X[rows, k], X[rows, k + 1]


def est_jacobsen(xm, x0, xp, n: int) -> np.ndarray:
    return ((xm - xp) / (2.0 * x0 - xm - xp)).real


def est_candan(xm, x0, xp, n: int) -> np.ndarray:
    """Jacobsen mit dem Faktor tan(pi/n) / (pi/n), der den Fehler endlicher Länge beseitigt."""
    return math.tan(math.pi / n) / (math.pi / n) * est_jacobsen(xm, x0, xp, n)


# --- aus pkr_tests.py (peak-oder-rauschen-demo) ----------------------------------------

def fisher_p(g: float, m: int) -> float:
    """Exakte Überschreitungswahrscheinlichkeit P(g' > g) von Fishers g bei m unabhängigen Exp-Werten (Fisher 1929): sum_j (-1)^(j-1) C(m, j) (1 - j g)^(m - 1), j = 1 .. 1/g.
    Die Terme laufen über Logarithmen der Gammafunktion und werden mit fsum summiert (Fehler unter 1e-12 bis m = 4095)."""
    if m < 1:
        raise ValueError("mindestens eine Frequenz")
    if g >= 1.0:
        return 0.0
    if g <= 1.0 / m:
        return 1.0
    terms = []
    for j in range(1, min(m, int(math.floor(1.0 / g))) + 1):
        base = 1.0 - j * g
        if base <= 0.0:
            continue
        log_term = math.lgamma(m + 1) - math.lgamma(j + 1) - math.lgamma(m - j + 1) + (m - 1) * math.log(base)
        terms.append((-1) ** (j - 1) * math.exp(log_term))
    return min(1.0, max(0.0, math.fsum(terms)))


def fisher_crit(alpha: float, m: int) -> float:
    """Kritischer Wert g_alpha mit P(g' > g_alpha) = alpha (Bisektion über fisher_p); m = 1: 1 (g ist dann immer 1)."""
    if m == 1:
        return 1.0
    lo, hi = 1.0 / m, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if fisher_p(mid, m) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --- aus pkr_prewhiten.py (peak-oder-rauschen-demo) ----------------------------------------

def residuals(x, a) -> np.ndarray:
    """Vorhersagefehler e[t] = x[t] - sum_k a_k x[t-k] für t = p .. n-1 (die ersten p Werte entfallen); x darf mehrere Zeilen haben."""
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    p = len(a)
    n = x.shape[-1]
    out = x[..., p:].copy()
    for k in range(1, p + 1):
        out = out - a[k - 1] * x[..., p - k:n - k]
    return out


def prewhiten(x, order: int = 2) -> np.ndarray:
    """Mittelwert abziehen, AR-Modell der Ordnung `order` per Burg anpassen und die Residuen liefern (je Zeile)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean(axis=-1, keepdims=True)
    if x.ndim == 1:
        return residuals(x, burg(x, order)[0])
    return np.stack([residuals(row, burg(row, order)[0]) for row in x])


# --- aus lsp_lomb.py (lomb-scargle-demo) ----------------------------------------

@dataclass(frozen=True)
class Sums:
    n: int
    ysum: float                     # sum y
    yy: float                       # sum y^2
    yc: np.ndarray
    ys: np.ndarray
    c1: np.ndarray
    s1: np.ndarray
    c2: np.ndarray
    s2: np.ndarray


CHUNK = 2_000_000        # höchstens so viele Auswertungen (Frequenzen mal Zeitpunkte) je Block, begrenzt den Speicher


def sums_naive(t, y, f) -> Sums:
    """Die sechs Summen je Frequenz durch direkte Summation (n mal Zahl der Frequenzen Auswertungen), blockweise über die Frequenzen."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    f = np.asarray(f, dtype=float)
    step = max(1, CHUNK // len(t))
    parts = []
    for lo in range(0, len(f), step):
        w = TWO_PI * f[lo:lo + step, None] * t[None, :]
        c, s = np.cos(w), np.sin(w)
        c2, s2 = c * c - s * s, 2.0 * c * s
        parts.append((c @ y, s @ y, c.sum(axis=1), s.sum(axis=1), c2.sum(axis=1), s2.sum(axis=1)))
    cols = [np.concatenate([p[k] for p in parts]) if parts else np.zeros(0) for k in range(6)]
    return Sums(len(t), float(y.sum()), float(y @ y), *cols)


def power_classical(s: Sums, normalize: bool = True) -> np.ndarray:
    """Klassischer Lomb-Scargle (Stichprobenmittel abgezogen): halbe Verringerung der Quadratsumme beim Fit von a cos + b sin an y - Mittelwert.

    normalize: Scargle-Normierung, geteilt durch die Stichprobenvarianz (n - 1); unter weißem Rauschen etwa Exp(1). Bei Frequenzen, bei denen
    Kosinus und Sinus linear abhängig sind (Vielfache der halben Abtastrate bei regelmäßigen Zeiten), gilt der Fit mit dem verbleibenden Term."""
    n = s.n
    ybar = s.ysum / n
    yc, ys = s.yc - ybar * s.c1, s.ys - ybar * s.s1
    cc, ss, cs = 0.5 * (n + s.c2), 0.5 * (n - s.c2), 0.5 * s.s2
    p = 0.5 * _reduction(cc, ss, cs, yc, ys, n)
    if not normalize:
        return p
    return p / ((s.yy - n * ybar * ybar) / (n - 1))


def _reduction(cc, ss, cs, yc, ys, n):
    """Verringerung der Quadratsumme beim Fit von a cos + b sin: (ss yc^2 + cc ys^2 - 2 cs yc ys) / (cc ss - cs^2); entartet: nur der tragende Term."""
    d = cc * ss - cs * cs
    ok = d > 1e-9 * n * n
    with np.errstate(divide="ignore", invalid="ignore"):
        general = (ss * yc ** 2 + cc * ys ** 2 - 2.0 * cs * yc * ys) / d
        only_c = np.where(cc > 1e-9 * n, yc ** 2 / cc, 0.0)
        only_s = np.where(ss > 1e-9 * n, ys ** 2 / ss, 0.0)
    return np.where(ok, general, np.where(cc >= ss, only_c, only_s))


def power_floating(s: Sums) -> np.ndarray:
    """Verallgemeinerter Lomb-Scargle (Zechmeister und Kürster 2009): Fit von a cos + b sin + c, normiert auf 0 bis 1 (Anteil der Quadratsumme um das Mittel)."""
    n = s.n
    ybar = s.ysum / n
    cm, sm = s.c1 / n, s.s1 / n
    cc, ss, cs = 0.5 * (1.0 + s.c2 / n) - cm * cm, 0.5 * (1.0 - s.c2 / n) - sm * sm, 0.5 * s.s2 / n - cm * sm
    yc, ys = s.yc / n - ybar * cm, s.ys / n - ybar * sm
    yy = s.yy / n - ybar * ybar
    d = cc * ss - cs * cs
    ok = d > 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        p = (ss * yc ** 2 + cc * ys ** 2 - 2.0 * cs * yc * ys) / (yy * d)
    return np.where(ok, p, 0.0)


def lomb_scargle(t, y, f, mean: str = "abgezogen", normalize: bool = True) -> np.ndarray:
    """Leistung bei den Frequenzen f: mean 'abgezogen' (klassisch) oder 'mitgefittet' (verallgemeinert, normiert auf 0 bis 1; `normalize` gilt nur für den klassischen)."""
    s = sums_naive(t, y, f)
    if mean == "mitgefittet":
        return power_floating(s)
    if mean == "abgezogen":
        return power_classical(s, normalize)
    raise ValueError(f"unbekannte Mittelwert-Behandlung: {mean}")


def grid(n: int, fmax: float, ofac: int = 10) -> np.ndarray:
    """Frequenzgitter (0, fmax] mit Schritt 1 / (ofac n); die Spanne ist n."""
    return np.arange(1, int(round(fmax * n * ofac)) + 1) / (ofac * n)


# --- aus lsp_times.py (lomb-scargle-demo) ----------------------------------------

KINDS = ("regelmäßig", "Jitter ±0,25", "Jitter ±0,5", "zufällig", "Nachtlücken", "Enden")


JITTER = {"Jitter ±0,25": 0.5, "Jitter ±0,5": 1.0}      # Breite des gleichverteilten Versatzes in Abständen


PERIOD = 10.0                                            # Nachtlücken: Periode und Breite des Beobachtungsfensters


WIDTH = 3.0


END_FRACTION = 0.1                                       # Enden: je die Hälfte der Punkte im ersten und im letzten Zehntel der Spanne


def times(kind: str, n: int, rng) -> np.ndarray:
    """Aufsteigend sortierte Messzeitpunkte des Musters `kind` (n Stück)."""
    if kind == "regelmäßig":
        return np.arange(n, dtype=float)
    if kind in JITTER:
        return np.arange(n) + JITTER[kind] * (rng.uniform(n) - 0.5)
    if kind == "zufällig":
        return np.sort(rng.uniform(n) * n)
    if kind == "Nachtlücken":
        k = int(math.ceil(n / PERIOD))
        u = rng.uniform(n) * (k * WIDTH)
        return np.sort(np.floor(u / WIDTH) * PERIOD + (u % WIDTH))
    if kind == "Enden":
        u = rng.uniform(n)
        return np.sort(np.where(np.arange(n) < n // 2, END_FRACTION * n * u, (1.0 - END_FRACTION) * n + END_FRACTION * n * u))
    raise ValueError(f"unbekanntes Abtastmuster: {kind}")
