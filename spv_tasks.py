"""Die sieben Aufgaben: je Aufgabe die Schätzer eines Laufs (für die Live-Ansicht) und die Zelle über viele Läufe (für die Tabelle).

Zelle = ein Szenario einer Aufgabe: {'values': {Verfahren: Kennzahl}, 'refs': {Bezug: Kennzahl}, 'extra': {...}}. Kennzahlen: Fehler (A in dB, C als Vielfaches der Cramér-Rao-Schranke, F als RMSE der Frequenz,
G als mittlere quadratische Abweichung) oder Erfolgsanteil in Prozent (B, D, E). Alle Zufallsströme sind fest (Seeds aus Szenario und Aufgabe).
"""

import math

import numpy as np

import signal_generator as G
import spv_constants as C
import spv_denoise as DN
import spv_estimators as E
import spv_scenarios as S
import spv_timefreq as TF

DB = 10.0 / math.log(10.0)
TWO_PI = 2.0 * math.pi

# --- A Spektrum schätzen ------------------------------------------------------------------------------------------------------------------

_TAPERS: dict = {}


def tapers(n: int, nw: float):
    """Slepian-Taper (2 NW - 1 Stück), je Länge einmal gerechnet."""
    if (n, nw) not in _TAPERS:
        _TAPERS[(n, nw)] = E.dpss(n, nw, int(round(2 * nw)) - 1)
    return _TAPERS[(n, nw)]


def burg_fit(x, p: int):
    """(a, Fehlervarianz der Ordnung p) nach Burg."""
    a, err, _ = E.burg(x, p)
    return a, float(err[-1])


def bic_order(x, n: int) -> int:
    """Ordnung nach dem BIC aus den Burg-Fehlervarianzen der Ordnungen 1 bis pmax = min(20, n/3)."""
    pmax = E.pmax_for(n, 20)
    errs = E.burg(x, pmax)[1][1:]
    orders = np.arange(1, pmax + 1)
    return E.select_order(E.criteria(n, errs, orders)["BIC"], orders)


def a_estimates(x, n: int, f, methods=C.METHODS["A"], ref_order: int = 0) -> dict:
    """Spektrum eines Laufs (Dichte, Abtastrate 1) je Verfahren auf den Frequenzen f = k / n, k = 0 .. n/2."""
    out = {}
    w = S.hann(n)
    if "Hann-Periodogramm" in methods:
        out["Hann-Periodogramm"] = np.abs(np.fft.rfft(x * w)) ** 2 / (w ** 2).sum()
    if "Welch (Hann, 50 %, n/4)" in methods:
        seg = max(16, n // 4)
        p, _ = E.welch(x, seg, E.hop_for(seg, 0.5), S.hann(seg), 1.0, seg)
        out["Welch (Hann, 50 %, n/4)"] = np.interp(f, np.arange(seg // 2 + 1) / seg, p)
    if "Multitaper (NW = 4)" in methods:
        out["Multitaper (NW = 4)"] = E.multitaper_equal(x, tapers(n, 4), 1.0, n)
    if "AR (Burg, BIC)" in methods:
        a, s2 = burg_fit(x, bic_order(x, n))
        out["AR (Burg, BIC)"] = E.ar_spectrum(a, s2, f, 1.0)
    if ref_order:
        a, s2 = burg_fit(x, ref_order)
        out["AR (Burg, feste Ordnung)"] = E.ar_spectrum(a, s2, f, 1.0)
    return out


def log_error(est, true) -> float:
    """RMS des Fehlers des Logspektrums in dB über die inneren Bins (4 an jedem Rand abgezogen)."""
    sl = slice(4, len(true) - 4)
    return float(math.sqrt(np.mean((DB * np.log(est[sl] / true[sl])) ** 2)))


def cell_a(bg: str, n: int, runs: int = C.RUNS_A) -> dict:
    x, f, true = S.background(bg, runs, n, 3100 + n)
    ref = C.REF_ORDER.get(bg, 0) if bg != "weiß" else 0
    acc = {}
    for r in range(runs):
        for name, est in a_estimates(x[r], n, f, C.METHODS["A"], ref).items():
            acc.setdefault(name, []).append(log_error(est, true))
    vals = {k: float(np.mean(v)) for k, v in acc.items()}
    return {"values": {k: vals[k] for k in C.METHODS["A"]}, "refs": {k: vals[k] for k in vals if k not in C.METHODS["A"]}, "extra": {"runs": runs}}


# --- B Zwei Linien trennen ----------------------------------------------------------------------------------------------------------------


def snapshot_length(n: int) -> int:
    """Länge der Schnappschüsse der Unterraumverfahren: n/3, mindestens 12."""
    return max(n // 3, 12)


def esprit_resolved(est, f1: float, f2: float) -> bool:
    """Beide Linien getrennt: genau zwei Schätzungen, jede innerhalb des halben Abstands der wahren Lage."""
    tol = 0.5 * (f2 - f1)
    return len(est) == 2 and abs(est[0] - f1) < tol and abs(est[1] - f2) < tol


def b_spectra(x, n: int, pad: int = C.B_PAD) -> dict:
    """Spektren und Pseudospektrum eines Laufs auf dem Gitter k / (pad n), k = 0 .. pad n / 2: Hann, Multitaper (NW = 2), AR (p = 16, BIC), MUSIC (p = 4, m = n/3)."""
    nfft = pad * n
    w = S.hann(n)
    out = {"Hann-Periodogramm": np.abs(np.fft.rfft(x * w, nfft)) ** 2}
    out["Multitaper (NW = 2)"] = E.multitaper_equal(x, tapers(n, 2), 1.0, nfft)
    fg = np.arange(nfft // 2 + 1) / nfft
    a, s2 = burg_fit(x, 16)
    out["AR (Burg, p = 16)"] = E.ar_spectrum(a, s2, fg, 1.0)
    a, s2 = burg_fit(x, bic_order(x, n))
    out["AR (Burg, BIC)"] = E.ar_spectrum(a, s2, fg, 1.0)
    m = snapshot_length(n)
    _w, v = E.eigendecomposition(E.covariance(x, m))
    out["MUSIC (p = 4)"] = E.music_spectrum(v, 4, fg)
    return out


def b_outcomes(x, n: int, f1: float, f2: float, pad: int = C.B_PAD) -> dict:
    """{Verfahren: aufgelöst?} eines Laufs: Spektren und MUSIC-Pseudospektrum aufgelöst bei Maximum an beiden Linienlagen und Einbruch auf unter die Hälfte dazwischen;
    ESPRIT aufgelöst, wenn beide Schätzungen innerhalb des halben Abstands liegen (mit MDL-Zählung nur bei vier gezählten Komponenten)."""
    out = {name: E.lines_resolved(spec, pad, f1 * n, f2 * n) for name, spec in b_spectra(x, n, pad).items()}
    m = snapshot_length(n)
    w_, v = E.eigendecomposition(E.covariance(x, m))
    esp = E.esprit(v, 4)
    out["ESPRIT (p = 4)"] = esprit_resolved(esp, f1, f2)
    out["ESPRIT, Linienzahl per MDL"] = bool(E.estimate_count(w_, n - m + 1, "MDL") == 4 and esprit_resolved(esp, f1, f2))
    return out


def cell_b(delta: float, snr: float, runs: int = C.RUNS_B) -> dict:
    n = C.N_B
    rng = G.SplitMix64(3200 + int(delta * 100) + int(snr))
    hits = {k: 0 for k in C.METHODS["B"]}
    for _ in range(runs):
        x, f2 = S.two_lines(n, C.B_F1, delta, snr, rng)
        for k, v in b_outcomes(x, n, C.B_F1, f2).items():
            hits[k] += bool(v)
    return {"values": {k: 100.0 * hits[k] / runs for k in C.METHODS["B"]}, "refs": {}, "extra": {"runs": runs}}


# --- C Eine Linie genau treffen -----------------------------------------------------------------------------------------------------------


def c_estimates(x, n: int) -> dict:
    """Frequenz der Linie in Bins je Verfahren (ein Lauf)."""
    out = {}
    X = np.fft.fft(x)[None, :n // 2 + 1]
    k = E.peak_bins(X, 4, n // 2 - 3)
    xm, x0, xp = E.three_bins(X, k)
    out["Jacobsen (3 Bins)"] = float((k + E.est_jacobsen(xm, x0, xp, n))[0])
    out["Candan (3 Bins)"] = float((k + E.est_candan(xm, x0, xp, n))[0])
    out["Spitze (Rechteck, 16-fach, Parabel)"] = E.periodogram_frequencies(x, 1, None, 16)[0] * n
    out["Hann-Spitze (16-fach, Parabel)"] = E.periodogram_frequencies(x, 1, S.hann(n), 16)[0] * n
    m = snapshot_length(n)
    _w, v = E.eigendecomposition(E.covariance(x, m))
    out["MUSIC (p = 2)"] = E.music_frequencies(v, 2, 1, np.linspace(0.05, 0.45, 801))[0] * n
    out["ESPRIT (p = 2)"] = E.esprit(v, 2)[0] * n
    a, s2 = burg_fit(x, 8)
    fg = np.arange(16 * n // 2 + 1) / (16 * n)
    out["AR-Spitze (Burg, p = 8)"] = E.peak_frequencies(E.ar_spectrum(a, s2, fg, 1.0), fg, 1)[0] * n
    return out


def cell_c(n: int, snr: float, runs: int = C.RUNS_C) -> dict:
    rng = G.SplitMix64(3300 + n + int(snr))
    err = {k: [] for k in C.METHODS["C"]}
    for _ in range(runs):
        x, fb = S.one_line(n, snr, rng)
        for k, v in c_estimates(x, n).items():
            err[k].append(v - fb)
    crb = E.crb_bins(n, snr)
    rmse = {k: float(math.sqrt(np.mean(np.square(v)))) for k, v in err.items()}
    return {"values": {k: rmse[k] / crb for k in C.METHODS["C"]}, "refs": {}, "extra": {"runs": runs, "crb": crb, "rmse": {k: rmse[k] for k in C.METHODS["C"]}}}


# --- D Eine Linie im Rauschen finden ------------------------------------------------------------------------------------------------------


def d_stats(x, n: int = C.N_D) -> tuple:
    """(Fishers g der Reihe, Fishers g der AR(2)-Residuen, größter F-Wert des Multitaper-F-Tests NW = 4 auf vierfach aufgefülltem Raster) eines Laufs."""
    xx = x - x.mean()
    p = (np.abs(np.fft.fft(xx)) ** 2)[1:(n - 1) // 2 + 1]
    s_white = float(p.max() / p.sum())
    r = E.prewhiten(x, 2)
    nr = len(r)
    pr = (np.abs(np.fft.fft(r - r.mean())) ** 2)[1:(nr - 1) // 2 + 1]
    s_pre = float(pr.max() / pr.sum())
    ff = E.f_statistic(x, tapers(n, 4), 4 * n)
    return s_white, s_pre, float(ff[4:-4].max())


def d_nominal_thresholds(n: int = C.N_D) -> tuple:
    """Schwellen aus der Theorie bei nominell 5 %: Fishers g (weißes Rauschen, m = (n - 1)/2 Frequenzen) und F-Test (7 Taper, Bonferroni über n/2 Frequenzen)."""
    return E.fisher_crit(0.05, (n - 1) // 2), E.f_threshold(7, 0.05 / (n // 2))


def cells_d(bg: str, runs: int = C.RUNS_D, null_runs: int = C.RUNS_D_NULL) -> dict:
    """{snr: Zelle} eines Hintergrunds. values: kalibrierte Detektionsrate (Schwelle aus null_runs Läufen nur Rauschen bei tatsächlich 5 %); extra: Schwellen, nominelle Falschalarmrate und
    nominelle Detektionsrate der Tests mit Theorieschwelle (Fisher-g, F-Test)."""
    n = C.N_D
    rng0 = G.SplitMix64(3400)
    noise = S.colored_noise(bg, null_runs, n, rng0)
    null = np.array([d_stats(noise[r], n) for r in range(null_runs)])
    cal = np.quantile(null, 0.95, axis=0)
    thr_g, thr_f = d_nominal_thresholds(n)
    fa = {"Fisher-g (weißer Hintergrund)": float((null[:, 0] > thr_g).mean()), "Harmonischer F-Test (NW = 4)": float((null[:, 2] > thr_f).mean())}
    cells = {}
    for snr in C.D_SNRS:
        rng = G.SplitMix64(3401 + int(snr))
        det_cal = np.zeros(3)
        det_nom = np.zeros(3)
        for _ in range(runs):
            ph = TWO_PI * rng.uniform(1)[0]
            base = S.colored_noise(bg, 1, n, rng)[0]
            x = base + S.amplitude(snr) * np.cos(TWO_PI * C.D_LINE * np.arange(n) + ph)
            s = np.array(d_stats(x, n))
            det_cal += s > cal
            det_nom += np.array([s[0] > thr_g, False, s[2] > thr_f])
        names = C.METHODS["D"]
        cells[snr] = {"values": {names[i]: 100.0 * float(det_cal[i]) / runs for i in range(3)}, "refs": {},
                      "extra": {"runs": runs, "null_runs": null_runs, "thresholds": [float(v) for v in cal], "nominal_thresholds": [thr_g, thr_f], "false_alarm_nominal": fa,
                                "nominal_rate": {names[0]: 100.0 * float(det_nom[0]) / runs, names[2]: 100.0 * float(det_nom[2]) / runs}}}
    return cells


# --- E Unregelmäßige Messzeiten -----------------------------------------------------------------------------------------------------------


def e_grid(line: float):
    """Frequenzgitter der Aufgabe E (Überabtastung 5): bis 2 bei der Linie über der mittleren Nyquist-Frequenz, sonst bis 0,5."""
    return E.grid(C.N_E, 2.0 if line > 0.5 else 0.5, 5)


def e_series(kind: str, line: float, snr: float, rng, n: int = C.N_E):
    """(Messzeiten, Messwerte) mit einer Linie der Frequenz `line` (Zeiteinheit = mittlerer Abstand) und zufälliger Phase."""
    t = E.times(kind, n, rng)
    ph = TWO_PI * rng.uniform(1)[0]
    return t, S.amplitude(snr) * np.cos(TWO_PI * line * t + ph) + rng.normal(n)


def e_interp_spectrum(t, y, n: int = C.N_E):
    """Lineares Interpolieren aufs Gitter 0, 1, ..., n - 1 und Hann-Periodogramm (achtfach aufgefüllt), die acht äußeren Punkte an jedem Rand ausgespart: (Frequenzen, Leistung)."""
    sp = np.abs(np.fft.rfft(np.interp(np.arange(n, dtype=float), t, y) * S.hann(n), 8 * n)) ** 2
    sp[:8] = 0.0
    sp[-8:] = 0.0
    return np.arange(8 * n // 2 + 1) / (8 * n), sp


def cell_e(line: float, snr: float, runs: int = C.RUNS_E) -> dict:
    n = C.N_E
    fg = e_grid(line)
    res = {}
    thr = {}
    for kind in C.E_KINDS:
        rng = G.SplitMix64(3500 + int(snr) + int(line * 10))
        nullmax = []
        for _ in range(runs):
            t = E.times(kind, n, rng)
            nullmax.append(E.lomb_scargle(t, rng.normal(n), fg).max())
        thr[kind] = float(np.quantile(nullmax, 0.95))
        hit = 0
        for _ in range(runs):
            t, y = e_series(kind, line, snr, rng)
            p = E.lomb_scargle(t, y, fg)
            i = int(np.argmax(p))
            hit += bool(p[i] > thr[kind] and abs(fg[i] - line) < 0.05)
        res[kind] = 100.0 * hit / runs
    if line < 0.5:
        rng = G.SplitMix64(3600 + int(snr))
        nm = []
        for _ in range(runs):
            t = E.times("zufällig", n, rng)
            nm.append(float(e_interp_spectrum(t, rng.normal(n))[1].max()))
        thr[C.E_INTERP] = float(np.quantile(nm, 0.95))
        hit = 0
        for _ in range(runs):
            t, y = e_series("zufällig", line, snr, rng)
            fgrid, sp = e_interp_spectrum(t, y)
            i = int(np.argmax(sp))
            hit += bool(sp[i] > thr[C.E_INTERP] and abs(fgrid[i] - line) < 0.05)
        res[C.E_INTERP] = 100.0 * hit / runs
    return {"values": res, "refs": {}, "extra": {"runs": runs, "thresholds": thr}}


# --- F Zeitabhängige Frequenz -------------------------------------------------------------------------------------------------------------


def ridge_window(n: int) -> tuple:
    """Zeitfenster der Frequenzfehler (60 bis 451 bei n = 512)."""
    lo = int(round(n * 60 / 512))
    return lo, n - lo


_FR = TF.freq_grid(0.03, 0.30, 24)


def f_estimates(x, n: int, sigma: float = C.F_SIGMA) -> dict:
    """Frequenz je Zeitpunkt aus dem Gipfel (Länge n): Spektrogramm, neu zugeordnete Frequenz am Gipfel, Synchrosqueezing, CWT-Kamm, Phasenableitung des analytischen Signals (21 geglättet)."""
    z = TF.analytic(x)
    half = TF.NFFT // 2
    X, f_hat, _t, _mask = TF.reassign_fields(z, sigma, TF.NFFT, -300.0)
    rows = np.arange(n)
    idx = np.argmax(np.abs(X[:, :half]), axis=1)
    out = {"Spektrogramm (Gauß)": idx / TF.NFFT, "Neu zugeordnet": f_hat[rows, idx]}
    sst = TF.squeezed_from(X, f_hat, np.ones(X.shape, bool))
    out["Synchrosqueezing"] = np.argmax(np.abs(sst[:, :half]), axis=1) / TF.NFFT
    W = TF.cwt(x, _FR, TF.W0)
    out["CWT-Kamm (Morlet)"] = TF.ridge_frequencies(TF.power(W, _FR, TF.W0), _FR)
    inst = np.gradient(np.unwrap(np.angle(z))) / TWO_PI
    out["Phasenableitung (21 geglättet)"] = np.convolve(inst, np.ones(21) / 21, mode="same")
    return out


def f_rmse(est, truth, n: int) -> float:
    lo, hi = ridge_window(n)
    return float(math.sqrt(np.mean((np.asarray(est)[lo:hi] - np.asarray(truth)[lo:hi]) ** 2)))


def cell_f(signal: str, snr, runs: int = C.RUNS_F) -> dict:
    n = C.N_F
    sig = S.tf_signal(signal, n)
    truth = sig.tracks[0]
    runs = 1 if snr is None else runs
    rng = G.SplitMix64(3700 + (0 if snr is None else int(snr)))
    acc = {}
    for _ in range(runs):
        x = sig.x + (0.0 if snr is None else TF.noise_sigma(snr) * rng.normal(n))
        for k, v in f_estimates(x, n).items():
            acc.setdefault(k, []).append(f_rmse(v, truth, n))
    return {"values": {k: float(np.mean(acc[k])) for k in C.METHODS["F"]}, "refs": {}, "extra": {"runs": runs}}


# --- G Entrauschen ------------------------------------------------------------------------------------------------------------------------


def g_estimates(y, n: int = C.N_G) -> dict:
    """Schätzungen des Signals aus der verrauschten Reihe (Rauschen N(0, 1), Pegel der Wavelet-Verfahren aus dem Median der Stufe 1; der Fourier-Pegel ist bekannt): Wavelet-SURE (D8, weich),
    Wavelet universal (hart) und Fourier universal (hart) und die Orakel-Bezüge (beste Fourier-Schwelle, beste Breite des gleitenden Mittelwerts) werden mit g_references berechnet."""
    h = DN.daubechies(4)
    details, _s = DN.dwt(y, h, 6)
    sigma = float(DN.mad_sigma(details[0]))
    out = {"Wavelet-SURE (D8, weich)": DN.shrink(y, h, 6, sigma, "SURE", "weich")[0], "Wavelet universal (hart)": DN.shrink(y, h, 6, sigma, "universal", "hart")[0]}
    out["Fourier universal (hart)"] = fourier_threshold(y, math.sqrt(2.0 * math.log(n)))
    return out


def fourier_threshold(y, lam: float):
    """Harte Schwelle lam auf dem Betrag der orthonormal skalierten komplexen Fourier-Koeffizienten (Rauschpegel 1); Gleichanteil und Nyquist bleiben."""
    n = len(y)
    Y = np.fft.rfft(y) / math.sqrt(n)
    Yc = Y.copy()
    Yc[1:n // 2] = np.where(math.sqrt(2) * np.abs(Y[1:n // 2]) > lam, Y[1:n // 2], 0.0)
    return np.fft.irfft(Yc * math.sqrt(n), n)


def g_references(y, truth) -> dict:
    """Orakel-Bezüge: beste Fourier-Schwelle (Gitter 0,5 bis 4 in Schritten von 0,25) und bester gleitender Mittelwert (Breiten 3, 5, 9, 17, 33, 65, periodisch)."""
    best_f = min(float(np.mean((fourier_threshold(y, lam) - truth) ** 2)) for lam in np.arange(0.5, 4.01, 0.25))
    best_m = min(float(np.mean((np.convolve(np.pad(y, 64, mode="wrap"), np.ones(w) / w, mode="same")[64:-64] - truth) ** 2)) for w in (3, 5, 9, 17, 33, 65))
    return {"Fourier beste Schwelle (Orakel)": best_f, "Gleitender Mittelwert (Orakel, beste Breite)": best_m}


def cell_g(name: str, runs: int = C.RUNS_G) -> dict:
    n = C.N_G
    f = DN.signal(name, n, C.SNR_G)
    rng = G.SplitMix64(3800 + len(name))
    acc = {}
    for _ in range(runs):
        y = f + rng.normal(n)
        acc.setdefault("verrauscht", []).append(float(np.mean((y - f) ** 2)))
        for k, v in g_estimates(y, n).items():
            acc.setdefault(k, []).append(float(np.mean((v - f) ** 2)))
        for k, v in g_references(y, f).items():
            acc.setdefault(k, []).append(v)
    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    return {"values": {k: mean[k] for k in C.METHODS["G"]}, "refs": {k: mean[k] for k in C.REFERENCES["G"]}, "extra": {"runs": runs}}
