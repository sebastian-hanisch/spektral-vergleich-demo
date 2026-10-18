"""Analyse eines Szenarios (eine Realisierung, live), Experimente aus der vorgerechneten Tabelle und Verdikte. Reine Funktionen; Zufall nur über SplitMix64."""

import math
from dataclasses import dataclass, field

import numpy as np

import signal_generator as G
import spv_constants as C
import spv_denoise as DN
import spv_estimators as E
import spv_regret as R
import spv_results as RES
import spv_scenarios as S
import spv_tasks as T
import spv_timefreq as TF


def de(x: float, digits: int = 1) -> str:
    """Zahl mit Dezimalkomma und Minuszeichen; „–“ für nicht definierte Werte, „∞“ für unendliche."""
    if x != x:
        return "–"
    if x in (math.inf, -math.inf):
        return "∞" if x > 0 else "−∞"
    return f"{x:.{digits}f}".replace(".", ",").replace("-", "−")


LIVE_DIGITS = {"A": 2, "B": 0, "C": 3, "D": 0, "E": 0, "F": 5, "G": 3}  # Nachkommastellen der Kennzahlen einer Realisierung (C in Bins, sonst wie die Tabellen)
DIGITS = {"A": 2, "B": 1, "C": 2, "D": 0, "E": 0, "F": 5, "G": 3}       # Nachkommastellen der Kennzahlen je Aufgabe in Tabellen und Texten


@dataclass(frozen=True)
class Settings:
    task: str = C.DEFAULT_TASK
    a_bg: str = C.DEFAULTS["a_bg"]
    a_n: int = C.DEFAULTS["a_n"]
    b_delta: float = C.DEFAULTS["b_delta"]
    b_snr: float = C.DEFAULTS["b_snr"]
    c_n: int = C.DEFAULTS["c_n"]
    c_snr: float = C.DEFAULTS["c_snr"]
    d_bg: str = C.DEFAULTS["d_bg"]
    d_snr: float = C.DEFAULTS["d_snr"]
    e_line: float = C.DEFAULTS["e_line"]
    e_snr: float = C.DEFAULTS["e_snr"]
    f_sig: str = C.DEFAULTS["f_sig"]
    f_snr: float = C.DEFAULTS["f_snr"]
    g_sig: str = C.DEFAULTS["g_sig"]
    seed: int = C.DEFAULT_SEED

    @property
    def params(self) -> dict:
        """Die Parameter der gewählten Aufgabe im Schlüsselformat der Tabelle (für Zellen und Entscheidungshilfe)."""
        t = self.task
        if t == "A":
            return {"bg": self.a_bg, "n": self.a_n}
        if t == "B":
            return {"delta": self.b_delta, "snr": self.b_snr}
        if t == "C":
            return {"n": self.c_n, "snr": self.c_snr}
        if t == "D":
            return {"bg": self.d_bg, "snr": self.d_snr}
        if t == "E":
            return {"line": self.e_line, "snr": self.e_snr}
        if t == "F":
            return {"signal": self.f_sig, "snr": self.f_snr}
        if t == "G":
            return {"signal": self.g_sig}
        raise ValueError(f"unbekannte Aufgabe: {t}")


@dataclass(frozen=True)
class Analysis:
    settings: Settings
    title: str
    unit: str
    values: dict                                    # Kennzahl je Verfahren in dieser Realisierung
    curves: dict                                    # Verfahren -> (x, y) der Darstellung
    truth_curve: tuple | None = None                # (x, y) der Wahrheit
    truth_lines: tuple = ()                         # senkrechte wahre Lagen
    points: dict = field(default_factory=dict)      # Verfahren -> Schätzwert (Linien, Frequenzen)
    extra: dict = field(default_factory=dict)
    refs: dict = field(default_factory=dict)        # Orakel-Bezüge dieser Realisierung


def _db(v):
    return 10.0 * np.log10(np.maximum(np.asarray(v, float), 1e-300))


def analyse(s: Settings) -> Analysis:
    return {"A": _analyse_a, "B": _analyse_b, "C": _analyse_c, "D": _analyse_d, "E": _analyse_e, "F": _analyse_f, "G": _analyse_g}[s.task](s)


def _analyse_a(s: Settings) -> Analysis:
    x, f, true = S.background(s.a_bg, 1, s.a_n, s.seed)
    ref = C.REF_ORDER.get(s.a_bg, 0) if s.a_bg != "weiß" else 0
    est = T.a_estimates(x[0], s.a_n, f, C.METHODS["A"], ref)
    vals = {k: T.log_error(v, true) for k, v in est.items()}
    curves = {k: (f, _db(est[k])) for k in C.METHODS["A"]}
    return Analysis(s, f"{s.a_bg}, n = {s.a_n}", "dB", {k: vals[k] for k in C.METHODS["A"]}, curves, (f, _db(true)), refs={k: v for k, v in vals.items() if k not in C.METHODS["A"]})


def _analyse_b(s: Settings) -> Analysis:
    n = C.N_B
    rng = G.SplitMix64(s.seed)
    x, f2 = S.two_lines(n, C.B_F1, s.b_delta, s.b_snr, rng)
    spectra = T.b_spectra(x, n)
    out = T.b_outcomes(x, n, C.B_F1, f2)
    nfft = C.B_PAD * n
    bins = np.arange(nfft // 2 + 1) / C.B_PAD
    lo, hi = C.B_F1 * n - 4.0, f2 * n + 4.0
    sel = (bins >= lo) & (bins <= hi)
    curves = {}
    for k, v in spectra.items():
        d = _db(v)
        curves[k] = (bins[sel], d[sel] - d[sel].max())
    m = T.snapshot_length(n)
    _w, v = E.eigendecomposition(E.covariance(x, m))
    esp = {"ESPRIT (p = 4)": [e * n for e in E.esprit(v, 4)]}
    return Analysis(s, f"Abstand {de(s.b_delta, 2)} Bins, {de(s.b_snr, 0)} dB je Linie, n = {n}", "aufgelöst (1) oder nicht (0)", {k: float(bool(val)) for k, val in out.items()}, curves,
                    truth_lines=(C.B_F1 * n, f2 * n), points=esp, extra={"f1_bins": C.B_F1 * n, "f2_bins": f2 * n})


def _analyse_c(s: Settings) -> Analysis:
    n = s.c_n
    rng = G.SplitMix64(s.seed)
    x, fb = S.one_line(n, s.c_snr, rng)
    est = T.c_estimates(x, n)
    pad = 16
    w = S.hann(n)
    sp = np.abs(np.fft.rfft(x * w, pad * n)) ** 2
    bins = np.arange(len(sp)) / pad
    sel = (bins >= fb - 6) & (bins <= fb + 6)
    d = _db(sp)
    crb = E.crb_bins(n, s.c_snr)
    return Analysis(s, f"n = {n}, {de(s.c_snr, 0)} dB", "Fehler in Bins", {k: float(abs(est[k] - fb)) for k in C.METHODS["C"]}, {"Hann-Periodogramm": (bins[sel], d[sel] - d[sel].max())}, truth_lines=(fb,),
                    points={k: float(est[k]) for k in C.METHODS["C"]}, extra={"fb": fb, "crb": crb})


def _analyse_d(s: Settings) -> Analysis:
    n = C.N_D
    rng = G.SplitMix64(s.seed)
    ph = 2 * math.pi * rng.uniform(1)[0]
    base = S.colored_noise(s.d_bg, 1, n, rng)[0]
    x = base + S.amplitude(s.d_snr) * np.cos(2 * math.pi * C.D_LINE * np.arange(n) + ph)
    stats = T.d_stats(x, n)
    cell = RES.find("D", bg=s.d_bg, snr=C.D_SNRS[-1])
    cal = cell["extra"]["thresholds"]
    thr_g, thr_f = cell["extra"]["nominal_thresholds"]
    names = C.METHODS["D"]
    vals = {names[i]: float(stats[i] > cal[i]) for i in range(3)}
    p = (np.abs(np.fft.fft(x - x.mean())) ** 2)[1:(n - 1) // 2 + 1]
    fr = np.arange(1, (n - 1) // 2 + 1) / n
    return Analysis(s, f"{s.d_bg}, Linie {de(s.d_snr, 0)} dB, n = {n}", "erkannt (1) oder nicht (0), kalibrierte Schwelle", vals, {"Periodogramm": (fr, _db(p))}, truth_lines=(C.D_LINE,),
                    extra={"stats": stats, "calibrated": cal, "nominal": (thr_g, thr_f), "nominal_decision": (float(stats[0] > thr_g), None, float(stats[2] > thr_f)),
                           "false_alarm_nominal": cell["extra"]["false_alarm_nominal"]})


def _analyse_e(s: Settings) -> Analysis:
    n = C.N_E
    fg = T.e_grid(s.e_line)
    cell = RES.find("E", line=s.e_line, snr=C.E_SNRS[0])
    thr = cell["extra"]["thresholds"]
    rng = G.SplitMix64(s.seed)
    curves, vals = {}, {}
    for kind in C.E_KINDS:
        t, y = T.e_series(kind, s.e_line, s.e_snr, rng)
        p = E.lomb_scargle(t, y, fg)
        i = int(np.argmax(p))
        vals[kind] = float(p[i] > thr[kind] and abs(fg[i] - s.e_line) < 0.05)
        curves[kind] = (fg, p)
    if s.e_line < 0.5:
        t, y = T.e_series("zufällig", s.e_line, s.e_snr, rng)
        fgrid, sp = T.e_interp_spectrum(t, y)
        i = int(np.argmax(sp))
        vals[C.E_INTERP] = float(sp[i] > thr[C.E_INTERP] and abs(fgrid[i] - s.e_line) < 0.05)
        curves[C.E_INTERP] = (fgrid, sp)
    return Analysis(s, f"Linie bei {de(s.e_line, 1)} (mittlere Nyquist-Frequenz 0,5), {de(s.e_snr, 0)} dB, n = {n}", "erkannt (1) oder nicht (0)", vals, curves, truth_lines=(s.e_line,), extra={"thresholds": thr})


def _analyse_f(s: Settings) -> Analysis:
    n = C.N_F
    sig = S.tf_signal(s.f_sig, n)
    rng = G.SplitMix64(s.seed)
    x = sig.x + TF.noise_sigma(s.f_snr) * rng.normal(n)
    est = T.f_estimates(x, n)
    lo, hi = T.ridge_window(n)
    t = np.arange(n)
    return Analysis(s, f"{s.f_sig}, {de(s.f_snr, 0)} dB, n = {n}", "RMSE der Frequenz", {k: T.f_rmse(v, sig.tracks[0], n) for k, v in est.items()}, {k: (t, v) for k, v in est.items()},
                    truth_curve=(t, sig.tracks[0]), extra={"window": (lo, hi)})


def _analyse_g(s: Settings) -> Analysis:
    n = C.N_G
    f = DN.signal(s.g_sig, n, C.SNR_G)
    rng = G.SplitMix64(s.seed)
    y = f + rng.normal(n)
    est = T.g_estimates(y, n)
    refs = T.g_references(y, f)
    refs["verrauscht"] = float(np.mean((y - f) ** 2))
    t = np.arange(n)
    curves = {k: (t, v) for k, v in est.items()}
    curves["verrauscht"] = (t, y)
    return Analysis(s, f"{s.g_sig}, Signal-Rausch-Verhältnis {de(C.SNR_G, 0)}, n = {n}", "mittlere quadratische Abweichung", {k: float(np.mean((v - f) ** 2)) for k, v in est.items()}, curves, (t, f), refs=refs)


def extremes(a: Analysis) -> tuple:
    """(beste Verfahren, schlechteste Verfahren, bester Wert, schlechtester Wert) einer Realisierung; bei Gleichstand alle Verfahren mit dem Wert, in der Reihenfolge der Verfahren."""
    vals = a.values
    kind_error = C.TASK_KIND[a.settings.task] == C.KIND_ERROR
    hi, lo = max(vals.values()), min(vals.values())
    best_value, worst_value = (lo, hi) if kind_error else (hi, lo)
    return [k for k, v in vals.items() if v == best_value], [k for k, v in vals.items() if v == worst_value], best_value, worst_value


def names_label(names: list) -> str:
    """Ein Name, bei Gleichstand „n gleichauf“ (die Namen stehen im Verdikt darunter)."""
    return names[0] if len(names) == 1 else f"{len(names)} gleichauf"


# --- Verdikt der Live-Ansicht ------------------------------------------------------------------------------------------------------------------


def verdict_main(a: Analysis) -> tuple[str, str]:
    t = a.settings.task
    kind_err = C.TASK_KIND[t] == C.KIND_ERROR
    vals = a.values
    best = min(vals, key=vals.get) if kind_err else None
    head = f"Szenario {a.title} (eine Realisierung, Seed {a.settings.seed}): "
    tail = " Die Tabellen unten zeigen viele Läufe; eine einzelne Realisierung kann vom Mittel abweichen."
    if t == "A":
        ar_type = a.settings.a_bg in C.AR_TYPE
        home = "Der Hintergrund ist autoregressiv: Das AR-Modell passt (Heimvorteil)." if ar_type else "Der Hintergrund ist kein autoregressives Spektrum."
        worst = max(vals, key=vals.get)
        return ("success" if best == "AR (Burg, BIC)" else "info"), (head + f"kleinster Fehler {best} mit {de(vals[best], 2)} dB, größter {worst} mit {de(vals[worst], 2)} dB. " + home + tail)
    if t == "C":
        crb = a.extra["crb"]
        return "info", (head + f"kleinster Fehler {best} mit {de(vals[best], 3)} Bins (Cramér-Rao-Schranke {de(crb, 3)} Bins), größter {max(vals, key=vals.get)} mit {de(max(vals.values()), 3)} Bins." + tail)
    if t == "F":
        return "info", (head + f"kleinster Fehler {best} mit {de(vals[best], 5)}, größter {max(vals, key=vals.get)} mit {de(max(vals.values()), 5)}." + tail)
    if t == "G":
        return "info", (head + f"kleinster Fehler {best} mit {de(vals[best], 3)} (verrauscht {de(a.refs['verrauscht'], 3)})." + tail)
    ok = [k for k, v in vals.items() if v >= 0.5]
    word = {"B": "aufgelöst", "D": "erkannt", "E": "erkannt"}[t]
    if not ok:
        return "warning", head + f"kein Verfahren hat die Linie(n) {word}." + tail
    if len(ok) == len(vals):
        return "success", head + f"alle Verfahren haben die Linie(n) {word}." + tail
    return "info", head + f"{word}: {', '.join(ok)}; nicht {word}: {', '.join(k for k in vals if k not in ok)}." + tail


# --- Experimente aus der Tabelle ----------------------------------------------------------------------------------------------------------


def exp_spectrum(ar_share_percent: int = 71) -> dict:
    """Aufgabe A: alle Zellen, Zusammenfassung unter dem Szenarienmix, Siege nach Hintergrundtyp."""
    cells = RES.cells("A")
    share = ar_share_percent / 100.0
    summary = R.summarize("A", cells, R.mix_weights(cells, share))
    wins = {k: {"ar": 0, "non_ar": 0} for k in C.METHODS["A"]}
    n_ar = n_non = 0
    for c in cells:
        ar = c["params"]["bg"] in C.AR_TYPE
        n_ar += ar
        n_non += not ar
        for k in R.winners("A", c):
            wins[k]["ar" if ar else "non_ar"] += 1
    return {"cells": cells, "summary": summary, "share": share, "minimax": R.minimax_choice(summary), "wins_by_type": wins, "n_ar": n_ar, "n_non_ar": n_non, "equal": R.summarize("A", cells)}


def mix_curves(step: int = 5) -> dict:
    """Aufgabe A: mittlerer Reuegrad und Siegesanteil je Verfahren über dem Anteil autoregressiver Hintergründe 0, step, ... 100 Prozent."""
    cells = RES.cells("A")
    shares = list(range(0, 101, step))
    out = {"shares": shares, "mean": {k: [] for k in C.METHODS["A"]}, "wins": {k: [] for k in C.METHODS["A"]}}
    for p in shares:
        summ = R.summarize("A", cells, R.mix_weights(cells, p / 100.0))
        for k in C.METHODS["A"]:
            out["mean"][k].append(summ[k]["mean"])
            out["wins"][k].append(summ[k]["wins_weighted"])
    return out


def wins_crossover(first: str = "AR (Burg, BIC)", second: str = "Welch (Hann, 50 %, n/4)") -> int:
    """Kleinster ganzzahliger AR-Anteil (Prozent), bei dem `first` nach gewichteten Siegen vor `second` liegt (101, wenn nie)."""
    cells = RES.cells("A")
    for p in range(0, 101):
        summ = R.summarize("A", cells, R.mix_weights(cells, p / 100.0))
        if summ[first]["wins_weighted"] > summ[second]["wins_weighted"] + 1e-12:
            return p
    return 101


def exp_lines() -> dict:
    b, c = RES.cells("B"), RES.cells("C")
    return {"b": b, "c": c, "summary_b": R.summarize("B", b), "summary_c": R.summarize("C", c)}


def exp_detection() -> dict:
    d = RES.cells("D")
    return {"cells": d, "summary": R.summarize("D", d)}


def exp_irregular() -> dict:
    e, f = RES.cells("E"), RES.cells("F")
    return {"e": e, "f": f, "summary_e": R.summarize("E", e), "summary_f": R.summarize("F", f)}


def exp_denoise() -> dict:
    g = RES.cells("G")
    return {"cells": g, "summary": R.summarize("G", g)}


def overall() -> dict:
    """{Aufgabe: {Verfahren: Zusammenfassung}} über alle Zellen der Aufgabe (gleich gewichtet)."""
    return {t: R.summarize(t, RES.cells(t)) for t in C.TASKS}


def verdict_spectrum(res: dict) -> tuple[str, str]:
    summ, eq, w = res["summary"], res["equal"], res["wins_by_type"]
    ar, hann = "AR (Burg, BIC)", "Hann-Periodogramm"
    cross = wins_crossover()
    return "warning", (f"Das AR-Spektrum (Burg, BIC) gewinnt {w[ar]['ar']} von {res['n_ar']} Szenarien mit autoregressivem oder weißem Hintergrund, aber nur {w[ar]['non_ar']} von {res['n_non_ar']} mit Kerbe oder Bandpass. "
                       f"Unter dem eingestellten Mix ({de(100 * res['share'], 0)} % autoregressiv) ist sein mittlerer Reuegrad {de(summ[ar]['mean'], 2)} und sein größter {de(summ[ar]['max'], 2)}; "
                       f"bei Welch {de(summ['Welch (Hann, 50 %, n/4)']['mean'], 2)} und {de(summ['Welch (Hann, 50 %, n/4)']['max'], 1)}, beim Multitaper {de(summ['Multitaper (NW = 4)']['mean'], 2)} und "
                       f"{de(summ['Multitaper (NW = 4)']['max'], 1)}, beim Hann-Periodogramm {de(summ[hann]['mean'], 1)} und {de(summ[hann]['max'], 1)}. Minimax-Wahl: {res['minimax']}. "
                       f"Nach Siegen gewinnt das AR-Spektrum erst ab einem autoregressiven Anteil von {cross} % vor Welch; nach mittlerem und größtem Reuegrad liegt es in diesen Szenarien bei jedem Mix vorn. Ein Vergleich ist so fair wie seine Szenarien.")


def verdict_lines(res: dict) -> tuple[str, str]:
    b = {(c["params"]["delta"], c["params"]["snr"]): c for c in res["b"]}
    cb = b[(1.0, 10.0)]["values"]
    cc = {(c["params"]["n"], c["params"]["snr"]): c for c in res["c"]}
    c1 = cc[(64, 10.0)]["values"]
    sc = res["summary_c"]
    return "info", (f"Zwei Linien im Abstand von einem Bin (n = 64, 10 dB je Linie): ESPRIT trennt sie in {de(cb['ESPRIT (p = 4)'], 1)} %, MUSIC in {de(cb['MUSIC (p = 4)'], 1)} %, das AR-Spektrum mit Ordnung 16 in "
                    f"{de(cb['AR (Burg, p = 16)'], 1)} %, mit BIC-Ordnung in {de(cb['AR (Burg, BIC)'], 1)} %, das Hann-Periodogramm in {de(cb['Hann-Periodogramm'], 1)} % und das Multitaper in "
                    f"{de(cb['Multitaper (NW = 2)'], 1)} %. Für eine einzelne Linie (n = 64, 10 dB) liegt der Fehler der Spitze des Rechteck-Periodogramms bei {de(c1['Spitze (Rechteck, 16-fach, Parabel)'], 2)}, "
                    f"ESPRIT bei {de(c1['ESPRIT (p = 2)'], 2)}, MUSIC bei {de(c1['MUSIC (p = 2)'], 2)}, Jacobsen bei {de(c1['Jacobsen (3 Bins)'], 2)} und der AR-Spitze bei {de(c1['AR-Spitze (Burg, p = 8)'], 1)} "
                    f"Cramér-Rao-Schranken; mittlerer Reuegrad der Spitze {de(sc['Spitze (Rechteck, 16-fach, Parabel)']['mean'], 2)}, größter {de(sc['Spitze (Rechteck, 16-fach, Parabel)']['max'], 2)}.")


def verdict_detection(res: dict) -> tuple[str, str]:
    cells = {(c["params"]["bg"], c["params"]["snr"]): c for c in res["cells"]}
    col = cells[("AR(1) 0,8", -12.0)]
    fa = col["extra"]["false_alarm_nominal"]
    white = cells[("weiß", -6.0)]["values"]
    return "warning", (f"Bei AR(1)-Hintergrund (φ = 0,8) schlägt Fishers Test auf weißes Rauschen in {de(100 * fa['Fisher-g (weißer Hintergrund)'], 1)} % der reinen Rauschläufe an; auf wirklich 5 % kalibriert findet er eine Linie von −12 dB in "
                       f"{de(col['values']['Fisher-g (weißer Hintergrund)'], 0)} % der Läufe, mit AR(2)-Vorfilterung in {de(col['values']['Fisher-g nach AR(2)-Vorfilterung'], 0)} % "
                       f"und der harmonische F-Test in {de(col['values']['Harmonischer F-Test (NW = 4)'], 0)} %. Bei weißem Hintergrund und −6 dB sind es {de(white['Fisher-g (weißer Hintergrund)'], 0)} %, "
                       f"{de(white['Fisher-g nach AR(2)-Vorfilterung'], 0)} % und {de(white['Harmonischer F-Test (NW = 4)'], 0)} %. Der F-Test mit Theorieschwelle hat bei AR(1)-Hintergrund nominell {de(100 * fa['Harmonischer F-Test (NW = 4)'], 0)} % Falschalarm statt 5 %.")


def verdict_irregular(res: dict) -> tuple[str, str]:
    e = {(c["params"]["line"], c["params"]["snr"]): c["values"] for c in res["e"]}
    f = {(c["params"]["signal"], c["params"]["snr"]): c["values"] for c in res["f"]}
    hi, lo = e[(1.3, 0.0)], e[(0.2, 0.0)]
    low6 = e[(0.2, -6.0)]
    c10, c0, s20 = f[("Chirp", 10.0)], f[("Chirp", 0.0)], f[("Sinus-FM", 20.0)]
    return "info", (f"Zufällige Messzeiten erkennen eine Linie bei der 2,6fachen mittleren Nyquist-Frequenz in {de(hi['zufällig'], 0)} % der Läufe (0 dB), regelmäßige in {de(hi['regelmäßig'], 0)} %. Unter der Nyquist-Grenze verliert das Interpolieren "
                    f"aufs Gitter: {de(lo[C.E_INTERP], 0)} % gegen {de(lo['zufällig'], 0)} % bei 0 dB und {de(low6[C.E_INTERP], 0)} % gegen {de(low6['zufällig'], 0)} % bei −6 dB. Beim Chirp mit 10 dB liegen Neu zuordnen "
                    f"({de(c10['Neu zugeordnet'], 5)}), Spektrogramm ({de(c10['Spektrogramm (Gauß)'], 5)}) und Synchrosqueezing ({de(c10['Synchrosqueezing'], 5)}) vor dem CWT-Kamm ({de(c10['CWT-Kamm (Morlet)'], 5)}); bei 0 dB bricht "
                    f"Synchrosqueezing ein ({de(c0['Synchrosqueezing'], 4)} gegen {de(c0['Spektrogramm (Gauß)'], 4)}). Bei der Sinus-FM (20 dB) liegen die Phasenableitung ({de(s20['Phasenableitung (21 geglättet)'], 4)}) "
                    f"und der CWT-Kamm ({de(s20['CWT-Kamm (Morlet)'], 4)}) vor dem Spektrogramm ({de(s20['Spektrogramm (Gauß)'], 4)}).")


def verdict_denoise(res: dict) -> tuple[str, str]:
    g = {c["params"]["signal"]: c for c in res["cells"]}
    sure = "Wavelet-SURE (D8, weich)"
    fou = "Fourier universal (hart)"
    return "success", (f"Wavelet-SURE gewinnt alle vier Signale (Blocks {de(g['Blocks']['values'][sure], 2)}, Bumps {de(g['Bumps']['values'][sure], 2)}, HeaviSine {de(g['HeaviSine']['values'][sure], 2)}, Doppler "
                       f"{de(g['Doppler']['values'][sure], 2)}) vor der Fourier-Schwelle ({de(g['Blocks']['values'][fou], 2)}, {de(g['Bumps']['values'][fou], 2)}, {de(g['HeaviSine']['values'][fou], 2)}, "
                       f"{de(g['Doppler']['values'][fou], 2)}); bei Blocks und Bumps ist selbst die beste Fourier-Schwelle (Orakel) mit {de(g['Blocks']['refs']['Fourier beste Schwelle (Orakel)'], 2)} und "
                       f"{de(g['Bumps']['refs']['Fourier beste Schwelle (Orakel)'], 2)} kaum besser als die verrauschte Reihe ({de(g['Blocks']['refs']['verrauscht'], 2)}). Die Basis entscheidet: Sprünge sind in Wavelets dünn, in Fourier nicht.")


# --- Tabellen der Experimente -------------------------------------------------------------------------------------------------------------


def table_rows(task: str, cells: list, with_refs: bool = False) -> list:
    """Eine Zeile je Zelle: Szenario und Kennzahl je Verfahren (Dezimalkomma); optional die Bezüge der Zelle."""
    d = DIGITS[task]
    rows = []
    for c in cells:
        row = {"Szenario": R.cell_label(task, c["params"])}
        for k in C.METHODS[task]:
            if k in c["values"]:
                row[k] = de(c["values"][k], d)
        if with_refs:
            for k, v in c["refs"].items():
                row[k] = de(v, d)
        rows.append(row)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    return [{k: row.get(k, "–") for k in keys} for row in rows]


def summary_rows(task: str, summary: dict) -> list:
    """Eine Zeile je Verfahren: mittlerer und größter Reuegrad, Siege (gewichtet als Anteil)."""
    ratio = C.TASK_KIND[task] == C.KIND_ERROR
    d = 2 if ratio else 1
    unit = "" if ratio else " P."
    return [{"Verfahren": k, "mittlerer Reuegrad": de(v["mean"], d) + unit, "größter Reuegrad": de(v["max"], d) + unit, "Siege": f"{v['wins']} von {v['n']}", "Siegesanteil": f"{de(100 * v['wins_weighted'], 0)} %"}
            for k, v in summary.items()]


def advice_rows(task: str, advice) -> list:
    """Zeilen der Entscheidungshilfe: Wert und Reuegrad im nächstgelegenen Szenario, größter und mittlerer Reuegrad über alle Szenarien, schlechtestes Szenario."""
    ratio = C.TASK_KIND[task] == C.KIND_ERROR
    d = 2 if ratio else 1
    unit = "" if ratio else " P."
    return [{"Verfahren": r["method"], f"Wert ({C.TASK_UNIT[task]})": de(r["value"], DIGITS[task]), "Reuegrad hier": de(r["regret"], d) + unit, "größter Reuegrad": de(r["max_regret"], d) + unit,
             "mittlerer Reuegrad": de(r["mean_regret"], d) + unit, "am schlechtesten bei": r["worst_cell"]} for r in advice.rows]
