"""Jede Zahl und Aussage in den Texten der App (Presets, Verdikte, Einleitung, Grenzen) und in der README ist hier an die Tabelle der Experimente gebunden."""

import math
import re
from pathlib import Path

import pytest

import spv_constants as C
import spv_evaluation as ev
import spv_regret as R
import spv_results as RES
from sweeps_cache import get

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
APP = (ROOT / "app.py").read_text(encoding="utf-8")
de = ev.de
P = list(C.PRESETS)
AR, WELCH, MT, HANN = "AR (Burg, BIC)", "Welch (Hann, 50 %, n/4)", "Multitaper (NW = 4)", "Hann-Periodogramm"


def v(task, **params):
    return RES.find(task, **params)["values"]


def mean_regret(share):
    cells = RES.cells("A")
    return R.summarize("A", cells, R.mix_weights(cells, share))


def has(text, *needles):
    missing = [n for n in needles if n not in text]
    assert not missing, missing


# --- Presets --------------------------------------------------------------------------------------------------------------------------------


def test_preset_help_steil():
    a = v("A", bg="Steil", n=256)
    assert C.PRESETS[P[0]]["a_bg"] == "Steil" and C.PRESETS[P[0]]["a_n"] == 256
    has(C.PRESET_HELP[P[0]], f"etwa {de(a[AR], 1)} dB Fehler", f"Welch bei {de(a[WELCH], 1)} dB", f"Multitaper bei {de(a[MT], 1)} dB", f"Hann-Periodogramm bei {de(a[HANN], 1)} dB", "Steiles autoregressives Spektrum, n = 256")
    assert min(a, key=a.get) == AR


def test_preset_help_kerbe():
    a = v("A", bg="Kerbe", n=1024)
    assert C.PRESETS[P[1]]["a_bg"] == "Kerbe" and C.PRESETS[P[1]]["a_n"] == 1024
    has(C.PRESET_HELP[P[1]], f"Welch und Multitaper liegen bei etwa {de(a[WELCH], 1)} und {de(a[MT], 1)} dB", f"das AR-Spektrum bei {de(a[AR], 1)} dB", "Hintergrund mit Kerbe (Nullstelle), n = 1024")
    assert min(a, key=a.get) == WELCH and a[AR] > a[MT] > a[WELCH]
    bp = v("A", bg="Bandpass", n=1024)
    assert bp[MT] > 5 * bp[WELCH] and "Beim Bandpass leckt das Multitaper noch stärker." in C.PRESET_HELP[P[1]]


def test_preset_help_two_lines():
    b = v("B", delta=1.0, snr=10.0)
    assert (C.PRESETS[P[2]]["b_delta"], C.PRESETS[P[2]]["b_snr"]) == (1.0, 10.0)
    has(C.PRESET_HELP[P[2]], "Zwei gleich starke Linien im Abstand von einem Bin (n = 64, 10 dB je Linie)", "ESPRIT trennt sie, MUSIC fast immer", "Ordnung 16 in etwa der Hälfte der Läufe", "das Hann-Periodogramm und das Multitaper nicht")
    assert b["ESPRIT (p = 4)"] == 100 and b["MUSIC (p = 4)"] > 90 and 40 <= b["AR (Burg, p = 16)"] <= 60 and b[HANN] < 15 and b["Multitaper (NW = 2)"] == 0


def test_preset_help_fisher():
    d = RES.find("D", bg="AR(1) 0,8", snr=-6.0)
    assert (C.PRESETS[P[3]]["d_bg"], C.PRESETS[P[3]]["d_snr"]) == ("AR(1) 0,8", -6.0)
    has(C.PRESET_HELP[P[3]], "Linie von −6 dB in AR(1)-Rauschen (φ = 0,8), n = 128", "in fast jedem Rauschlauf an", "keine Trennschärfe", "AR(2)-Vorfilterung oder dem F-Test")
    assert d["extra"]["false_alarm_nominal"]["Fisher-g (weißer Hintergrund)"] > 0.95 and d["values"]["Fisher-g (weißer Hintergrund)"] < 5 and d["values"]["Fisher-g nach AR(2)-Vorfilterung"] > 95 and d["values"]["Harmonischer F-Test (NW = 4)"] > 95


def test_preset_help_lomb_scargle():
    e = v("E", line=1.3, snr=0.0)
    assert (C.PRESETS[P[4]]["e_line"], C.PRESETS[P[4]]["e_snr"]) == (1.3, 0.0)
    has(C.PRESET_HELP[P[4]], "Linie bei der 2,6fachen mittleren Nyquist-Frequenz (n = 64, 0 dB)", "in allen Läufen", "bei regelmäßigen nur in etwa einem Viertel")
    assert e["zufällig"] == 100 and 20 <= e["regelmäßig"] <= 30 and 1.3 / 0.5 == pytest.approx(2.6)


# --- Verdikte der Experimente ---------------------------------------------------------------------------------------------------------------


def test_spectrum_verdict_numbers():
    r = get("spectrum")
    text = ev.verdict_spectrum(r)[1]
    s, eq = r["summary"], r["equal"]
    cells = RES.cells("A")
    ar_cells = [c for c in cells if c["params"]["bg"] in C.AR_TYPE]
    non = [c for c in cells if c["params"]["bg"] not in C.AR_TYPE]
    assert (len(ar_cells), len(non)) == (15, 6)
    assert sum(R.winners("A", c) == [AR] for c in ar_cells) == 15 and sum(AR in R.winners("A", c) for c in non) == 1
    has(text, "gewinnt 15 von 15 Szenarien mit autoregressivem oder weißem Hintergrund, aber nur 1 von 6 mit Kerbe oder Bandpass", f"mittlerer Reuegrad {de(s[AR]['mean'], 2)} und sein größter {de(s[AR]['max'], 2)}",
        f"bei Welch {de(s[WELCH]['mean'], 2)} und {de(s[WELCH]['max'], 1)}", f"beim Multitaper {de(s[MT]['mean'], 2)} und {de(s[MT]['max'], 1)}", f"beim Hann-Periodogramm {de(s[HANN]['mean'], 1)} und {de(s[HANN]['max'], 1)}",
        "Minimax-Wahl: AR (Burg, BIC)", "Unter dem eingestellten Mix (71 % autoregressiv)")
    assert eq[AR]["mean"] < 1.2 and r["minimax"] == AR


def test_the_ar_spectrum_is_best_by_mean_and_max_at_every_mix_but_not_by_wins():
    cells = RES.cells("A")
    for share in [i / 20 for i in range(21)]:
        s = R.summarize("A", cells, R.mix_weights(cells, share))
        assert min(s, key=lambda k: s[k]["mean"]) == AR and R.minimax_choice(s) == AR, share
    low = R.summarize("A", cells, R.mix_weights(cells, 0.0))
    assert max(low, key=lambda k: low[k]["wins_weighted"]) == WELCH and low[WELCH]["wins_weighted"] == pytest.approx(0.5)
    assert ev.wins_crossover() == 26
    q = 0.26                                                                      # Siegesanteil AR: q + (1 - q)/6, Welch: (1 - q)/2
    assert q + (1 - q) / 6 > (1 - q) / 2 and 0.25 + 0.75 / 6 == pytest.approx(0.75 / 2)


def test_lines_verdict_numbers():
    b = v("B", delta=1.0, snr=10.0)
    c = RES.find("C", n=64, snr=10.0)["values"]
    r = get("lines")
    text = ev.verdict_lines(r)[1]
    has(text, f"ESPRIT trennt sie in {de(b['ESPRIT (p = 4)'], 1)} %", f"MUSIC in {de(b['MUSIC (p = 4)'], 1)} %", f"Ordnung 16 in {de(b['AR (Burg, p = 16)'], 1)} %", f"BIC-Ordnung in {de(b['AR (Burg, BIC)'], 1)} %",
        f"Hann-Periodogramm in {de(b[HANN], 1)} %", f"Multitaper in {de(b['Multitaper (NW = 2)'], 1)} %", f"Rechteck-Periodogramms bei {de(c['Spitze (Rechteck, 16-fach, Parabel)'], 2)}", f"ESPRIT bei {de(c['ESPRIT (p = 2)'], 2)}",
        f"MUSIC bei {de(c['MUSIC (p = 2)'], 2)}", f"Jacobsen bei {de(c['Jacobsen (3 Bins)'], 2)}", f"AR-Spitze bei {de(c['AR-Spitze (Burg, p = 8)'], 1)}".replace("AR-Spitze bei", "AR-Spitze bei"))
    assert (b["ESPRIT (p = 4)"], b["MUSIC (p = 4)"], b["AR (Burg, p = 16)"], b["AR (Burg, BIC)"], b[HANN], b["Multitaper (NW = 2)"]) == (100.0, 96.5, 45.0, 0.5, 12.5, 0.0)
    assert min(x["values"]["ESPRIT (p = 4)"] for x in r["b"]) == 94.0 and r["summary_b"]["ESPRIT (p = 4)"]["max"] == 0.0 and r["summary_b"]["ESPRIT (p = 4)"]["wins"] == 8


def test_detection_verdict_numbers():
    d = RES.find("D", bg="AR(1) 0,8", snr=-12.0)
    w = v("D", bg="weiß", snr=-6.0)
    text = ev.verdict_detection(get("detection"))[1]
    fa = d["extra"]["false_alarm_nominal"]
    has(text, f"in {de(100 * fa['Fisher-g (weißer Hintergrund)'], 1)} % der reinen Rauschläufe", f"in {de(d['values']['Fisher-g (weißer Hintergrund)'], 0)} % der Läufe", f"Vorfilterung in {de(d['values']['Fisher-g nach AR(2)-Vorfilterung'], 0)} %",
        f"F-Test in {de(d['values']['Harmonischer F-Test (NW = 4)'], 0)} %", f"sind es {de(w['Fisher-g (weißer Hintergrund)'], 0)} %, {de(w['Fisher-g nach AR(2)-Vorfilterung'], 0)} % und {de(w['Harmonischer F-Test (NW = 4)'], 0)} %",
        f"hat bei AR(1)-Hintergrund {de(100 * fa['Harmonischer F-Test (NW = 4)'], 0)} % und schon bei weißem Hintergrund {de(100 * RES.find('D', bg='weiß', snr=-6.0)['extra']['false_alarm_nominal']['Harmonischer F-Test (NW = 4)'], 0)} % Falschalarm statt 5 %")
    assert (de(100 * fa["Fisher-g (weißer Hintergrund)"], 1), de(d["values"]["Fisher-g (weißer Hintergrund)"], 0)) == ("99,2", "5")


def test_irregular_verdict_numbers():
    e_hi, e_lo, e_lo6 = v("E", line=1.3, snr=0.0), v("E", line=0.2, snr=0.0), v("E", line=0.2, snr=-6.0)
    f10, f0, s20 = v("F", signal="Chirp", snr=10.0), v("F", signal="Chirp", snr=0.0), v("F", signal="Sinus-FM", snr=20.0)
    text = ev.verdict_irregular(get("irregular"))[1]
    has(text, f"in {de(e_hi['zufällig'], 0)} % der Läufe (0 dB), regelmäßige in {de(e_hi['regelmäßig'], 0)} %", f"{de(e_lo[C.E_INTERP], 0)} % gegen {de(e_lo['zufällig'], 0)} % bei 0 dB", f"{de(e_lo6[C.E_INTERP], 0)} % gegen {de(e_lo6['zufällig'], 0)} % bei −6 dB",
        f"Neu zuordnen ({de(f10['Neu zugeordnet'], 5)})", f"Spektrogramm ({de(f10['Spektrogramm (Gauß)'], 5)})", f"Synchrosqueezing ({de(f10['Synchrosqueezing'], 5)})", f"CWT-Kamm ({de(f10['CWT-Kamm (Morlet)'], 5)})",
        f"({de(f0['Synchrosqueezing'], 4)} gegen {de(f0['Spektrogramm (Gauß)'], 4)})", f"Phasenableitung ({de(s20['Phasenableitung (21 geglättet)'], 4)})", f"CWT-Kamm ({de(s20['CWT-Kamm (Morlet)'], 4)})", f"Spektrogramm ({de(s20['Spektrogramm (Gauß)'], 4)})")
    assert max(f10["Neu zugeordnet"], f10["Spektrogramm (Gauß)"], f10["Synchrosqueezing"]) < f10["CWT-Kamm (Morlet)"] and f0["Synchrosqueezing"] > 10 * f0["Spektrogramm (Gauß)"]
    assert max(s20["Phasenableitung (21 geglättet)"], s20["CWT-Kamm (Morlet)"]) < s20["Spektrogramm (Gauß)"]


def test_denoise_verdict_numbers():
    g = {name: RES.find("G", signal=name) for name in C.G_SIGNALS}
    sure, fou = "Wavelet-SURE (D8, weich)", "Fourier universal (hart)"
    text = ev.verdict_denoise(get("denoise"))[1]
    has(text, f"Blocks {de(g['Blocks']['values'][sure], 2)}", f"Bumps {de(g['Bumps']['values'][sure], 2)}", f"HeaviSine {de(g['HeaviSine']['values'][sure], 2)}", f"Doppler {de(g['Doppler']['values'][sure], 2)}",
        f"({de(g['Blocks']['values'][fou], 2)}, {de(g['Bumps']['values'][fou], 2)}, {de(g['HeaviSine']['values'][fou], 2)}, {de(g['Doppler']['values'][fou], 2)})",
        f"(Orakel) mit {de(g['Blocks']['refs']['Fourier beste Schwelle (Orakel)'], 2)} und {de(g['Bumps']['refs']['Fourier beste Schwelle (Orakel)'], 2)}", f"verrauschte Reihe ({de(g['Blocks']['refs']['verrauscht'], 2)})")
    assert all(min(c["values"], key=c["values"].get) == sure for c in g.values())


# --- README ---------------------------------------------------------------------------------------------------------------------------------


def test_readme_spectrum_rows():
    steil, kerbe, band = v("A", bg="Steil", n=256), v("A", bg="Kerbe", n=1024), v("A", bg="Bandpass", n=256)
    has(README, f"AR {de(steil[AR], 2)} dB, Welch {de(steil[WELCH], 2)} dB, Multitaper {de(steil[MT], 2)} dB, Hann-Periodogramm {de(steil[HANN], 2)} dB",
        f"Welch {de(kerbe[WELCH], 2)} dB, Multitaper {de(kerbe[MT], 2)} dB, AR {de(kerbe[AR], 2)} dB",
        f"Hann-Periodogramm {de(band[HANN], 2)} dB, AR {de(band[AR], 2)} dB, Welch {de(band[WELCH], 2)} dB, Multitaper {de(band[MT], 2)} dB", "gewinnt 15 von 15 Szenarien", "nur 1 von 6")
    assert min(band, key=band.get) == HANN and sorted(band, key=band.get) == [HANN, AR, WELCH, MT]
    assert sorted(steil, key=steil.get) == [AR, WELCH, MT, HANN] and sorted(kerbe, key=kerbe.get)[:3] == [WELCH, MT, AR]


def test_readme_regret_row():
    eq, s71, s0 = get("spectrum")["equal"], mean_regret(0.71), mean_regret(0.0)
    has(README, f"AR {de(eq[AR]['max'], 2)}, Multitaper {de(eq[MT]['max'], 2)}, Welch {de(eq[WELCH]['max'], 2)}, Hann-Periodogramm {de(eq[HANN]['max'], 2)}",
        f"{de(s71[AR]['mean'], 2)} (AR), {de(s71[WELCH]['mean'], 2)} (Welch), {de(s71[MT]['mean'], 2)} (Multitaper) und {de(s71[HANN]['mean'], 2)} (Hann-Periodogramm)",
        f"bei 0 % autoregressiv {de(s0[AR]['mean'], 2)}, {de(s0[WELCH]['mean'], 2)}, {de(s0[MT]['mean'], 2)} und {de(s0[HANN]['mean'], 2)}", "ab einem autoregressiven Anteil von 26 % vor Welch", f"({de(s0[AR]['mean'], 2)} gegen {de(s0[WELCH]['mean'], 2)} bei Welch)")
    assert sorted(eq, key=lambda k: eq[k]["max"]) == [AR, MT, WELCH, HANN]


def test_readme_line_rows():
    b = v("B", delta=1.0, snr=10.0)
    r = get("lines")
    c = v("C", n=64, snr=10.0)
    sb, sc = r["summary_b"], r["summary_c"]
    has(README, f"ESPRIT {de(b['ESPRIT (p = 4)'], 1)} % der Läufe, MUSIC {de(b['MUSIC (p = 4)'], 1)} %, das AR-Spektrum mit Ordnung 16 nur {de(b['AR (Burg, p = 16)'], 1)} %, das Hann-Periodogramm {de(b[HANN], 1)} %, "
        f"das AR-Spektrum mit BIC-Ordnung {de(b['AR (Burg, BIC)'], 1)} % und das Multitaper {de(b['Multitaper (NW = 2)'], 1)} %",
        f"MUSIC ist {de(sb['MUSIC (p = 4)']['max'], 0)} Punkte, der des AR-Spektrums mit Ordnung 16 {de(sb['AR (Burg, p = 16)']['max'], 0)}", "in allen acht Szenarien mindestens 94,0 % und hat überall den Reuegrad null",
        f"{de(c['Spitze (Rechteck, 16-fach, Parabel)'], 2)} Cramér-Rao-Schranken, bei ESPRIT {de(c['ESPRIT (p = 2)'], 2)}, MUSIC {de(c['MUSIC (p = 2)'], 2)}, Jacobsen und Candan {de(c['Jacobsen (3 Bins)'], 2)} und der AR-Spitze {de(c['AR-Spitze (Burg, p = 8)'], 2)}",
        f"Spitze {de(sc['Spitze (Rechteck, 16-fach, Parabel)']['mean'], 2)}, größter {de(sc['Spitze (Rechteck, 16-fach, Parabel)']['max'], 2)}", f"die AR-Spitze hat {de(sc['AR-Spitze (Burg, p = 8)']['mean'], 2)} und {de(sc['AR-Spitze (Burg, p = 8)']['max'], 2)}")
    assert v("C", n=64, snr=10.0)["Jacobsen (3 Bins)"] == pytest.approx(v("C", n=64, snr=10.0)["Candan (3 Bins)"], abs=0.005) and len(RES.cells("B")) == 8


def test_readme_detection_row():
    d = RES.find("D", bg="AR(1) 0,8", snr=-12.0)
    fa = d["extra"]["false_alarm_nominal"]
    summ = get("detection")["summary"]
    has(README, f"in {de(100 * fa['Fisher-g (weißer Hintergrund)'], 1)} % der reinen Rauschläufe", "Linie von −12 dB in 5 %, mit AR(2)-Vorfilterung in 72 % und der harmonische F-Test in 38 %", "sind es 68 %, 82 % und 52 %",
        f"und bei AR(1)-Hintergrund {de(100 * fa['Harmonischer F-Test (NW = 4)'], 0)} % Falschalarm statt 5 %", "ist in allen zehn Szenarien Sieger oder gleichauf")
    assert summ["Fisher-g nach AR(2)-Vorfilterung"]["wins"] == 10 and len(RES.cells("D")) == 10


def test_readme_irregular_and_time_frequency_rows():
    e0, e6 = v("E", line=1.3, snr=0.0), v("E", line=0.2, snr=-6.0)
    e00 = v("E", line=0.2, snr=0.0)
    f10, f0, s20 = v("F", signal="Chirp", snr=10.0), v("F", signal="Chirp", snr=0.0), v("F", signal="Sinus-FM", snr=20.0)
    sf = get("irregular")["summary_f"]
    has(README, f"zufällige Messzeiten {de(e0['zufällig'], 0)} % der Läufe, regelmäßige {de(e0['regelmäßig'], 0)} %, Jitter {de(e0['Jitter ±0,25'], 0)} %, Nachtlücken {de(e0['Nachtlücken'], 0)} %",
        f"{de(e00[C.E_INTERP], 0)} % gegen {de(e00['zufällig'], 0)} % bei 0 dB und {de(e6[C.E_INTERP], 0)} % gegen {de(e6['zufällig'], 0)} % bei −6 dB",
        f"Neuzuordnung ({de(f10['Neu zugeordnet'], 5)}), Spektrogramm ({de(f10['Spektrogramm (Gauß)'], 5)}) und Synchrosqueezing ({de(f10['Synchrosqueezing'], 5)}) vor dem CWT-Kamm ({de(f10['CWT-Kamm (Morlet)'], 5)})",
        f"({de(f0['Synchrosqueezing'], 4)} gegen {de(f0['Spektrogramm (Gauß)'], 4)})", f"Phasenableitung ({de(s20['Phasenableitung (21 geglättet)'], 4)}) und der CWT-Kamm ({de(s20['CWT-Kamm (Morlet)'], 4)}) vor dem Spektrogramm ({de(s20['Spektrogramm (Gauß)'], 4)})",
        f"Neuzuordnung {de(sf['Neu zugeordnet']['mean'], 2)} (größter {de(sf['Neu zugeordnet']['max'], 2)}), Phasenableitung {de(sf['Phasenableitung (21 geglättet)']['mean'], 2)} ({de(sf['Phasenableitung (21 geglättet)']['max'], 2)}), "
        f"Synchrosqueezing {de(sf['Synchrosqueezing']['mean'], 2)} ({de(sf['Synchrosqueezing']['max'], 2)}, in der rauschfreien Chirp-Zelle vom Frequenzraster bestimmt)")


def test_readme_denoise_row():
    g = {name: RES.find("G", signal=name) for name in C.G_SIGNALS}
    sure, fou = "Wavelet-SURE (D8, weich)", "Fourier universal (hart)"
    has(README, f"(Blocks {de(g['Blocks']['values'][sure], 2)}, Bumps {de(g['Bumps']['values'][sure], 2)}, HeaviSine {de(g['HeaviSine']['values'][sure], 2)}, Doppler {de(g['Doppler']['values'][sure], 2)})",
        f"({de(g['Blocks']['values'][fou], 2)}, {de(g['Bumps']['values'][fou], 2)}, {de(g['HeaviSine']['values'][fou], 2)}, {de(g['Doppler']['values'][fou], 2)})",
        f"(Orakel) mit {de(g['Blocks']['refs']['Fourier beste Schwelle (Orakel)'], 2)} und {de(g['Bumps']['refs']['Fourier beste Schwelle (Orakel)'], 2)} kaum besser als die verrauschte Reihe ({de(g['Blocks']['refs']['verrauscht'], 2)} und {de(g['Bumps']['refs']['verrauscht'], 2)})")


def test_readme_counts_and_methodology():
    assert len(RES.cells("A")) == 21 and len(RES.load()["cells"]) == 63
    has(README, "63 Szenarien", "100 Läufe je Szenario bei A, 200 bei B, 300 bei C, 300 bei D", "(dazu 400 Rauschläufe für die Schwellen), 300 bei E, 20 bei F (verrauscht) und 40 bei G", "Sechzehnte und letzte Demo")
    assert (C.RUNS_A, C.RUNS_B, C.RUNS_C, C.RUNS_D, C.RUNS_D_NULL, C.RUNS_E, C.RUNS_F, C.RUNS_G) == (100, 200, 300, 300, 400, 300, 20, 40)
    assert (C.N_B, C.N_D, C.N_E, C.N_F, C.N_G) == (64, 128, 64, 512, 1024) and (C.B_DELTAS[0], C.B_DELTAS[-1]) == (0.75, 2.5) and C.D_POLES["AR(1) 0,8"] == ((0.0, 0.8),)
    assert C.E_LINES[0] / 0.5 == pytest.approx(2.6) and C.E_LINES[1] / 0.5 == pytest.approx(0.4) and len(C.AR_TYPE) == 5 and len(C.BACKGROUNDS_A) == 7
    has(README, "Fünf der sieben Hintergründe der Aufgabe A", "bei Anteilen mit 300 Läufen höchstens 2,9 Punkte Standardfehler")
    assert de(math.sqrt(0.25 / 300) * 100, 1) == "2,9"


def test_readme_limit_about_interpolation_and_the_mix():
    cells = [c for c in RES.cells("E") if C.E_INTERP in c["values"]]
    assert len(cells) == 3 and all(c["params"]["line"] < 0.5 for c in cells) and get("irregular") is not None
    assert "sein Reuegrad stammt aus den drei Szenarien darunter" in README


# --- Texte der App --------------------------------------------------------------------------------------------------------------------------


def test_app_intro_and_limits_statements():
    has(APP, "sechzehnte und letzte Demo", "sieben Aufgaben mit\nbekannter Wahrheit", "Neue Verfahren kommen nicht vor", "Das AR-Spektrum gewinnt jedes Szenario mit autoregressivem\nHintergrund und verliert in fünf von sechs Szenarien mit Kerbe oder Bandpass die Siege")
    assert len(C.TASKS) == 7 and all(len(m) >= 3 for m in C.METHODS.values())
    cells = RES.cells("A")
    assert all(R.winners("A", c) == [AR] for c in cells if c["params"]["bg"] in C.AR_TYPE) and sum(R.winners("A", c) != [AR] for c in cells if c["params"]["bg"] in C.NON_AR_TYPE) == 5
    d_fisher = RES.find("D", bg="AR(1) 0,8", snr=-12.0)["extra"]["false_alarm_nominal"]["Fisher-g (weißer Hintergrund)"]
    assert d_fisher > 0.95                                                       # „ein Test auf weißes Rauschen schlägt in gefärbtem Rauschen fast immer an“


def test_app_statement_that_the_vehicle_is_an_unseen_scenario_choice():
    has(APP, "Dies ist die sechzehnte und letzte Demo der Reihe „Spektral- und Wavelet-Analyse“", "die Wahrheit ist bekannt", "Nur über diese Szenarien, nie allgemein")
    assert re.search(r"Anteil der autoregressiven Hintergründe im Mix \(%\)", APP)


def test_readme_why_statements():
    has(README, "In dieser Tabelle gewinnt das AR-Spektrum jedes Szenario mit autoregressivem Hintergrund und verliert bei der Kerbe in allen drei Längen (n = 256 und 1024 gegen Welch, n = 64 gegen Multitaper und Welch)", "in AR(1)-Rauschen nutzlos",
        "Synchrosqueezing liegt beim Chirp mit 10 dB gleichauf mit dem Spektrogramm und bricht bei 0 dB ein")
    for n in (256, 1024):
        k = v("A", bg="Kerbe", n=n)
        assert min(k, key=k.get) == WELCH and k[AR] > k[WELCH]
    f10, f0 = v("F", signal="Chirp", snr=10.0), v("F", signal="Chirp", snr=0.0)
    assert f10["Synchrosqueezing"] == pytest.approx(f10["Spektrogramm (Gauß)"], rel=0.1) and f0["Synchrosqueezing"] > 10 * f0["Spektrogramm (Gauß)"]
    assert max(RES.find("D", bg="AR(1) 0,8", snr=s)["values"]["Fisher-g (weißer Hintergrund)"] for s in C.D_SNRS) <= 5.0           # auf 5 % kalibriert: höchstens der Zufall
    k64 = v("A", bg="Kerbe", n=64)
    assert min(k64, key=k64.get) == MT and k64[AR] > k64[WELCH] > k64[MT]
    s10 = v("F", signal="Sinus-FM", snr=10.0)
    assert s10["Synchrosqueezing"] > 5 * s10["Spektrogramm (Gauß)"] and "bei der Sinus-FM schon bei 10 dB" in README


def test_readme_and_app_qualifications_after_the_independent_review():
    import spv_estimators as E
    wh = v("A", bg="weiß", n=1024)
    assert round(wh[HANN] / wh[AR], 2) == 28.31 and de(wh[AR], 2) == "0,22" and "wo der beste Fehler nur 0,22 dB beträgt" in README
    b = v("B", delta=1.0, snr=10.0)
    r = get("lines")
    mdl = "ESPRIT, Linienzahl per MDL"
    assert b[mdl] == 93.5 and min(x["values"][mdl] for x in r["b"]) == 85.0 and r["summary_b"][mdl]["max"] == 9.0
    has(README, "wahre Linienzahl (4) vorgegeben", "trennt in dieser Zelle 93,5 %", "mindestens 85,0 % und größter Reuegrad 9", "ESPRIT mit MDL-Zählung ist die Variante ohne")
    has(APP, "MUSIC und ESPRIT bekommen die wahre Linienzahl vorgegeben")
    c = v("C", n=64, snr=10.0)
    assert c["ESPRIT (p = 2)"] / c["Spitze (Rechteck, 16-fach, Parabel)"] - 1 < math.sqrt(1 / (2 * 300))
    has(README, "liegt innerhalb der Stichprobenstreuung (300 Läufe, Standardfehler eines RMSE etwa 4 %)")
    fw = RES.find("D", bg="weiß", snr=-6.0)["extra"]["false_alarm_nominal"]["Harmonischer F-Test (NW = 4)"]
    assert de(100 * fw, 0) == "14" and "schon bei weißem Rauschen 14 % und bei AR(1)-Hintergrund" in README
    f0 = v("F", signal="Chirp", snr=None)
    assert f0["Synchrosqueezing"] / min(f0.values()) == pytest.approx(39.12, abs=0.01) and "vom Frequenzraster bestimmt" in README
    assert de(math.sqrt(0.25 / 200) * 100, 1) == "3,5" and "bei Aufgabe B mit 200 Läufen 3,5" in README
    n, snr_db = 64, 10.0
    var = 6 * (1.0 / (2 * 10 ** (snr_db / 10))) / (math.pi ** 2 * n * (n ** 2 - 1))                  # A = 1, SNR = A^2 / (2 sigma^2)
    assert math.sqrt(var) * n == pytest.approx(E.crb_bins(n, snr_db), rel=1e-12)
    assert r"\ge 6 \sigma^2 / (\pi^2 A^2 n (n^2 - 1))" in APP and r"12 \sigma^2 / (A^2" not in APP
