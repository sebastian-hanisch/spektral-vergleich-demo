"""Feste Werte der Experimente (Tabelle und Zusammenfassungen); jede Änderung an den Verfahren oder Szenarien muss hier bewusst nachgezogen werden."""

import pytest

import spv_constants as C
import spv_evaluation as ev
import spv_results as RES

# {Aufgabe: {Verfahren: (mittlerer Reuegrad, größter Reuegrad, Siege)}} über alle Zellen der Aufgabe, gleich gewichtet
GOLDEN_OVERALL = {
    "A": {"Hann-Periodogramm": (7.738, 28.314, 1), "Welch (Hann, 50 %, n/4)": (3.807, 8.845, 3), "Multitaper (NW = 4)": (4.180, 7.957, 1), "AR (Burg, BIC)": (1.123, 1.907, 16)},
    "B": {"Hann-Periodogramm": (59.0, 87.5, 1), "Multitaper (NW = 2)": (99.25, 100.0, 0), "AR (Burg, p = 16)": (18.5, 80.0, 4), "AR (Burg, BIC)": (44.25, 99.5, 1), "MUSIC (p = 4)": (5.938, 44.0, 6),
          "ESPRIT (p = 4)": (0.0, 0.0, 8), "ESPRIT, Linienzahl per MDL": (4.875, 9.0, 0)},
    "C": {"Spitze (Rechteck, 16-fach, Parabel)": (1.097, 1.493, 4), "Jacobsen (3 Bins)": (1.506, 1.604, 0), "Candan (3 Bins)": (1.507, 1.604, 0), "Hann-Spitze (16-fach, Parabel)": (1.496, 1.624, 0),
          "MUSIC (p = 2)": (1.167, 1.374, 0), "ESPRIT (p = 2)": (1.095, 1.374, 2), "AR-Spitze (Burg, p = 8)": (10.170, 22.366, 0)},
    "D": {"Fisher-g (weißer Hintergrund)": (49.267, 100.0, 1), "Fisher-g nach AR(2)-Vorfilterung": (0.0, 0.0, 10), "Harmonischer F-Test (NW = 4)": (11.467, 34.0, 2)},
    "E": {"regelmäßig": (26.611, 73.667, 3), "Jitter ±0,25": (2.278, 7.333, 3), "zufällig": (3.778, 19.0, 3), "Nachtlücken": (35.222, 69.667, 0), C.E_INTERP: (28.0, 72.333, 0)},
    "F": {"Spektrogramm (Gauß)": (6.294, 38.965, 0), "Neu zugeordnet": (1.585, 2.544, 4), "Synchrosqueezing": (11.109, 39.119, 0), "CWT-Kamm (Morlet)": (3.801, 15.451, 0), "Phasenableitung (21 geglättet)": (2.595, 6.830, 4)},
    "G": {"Wavelet-SURE (D8, weich)": (1.0, 1.0, 4), "Wavelet universal (hart)": (1.255, 1.392, 0), "Fourier universal (hart)": (3.002, 3.413, 0)},
}

# Mittel der Kennzahl über alle Zellen, in denen das Verfahren vorkommt
GOLDEN_MEANS = {
    "A": {"Hann-Periodogramm": 6.520, "Welch (Hann, 50 %, n/4)": 6.263, "Multitaper (NW = 4)": 8.271, "AR (Burg, BIC)": 2.464},
    "B": {"Hann-Periodogramm": 40.25, "Multitaper (NW = 2)": 0.0, "AR (Burg, p = 16)": 80.75, "AR (Burg, BIC)": 55.0, "MUSIC (p = 4)": 93.312, "ESPRIT (p = 4)": 99.25, "ESPRIT, Linienzahl per MDL": 94.375},
    "C": {"Spitze (Rechteck, 16-fach, Parabel)": 1.127, "Jacobsen (3 Bins)": 1.547, "Candan (3 Bins)": 1.547, "Hann-Spitze (16-fach, Parabel)": 1.535, "MUSIC (p = 2)": 1.197, "ESPRIT (p = 2)": 1.122, "AR-Spitze (Burg, p = 8)": 10.672},
    "D": {"Fisher-g (weißer Hintergrund)": 32.433, "Fisher-g nach AR(2)-Vorfilterung": 81.7, "Harmonischer F-Test (NW = 4)": 70.233},
    "E": {"regelmäßig": 54.944, "Jitter ±0,25": 79.278, "zufällig": 77.778, "Nachtlücken": 46.333, C.E_INTERP: 31.444},
    "F": {"Spektrogramm (Gauß)": 0.00456, "Neu zugeordnet": 0.00449, "Synchrosqueezing": 0.03103, "CWT-Kamm (Morlet)": 0.0072, "Phasenableitung (21 geglättet)": 0.00769},
    "G": {"Wavelet-SURE (D8, weich)": 0.275, "Wavelet universal (hart)": 0.356, "Fourier universal (hart)": 0.894},
}


@pytest.mark.parametrize("task", C.TASKS)
def test_overall_summary_is_golden(task):
    got = ev.overall()[task]
    assert set(got) == set(GOLDEN_OVERALL[task])
    for k, (mean, mx, wins) in GOLDEN_OVERALL[task].items():
        assert got[k]["mean"] == pytest.approx(mean, abs=1e-3) and got[k]["max"] == pytest.approx(mx, abs=1e-3) and got[k]["wins"] == wins, (task, k)


@pytest.mark.parametrize("task", C.TASKS)
def test_means_per_method_are_golden(task):
    cells = RES.cells(task)
    for k, want in GOLDEN_MEANS[task].items():
        vals = [c["values"][k] for c in cells if k in c["values"]]
        got = sum(vals) / len(vals)
        assert got == pytest.approx(want, abs=1e-3 if want >= 0.1 else 1e-4), (task, k)


def test_selected_cells_are_golden():
    a = RES.find("A", bg="Resonanz", n=256)
    assert a["values"] == pytest.approx({"Hann-Periodogramm": 6.09, "Welch (Hann, 50 %, n/4)": 2.24, "Multitaper (NW = 4)": 2.14, "AR (Burg, BIC)": 0.72}, abs=0.01)
    assert RES.find("C", n=256, snr=10.0)["extra"]["crb"] == pytest.approx(0.011, abs=0.0005)
    d = RES.find("D", bg="weiß", snr=-12.0)["extra"]
    assert d["thresholds"] == pytest.approx([0.1074, 0.0998, 16.9601], abs=1e-3) and d["nominal_thresholds"] == pytest.approx([0.1086, 13.7706], abs=1e-3)
    e = RES.find("E", line=0.2, snr=0.0)["extra"]["thresholds"]
    assert set(e) == set(C.METHODS["E"]) and e["regelmäßig"] > 0
    assert RES.find("F", signal="Chirp", snr=None)["values"]["Phasenableitung (21 geglättet)"] == pytest.approx(1e-5, abs=1e-5)
    g = RES.find("G", signal="Blocks")
    assert g["values"] == pytest.approx({"Wavelet-SURE (D8, weich)": 0.37, "Wavelet universal (hart)": 0.51, "Fourier universal (hart)": 1.25}, abs=0.01)


def test_cell_and_value_counts_are_golden():
    assert len(RES.specs()) == 55 and len(RES.load()["cells"]) == 63
    assert sum(len(c["values"]) for c in RES.load()["cells"]) == 21 * 4 + 8 * 7 + 6 * 7 + 10 * 3 + (3 * 5 + 3 * 4) + 8 * 5 + 4 * 3
