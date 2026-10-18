"""Konstanten: Aufgaben, Szenarien, Verfahren, Läufe, Regler-Grenzen, Farben und Presets (Kern: spv_scenarios.py, spv_tasks.py, spv_regret.py, spv_advisor.py)."""

TASKS = ("A", "B", "C", "D", "E", "F", "G")
TASK_LABELS = {"A": "A Spektrum schätzen", "B": "B Zwei Linien trennen", "C": "C Eine Linie genau treffen", "D": "D Linie im Rauschen finden", "E": "E Unregelmäßige Messzeiten",
               "F": "F Zeitabhängige Frequenz", "G": "G Entrauschen"}
DEFAULT_TASK = "A"
KIND_ERROR, KIND_RATE = "Fehler", "Erfolgsanteil"
TASK_KIND = {"A": KIND_ERROR, "B": KIND_RATE, "C": KIND_ERROR, "D": KIND_RATE, "E": KIND_RATE, "F": KIND_ERROR, "G": KIND_ERROR}
TASK_UNIT = {"A": "dB", "B": "%", "C": "× Cramér-Rao-Schranke", "D": "%", "E": "%", "F": "Frequenz (Zyklen je Abtastwert)", "G": "mittlere quadratische Abweichung"}
TASK_TRUTH = {"A": "wahres Spektrum", "B": "wahre Linien", "C": "wahre Frequenz", "D": "wahre Linie (Test mit bekanntem Falschalarm)", "E": "wahre Frequenz", "F": "wahre Momentanfrequenz", "G": "wahres Signal"}

# --- Szenarien ----------------------------------------------------------------------------------------------------------------------------

BACKGROUNDS_A = ("Resonanz", "Resonanz + weißes Rauschen", "Steil", "Zwei scharfe Resonanzen", "weiß", "Kerbe", "Bandpass")
AR_TYPE = ("Resonanz", "Resonanz + weißes Rauschen", "Steil", "Zwei scharfe Resonanzen", "weiß")        # der Hintergrund ist autoregressiv (oder weiß): Heimvorteil des AR
NON_AR_TYPE = ("Kerbe", "Bandpass")
POLES = {"weiß": (), "Resonanz": ((0.2, 0.98),), "Resonanz + weißes Rauschen": ((0.2, 0.98),), "Steil": ((0.2, 0.99), (0.0, 0.9), (0.0, 0.9)), "Zwei scharfe Resonanzen": ((0.1, 0.998), (0.3, 0.998))}
EXTRA_DB = {"Resonanz + weißes Rauschen": 10.0}
REF_ORDER = {"Resonanz": 2, "Resonanz + weißes Rauschen": 2, "Steil": 4, "Zwei scharfe Resonanzen": 4, "Kerbe": 8, "Bandpass": 8}       # feste Ordnung des Orakel-Bezugs (wahre Ordnung bei AR)
NOTCH = (1.0, 0.0, 0.97 ** 2)                                                                                   # 1 + 0,9409 z^-2: Nullstellen bei ±0,25 (Radius 0,97)
BAND_LO, BAND_HI, BAND_TAPS, BAND_BETA = 0.1, 0.2, 41, 7.0
N_A = (64, 256, 1024)
RUNS_A = 100
B_DELTAS, B_SNRS, N_B, RUNS_B = (0.75, 1.0, 1.5, 2.5), (20.0, 10.0), 64, 200
B_F1 = 0.2
B_PAD = 16
C_NS, C_SNRS, RUNS_C = (64, 256), (20.0, 10.0, 0.0), 300
D_BGS = ("weiß", "AR(1) 0,8")
D_POLES = {"weiß": (), "AR(1) 0,8": ((0.0, 0.8),)}
D_SNRS, N_D, RUNS_D, RUNS_D_NULL, D_LINE = (-12.0, -9.0, -6.0, -3.0, 0.0), 128, 300, 400, 0.2
E_LINES, E_SNRS, N_E, RUNS_E = (1.3, 0.2), (0.0, -3.0, -6.0), 64, 300
E_KINDS = ("regelmäßig", "Jitter ±0,25", "zufällig", "Nachtlücken")
E_INTERP = "zufällig, linear interpoliert + Hann-Periodogramm"
F_SIGNALS, F_SNRS, N_F, RUNS_F = ("Chirp", "Sinus-FM"), (None, 20.0, 10.0, 0.0), 512, 20
CHIRP_F0, CHIRP_F1 = 0.05, 0.20
SINFM_FC, SINFM_DF, SINFM_FM = 0.12, 0.04, 0.01
F_SIGMA = 12.0
G_SIGNALS, N_G, SNR_G, RUNS_G = ("Blocks", "Bumps", "HeaviSine", "Doppler"), 1024, 7.0, 40
SEED_MAX, DEFAULT_SEED = 999999, 47

# --- Verfahren ----------------------------------------------------------------------------------------------------------------------------

METHODS = {
    "A": ("Hann-Periodogramm", "Welch (Hann, 50 %, n/4)", "Multitaper (NW = 4)", "AR (Burg, BIC)"),
    "B": ("Hann-Periodogramm", "Multitaper (NW = 2)", "AR (Burg, p = 16)", "AR (Burg, BIC)", "MUSIC (p = 4)", "ESPRIT (p = 4)", "ESPRIT, Linienzahl per MDL"),
    "C": ("Spitze (Rechteck, 16-fach, Parabel)", "Jacobsen (3 Bins)", "Candan (3 Bins)", "Hann-Spitze (16-fach, Parabel)", "MUSIC (p = 2)", "ESPRIT (p = 2)", "AR-Spitze (Burg, p = 8)"),
    "D": ("Fisher-g (weißer Hintergrund)", "Fisher-g nach AR(2)-Vorfilterung", "Harmonischer F-Test (NW = 4)"),
    "E": E_KINDS + (E_INTERP,),
    "F": ("Spektrogramm (Gauß)", "Neu zugeordnet", "Synchrosqueezing", "CWT-Kamm (Morlet)", "Phasenableitung (21 geglättet)"),
    "G": ("Wavelet-SURE (D8, weich)", "Wavelet universal (hart)", "Fourier universal (hart)"),
}
REFERENCES = {"A": ("AR (Burg, feste Ordnung)",), "G": ("verrauscht", "Fourier beste Schwelle (Orakel)", "Gleitender Mittelwert (Orakel, beste Breite)")}        # Bezüge: zählen nicht als Verfahren
BIAS_TASKS = ("A", "B", "C", "D", "E", "F", "G")

COLOR_TRUE = "#B03A2E"
COLOR_EST = "#4C9F70"
COLOR_NOISY = "rgba(120,120,120,0.7)"
COLOR_AUX = "#E8833A"
COLOR_BLUE = "rgba(76,120,168,0.95)"
PALETTE = ("#4C78A8", "#4C9F70", "#E8833A", "#9467BD", "#B03A2E", "#17BECF", "#8C564B")

# --- Regler der Live-Ansicht (je Aufgabe) -------------------------------------------------------------------------------------------------

B_DELTA_MIN, B_DELTA_MAX, B_DELTA_STEP, DEFAULT_B_DELTA = 0.5, 4.0, 0.25, 1.0
B_SNR_MIN, B_SNR_MAX, B_SNR_STEP, DEFAULT_B_SNR = 0.0, 30.0, 1.0, 10.0
C_SNR_MIN, C_SNR_MAX, DEFAULT_C_SNR = -5.0, 30.0, 10.0
D_SNR_MIN, D_SNR_MAX, DEFAULT_D_SNR = -15.0, 3.0, -6.0
E_SNR_MIN, E_SNR_MAX, DEFAULT_E_SNR = -9.0, 3.0, 0.0
F_SNR_MIN, F_SNR_MAX, DEFAULT_F_SNR = -5.0, 40.0, 10.0
DEFAULTS = {"a_bg": "Resonanz", "a_n": 256, "b_delta": DEFAULT_B_DELTA, "b_snr": DEFAULT_B_SNR, "c_n": 64, "c_snr": DEFAULT_C_SNR, "d_bg": "AR(1) 0,8", "d_snr": DEFAULT_D_SNR, "e_line": 1.3,
            "e_snr": DEFAULT_E_SNR, "f_sig": "Chirp", "f_snr": DEFAULT_F_SNR, "g_sig": "Blocks"}

# --- Entscheidungshilfe -------------------------------------------------------------------------------------------------------------------

KNOWN_YES, KNOWN_NO = "Ja, ich kenne den Typ", "Nein, er ist unbekannt"
KNOWN_CHOICES = (KNOWN_YES, KNOWN_NO)
MIX_MIN, MIX_MAX, DEFAULT_MIX = 0, 100, 71            # Anteil der autoregressiven Hintergründe im Szenarienmix der Aufgabe A (Prozent)
E_LINE_CHOICES = E_LINES

# --- Presets ------------------------------------------------------------------------------------------------------------------------------

_BASE = {"task": "A", **DEFAULTS, "seed": DEFAULT_SEED, "mix": DEFAULT_MIX, "known": KNOWN_YES}
PRESETS = {
    "Heimvorteil: Das AR-Spektrum siegt auf AR-Hintergründen": {**_BASE, "task": "A", "a_bg": "Steil", "a_n": 256},
    "Kerbe und Bandpass: Welch statt AR": {**_BASE, "task": "A", "a_bg": "Kerbe", "a_n": 1024},
    "Zwei Linien im Abstand eines Bins: ESPRIT, MUSIC, AR": {**_BASE, "task": "B", "b_delta": 1.0, "b_snr": 10.0},
    "Linie im gefärbten Rauschen: der Fisher-Test versagt": {**_BASE, "task": "D", "d_bg": "AR(1) 0,8", "d_snr": -6.0},
    "Unregelmäßige Zeiten: Lomb-Scargle über der Nyquist-Grenze": {**_BASE, "task": "E", "e_line": 1.3, "e_snr": 0.0},
}
PRESET_HELP = {
    "Heimvorteil: Das AR-Spektrum siegt auf AR-Hintergründen": "Steiles autoregressives Spektrum, n = 256: Das AR-Spektrum mit BIC-Ordnung liegt bei etwa 0,8 dB Fehler, Welch bei 2,9 dB, Multitaper bei 4,0 dB und das Hann-Periodogramm bei 6,0 dB; der Hintergrund ist autoregressiv, das Modell passt.",
    "Kerbe und Bandpass: Welch statt AR": "Hintergrund mit Kerbe (Nullstelle), n = 1024: Welch und Multitaper liegen bei etwa 1,6 und 1,7 dB, das AR-Spektrum bei 2,1 dB; die Kerbe ist kein autoregressives Spektrum. Beim Bandpass leckt das Multitaper noch stärker.",
    "Zwei Linien im Abstand eines Bins: ESPRIT, MUSIC, AR": "Zwei gleich starke Linien im Abstand von einem Bin (n = 64, 10 dB je Linie): ESPRIT trennt sie, MUSIC fast immer (beide mit vorgegebener Linienzahl), das AR-Spektrum mit Ordnung 16 in etwa der Hälfte der Läufe, das Hann-Periodogramm und das Multitaper nicht.",
    "Linie im gefärbten Rauschen: der Fisher-Test versagt": "Linie von −6 dB in AR(1)-Rauschen (φ = 0,8), n = 128: Der Fisher-Test auf weißes Rauschen schlägt in fast jedem Rauschlauf an und hat auf wirklich 5 % kalibriert keine Trennschärfe; mit AR(2)-Vorfilterung oder dem F-Test wird die Linie gefunden.",
    "Unregelmäßige Zeiten: Lomb-Scargle über der Nyquist-Grenze": "Linie bei der 2,6fachen mittleren Nyquist-Frequenz (n = 64, 0 dB): Bei zufälligen Messzeiten findet das Lomb-Scargle-Periodogramm sie in allen Läufen, bei regelmäßigen nur in etwa einem Viertel.",
}
