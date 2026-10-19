"""Welches Verfahren, gemessen woran? Der Sieger steht im Szenarienmix – der Reuegrad zeigt, was robust ist - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 16 (letztes) der Reihe „Spektral- und Wavelet-Analyse“. Die Verfahren der Reihe an sieben Aufgaben mit bekannter Wahrheit, verglichen mit Reuegrad (Regret) und Szenarienmix; die Tabelle der Experimente ist vorgerechnet
(spv_results.json), die Live-Ansicht rechnet eine Realisierung. Aufbau und Regeln: DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import pandas as pd
import streamlit as st

import spv_advisor as ADV
import spv_constants as C
import spv_evaluation as ev
import spv_regret as R
import spv_results as RES
from spv_evaluation import Settings, analyse, de
from spv_pdf_export import build_pdf
from spv_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from spv_visualization import (
    build_advice,
    build_d_stats,
    build_heatmap,
    build_live,
    build_mix,
    build_sweep,
    build_values,
)

st.set_page_config(page_title="Verfahrensvergleich – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _mix_curves():
    return ev.mix_curves()


def _table(rows):
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


st.title("⚖️ Welches Verfahren, gemessen woran?")
st.markdown(
    """
Jedes Verfahren dieser Reihe hat eine Aufgabe, in der es glänzt. Der **Vergleich** entscheidet aber nicht allein die Rechnung, sondern die **Wahl der Szenarien**: Das AR-Spektrum gewinnt jedes Szenario mit autoregressivem
Hintergrund und verliert in fünf von sechs Szenarien mit Kerbe oder Bandpass die Siege (im mittleren Reuegrad bleibt es vorn); ein Test auf weißes Rauschen schlägt in gefärbtem Rauschen fast immer an. Hier laufen die Verfahren der Reihe an **sieben Aufgaben mit
bekannter Wahrheit**. Als Maß dient der **Reuegrad** (englisch *regret*): wie viel schlechter ein Verfahren in einem Szenario ist als das beste dort. Sein Mittel hängt am **Szenarienmix**, sein Maximum zeigt, was **robust** ist.

Dies ist die sechzehnte und letzte Demo der Reihe „Spektral- und Wavelet-Analyse“. Neue Verfahren kommen nicht vor; alles ist **simuliert**, die Wahrheit ist bekannt. Weiter unten: **🔬 Experimente**,
**🧭 Entscheidungshilfe**, **🚧 Wo die Aussagen enden** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So wird verglichen", expanded=True):
    st.markdown(
        """
1. **Aufgabe und Szenario:** Eine von sieben Aufgaben (Spektrum schätzen, zwei Linien trennen, eine Linie genau treffen, eine Linie im Rauschen finden, unregelmäßige Messzeiten, zeitabhängige Frequenz, Entrauschen); ein Szenario legt Hintergrund, Länge und Rauschabstand fest.
2. **Kennzahl gegen die Wahrheit:** Fehler (Spektrum in dB, Frequenz als Vielfaches der Cramér-Rao-Schranke, mittlere quadratische Abweichung) oder Erfolgsanteil (aufgelöst, erkannt) über viele Läufe mit festen Zufallsströmen.
3. **Reuegrad:** Bei Fehlern der Quotient aus eigenem und bestem Fehler im Szenario (1 = bester), bei Erfolgsanteilen die Punkte hinter dem besten.
4. **Mix:** Mittlerer Reuegrad mit Gewichten je Szenario; größter Reuegrad (Minimax) und Zahl der Siege daneben.
5. **Entscheidungshilfe:** Typ bekannt: der Sieger des nächstgelegenen Szenarios; Typ unbekannt: das Verfahren mit dem kleinsten größten Reuegrad. Nur über diese Szenarien, nie allgemein.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()
ss = st.session_state

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Aufgabe**")
    task = st.selectbox("Aufgabe", list(C.TASKS), format_func=lambda t: C.TASK_LABELS[t], key="task_select",
                        help="Sieben Aufgaben mit bekannter Wahrheit; die Live-Ansicht rechnet eine Realisierung, die Experimente unten fassen viele Läufe zusammen.")
    st.markdown("**Szenario**")
    if task == "A":
        a_bg = st.selectbox("Hintergrund", list(C.BACKGROUNDS_A), index=C.BACKGROUNDS_A.index(ss["a_bg_kept"]),
                            help="Wahres Spektrum: autoregressive Hintergründe (Resonanz, Steil, zwei scharfe Resonanzen, weiß) oder Kerbe und Bandpass (kein autoregressives Spektrum).")
        a_n = st.select_slider("Länge n", options=list(C.N_A), value=ss["a_n_kept"], help="Zahl der Abtastwerte.")
        ss["a_bg_kept"], ss["a_n_kept"] = a_bg, int(a_n)
    else:
        a_bg, a_n = ss["a_bg_kept"], int(ss["a_n_kept"])
    if task == "B":
        b_delta = st.slider("Abstand der Linien (Bins)", C.B_DELTA_MIN, C.B_DELTA_MAX, float(ss["b_delta_kept"]), step=C.B_DELTA_STEP, help="Abstand in Bins von n = 64 (ein Bin ist 1/64).")
        b_snr = st.slider("Rauschabstand je Linie (dB)", C.B_SNR_MIN, C.B_SNR_MAX, float(ss["b_snr_kept"]), step=C.B_SNR_STEP, help="Linie gegen Rauschvarianz 1, je Abtastwert.")
        ss["b_delta_kept"], ss["b_snr_kept"] = float(b_delta), float(b_snr)
    else:
        b_delta, b_snr = float(ss["b_delta_kept"]), float(ss["b_snr_kept"])
    if task == "C":
        c_n = st.select_slider("Länge n", options=list(C.C_NS), value=ss["c_n_kept"], help="Zahl der Abtastwerte.")
        c_snr = st.slider("Rauschabstand (dB)", C.C_SNR_MIN, C.C_SNR_MAX, float(ss["c_snr_kept"]), step=1.0, help="Linie gegen Rauschvarianz 1.")
        ss["c_n_kept"], ss["c_snr_kept"] = int(c_n), float(c_snr)
    else:
        c_n, c_snr = int(ss["c_n_kept"]), float(ss["c_snr_kept"])
    if task == "D":
        d_bg = st.radio("Hintergrund", list(C.D_BGS), index=C.D_BGS.index(ss["d_bg_kept"]), horizontal=True, help="Weißes Rauschen oder AR(1) mit Pol 0,8 (n = 128, Linie bei 0,2).")
        d_snr = st.slider("Rauschabstand der Linie (dB)", C.D_SNR_MIN, C.D_SNR_MAX, float(ss["d_snr_kept"]), step=1.0, help="Linie gegen die Rauschvarianz 1 (bei AR(1) gegen die Varianz der Zufuhr).")
        ss["d_bg_kept"], ss["d_snr_kept"] = d_bg, float(d_snr)
    else:
        d_bg, d_snr = ss["d_bg_kept"], float(ss["d_snr_kept"])
    if task == "E":
        e_line = st.radio("Lage der Linie", list(C.E_LINE_CHOICES), index=C.E_LINE_CHOICES.index(ss["e_line_kept"]), horizontal=True, format_func=lambda v: f"{v:g}".replace(".", ","),
                          help="In Zyklen je mittlerem Abstand der Messzeiten; die mittlere Nyquist-Frequenz ist 0,5. 1,3 liegt darüber, 0,2 darunter.")
        e_snr = st.slider("Rauschabstand (dB)", C.E_SNR_MIN, C.E_SNR_MAX, float(ss["e_snr_kept"]), step=1.0, help="Linie gegen Rauschvarianz 1.")
        ss["e_line_kept"], ss["e_snr_kept"] = float(e_line), float(e_snr)
    else:
        e_line, e_snr = float(ss["e_line_kept"]), float(ss["e_snr_kept"])
    if task == "F":
        f_sig = st.radio("Signal", list(C.F_SIGNALS), index=C.F_SIGNALS.index(ss["f_sig_kept"]), horizontal=True, help="Linearer Chirp (0,05 bis 0,20) oder sinusförmige Frequenzmodulation um 0,12.")
        f_snr = st.slider("Rauschabstand (dB)", C.F_SNR_MIN, C.F_SNR_MAX, float(ss["f_snr_kept"]), step=1.0, help="Signal gegen weißes Rauschen (n = 512).")
        ss["f_sig_kept"], ss["f_snr_kept"] = f_sig, float(f_snr)
    else:
        f_sig, f_snr = ss["f_sig_kept"], float(ss["f_snr_kept"])
    if task == "G":
        g_sig = st.radio("Signal", list(C.G_SIGNALS), index=C.G_SIGNALS.index(ss["g_sig_kept"]), horizontal=True, help="Vier Testsignale (n = 1024) mit Rauschabstand 7 gegen die Standardabweichung 1.")
        ss["g_sig_kept"] = g_sig
    else:
        g_sig = ss["g_sig_kept"]
    st.markdown("**Zufall**")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neu würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"task_select": task, "a_bg_kept": a_bg, "a_n_kept": int(a_n), "b_delta_kept": float(b_delta), "b_snr_kept": float(b_snr), "c_n_kept": int(c_n), "c_snr_kept": float(c_snr), "d_bg_kept": d_bg,
                   "d_snr_kept": float(d_snr), "e_line_kept": float(e_line), "e_snr_kept": float(e_snr), "f_sig_kept": f_sig, "f_snr_kept": float(f_snr), "g_sig_kept": g_sig, "mix_input": int(ss["mix_input"]),
                   "known_select": ss["known_select"], "seed_input": int(seed)})

settings = Settings(task, a_bg, int(a_n), float(b_delta), float(b_snr), int(c_n), float(c_snr), d_bg, float(d_snr), float(e_line), float(e_snr), f_sig, float(f_snr), g_sig, int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
kind_error = C.TASK_KIND[task] == C.KIND_ERROR
dg = ev.LIVE_DIGITS[task]
best, worst, best_value, worst_value = ev.extremes(a)
near, exact = ADV.nearest_cell(task, settings.params)
metrics = [("Bestes Verfahren" if kind_error else "Erfolgreichstes Verfahren", ev.names_label(best), de(best_value, dg), "Kleinster Fehler bzw. größter Erfolg in dieser einen Realisierung; die Experimente unten fassen viele Läufe zusammen."),
           ("Schlechtestes Verfahren", ev.names_label(worst), de(worst_value, dg), "Größter Fehler bzw. kleinster Erfolg in dieser Realisierung."),
           ("Nächstes Tabellenszenario", R.cell_label(task, near["params"]), "genau" if exact else "nächstgelegen", "Szenario der vorgerechneten Tabelle, das dem gewählten am nächsten liegt (Grundlage der Entscheidungshilfe).")]
if task == "C":
    metrics.append(("Cramér-Rao-Schranke", de(a.extra["crb"], 3), "Bins (Standardabweichung)", "Untere Schranke für die Standardabweichung einer erwartungstreuen Frequenzschätzung (Bins)."))
if task == "G":
    metrics.append(("Verrauscht", de(a.refs["verrauscht"], 3), "mittlere quadratische Abweichung", "Abweichung der verrauschten Reihe vom wahren Signal (Rauschvarianz 1)."))
for col, (label, value, delta, help_text) in zip(st.columns(len(metrics)), metrics):
    col.metric(label, value, delta=delta, delta_color="off", help=help_text)

st.plotly_chart(build_live(a), width="stretch", key="live")
CAPTIONS = {
    "A": "Geschätztes Spektrum (dB) je Verfahren gegen das wahre (rot gestrichelt) in einer Realisierung; der Fehler ist der quadratische Mittelwert der Differenz der Logspektren in dB über die inneren Bins.",
    "B": "Spektren und MUSIC-Pseudospektrum in dB gegen ihr Maximum, rot gestrichelt die wahren Linien, Kreuze die ESPRIT-Schätzungen; MUSIC und ESPRIT bekommen die wahre Linienzahl vorgegeben. „Aufgelöst“ heißt: Maximum an beiden Linienlagen und Einbruch auf unter die Hälfte dazwischen.",
    "C": "Hann-Periodogramm in der Umgebung der wahren Linie (rot gestrichelt); die Balken zeigen den Fehler der Frequenzschätzer in Bins gegen die Cramér-Rao-Schranke.",
    "D": "Periodogramm der Reihe, rot gestrichelt die wahre Linie. „Erkannt“ gilt gegen die auf 5 % Falschalarm kalibrierte Schwelle (aus Rauschläufen); die Rauten zeigen die Schwelle aus der Theorie.",
    "E": "Lomb-Scargle-Leistung je Messzeit-Art, rot gestrichelt die wahre Linie. Erkannt: Maximum über der 95-%-Schwelle aus Rauschläufen und innerhalb von 0,05 der wahren Lage.",
    "F": "Geschätzte Frequenz über der Zeit je Verfahren gegen die wahre (rot gestrichelt); der Fehler zählt im orangen Fenster ohne die Ränder.",
    "G": "Wahres Signal (rot gestrichelt), verrauschte Reihe (grau) und die drei Schätzungen.",
}
st.caption(CAPTIONS[task])
st.plotly_chart(build_values(a), width="stretch", key="values")
if task == "D":
    st.plotly_chart(build_d_stats(a), width="stretch", key="d_stats")
    nd = a.extra["nominal_decision"]
    st.caption(f"Mit der Theorieschwelle (nominell 5 %) hätte der Fisher-Test {'angeschlagen' if nd[0] else 'nicht angeschlagen'} und der F-Test {'angeschlagen' if nd[2] else 'nicht angeschlagen'}; "
               f"bei {a.settings.d_bg} schlägt der Fisher-Test auf reinem Rauschen in {de(100 * a.extra['false_alarm_nominal']['Fisher-g (weißer Hintergrund)'], 1)} % der Läufe an.")

if st.button("📄 PDF erstellen", key="pdf_make"):
    st.session_state["pdf_bytes"] = build_pdf(a)
    st.session_state["pdf_for"] = settings
if st.session_state.get("pdf_for") == settings and "pdf_bytes" in st.session_state:
    st.download_button("⬇️ PDF herunterladen", st.session_state["pdf_bytes"], file_name="verfahrensvergleich.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Welches Verfahren ist das richtige – und woran messen wir das?")
kind_v, text = ev.verdict_main(a)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experimente")
st.caption("Vorgerechnete Tabelle (`spv_results.json`, feste Zufallsströme, unabhängig von den Reglern; die Tests rechnen Stichproben nach). Zellen sind Szenarien; „wahr“ ist der Wert, mit dem die Reihen gebaut sind.")

# Spektrum und Szenarienmix
st.markdown("#### Spektrum schätzen und der Szenarienmix")
sp = ev.exp_spectrum(int(ss["mix_input"]))
st.caption(f"21 Szenarien: sieben Hintergründe (fünf autoregressiv oder weiß, dazu Kerbe und Bandpass) bei n = 64, 256 und 1024, je {C.RUNS_A} Läufe; Fehler = quadratischer Mittelwert der Differenz der Logspektren in dB (innere Bins).")
_table(ev.table_rows("A", sp["cells"], with_refs=True))
st.slider("Anteil der autoregressiven Hintergründe im Mix (%)", C.MIX_MIN, C.MIX_MAX, key="mix_input",
          help="Gewicht der Szenarien mit autoregressivem oder weißem Hintergrund gegen Kerbe und Bandpass; 100 % heißt: nur Heimvorteil für das AR-Spektrum.")
st.caption(f"Zusammenfassung über den Mix {de(100 * sp['share'], 0)} % autoregressiv ({sp['n_ar']} Szenarien) und {de(100 - 100 * sp['share'], 0)} % Kerbe oder Bandpass ({sp['n_non_ar']} Szenarien); gleich gewichtet daneben.")
_table(ev.summary_rows("A", sp["summary"]))
st.plotly_chart(build_mix(_mix_curves(), sp["share"]), width="stretch", key="exp_mix")
st.plotly_chart(build_heatmap("A", sp["cells"]), width="stretch", key="exp_heat_a")
st.caption("Reuegrad je Verfahren und Szenario (Farbe und Zahl: Fehler durch besten Fehler im Szenario).")
st.plotly_chart(build_sweep("A", sp["cells"], "bg", list(C.BACKGROUNDS_A), "n", list(C.N_A)), width="stretch", key="exp_sweep_a")
kind_v, text = ev.verdict_spectrum(sp)
getattr(st, kind_v)(text)

# Linien
st.markdown("#### Linien trennen und genau treffen")
ln = ev.exp_lines()
st.caption(f"Zwei gleich starke Linien bei 0,2 und im Abstand Δ Bins (n = {C.N_B}), {C.RUNS_B} Läufe je Szenario: Anteil der Läufe, in denen das Verfahren beide Linien trennt (in %).")
_table(ev.table_rows("B", ln["b"]))
st.plotly_chart(build_sweep("B", ln["b"], "delta", list(C.B_DELTAS), "snr", list(C.B_SNRS)), width="stretch", key="exp_sweep_b")
st.caption(f"Eine Linie mit zufälliger Frequenz, {C.RUNS_C} Läufe je Szenario: Wurzel des mittleren quadratischen Fehlers der Frequenz, geteilt durch die Cramér-Rao-Schranke (1 = Cramér-Rao-Schranke für erwartungstreue Schätzer; Werte etwas darunter liegen im Stichprobenrauschen). MUSIC und ESPRIT bekommen die wahre Linienzahl vorgegeben.")
_table(ev.table_rows("C", ln["c"]))
st.plotly_chart(build_sweep("C", ln["c"], "snr", list(C.C_SNRS), "n", list(C.C_NS)), width="stretch", key="exp_sweep_c")
st.plotly_chart(build_heatmap("B", ln["b"]), width="stretch", key="exp_heat_b")
st.caption("Reuegrad der Auflösung in Prozentpunkten hinter dem besten Verfahren des Szenarios.")
kind_v, text = ev.verdict_lines(ln)
getattr(st, kind_v)(text)

# Erkennen
st.markdown("#### Eine Linie im Rauschen finden")
dt = ev.exp_detection()
st.caption(f"Linie bei {C.D_LINE:g} in weißem und in AR(1)-Rauschen (n = {C.N_D}), {C.RUNS_D} Läufe je Szenario, Schwellen aus {C.RUNS_D_NULL} Rauschläufen auf wirklich 5 % Falschalarm kalibriert: Anteil erkannter Linien (in %). "
           "Gepunktet die Rate der Tests mit Theorieschwelle (nominell 5 %; der F-Test liegt schon bei weißem Rauschen darüber).")
_table(ev.table_rows("D", dt["cells"]))
st.plotly_chart(build_sweep("D", dt["cells"], "snr", list(C.D_SNRS), "bg", list(C.D_BGS), nominal=True), width="stretch", key="exp_sweep_d")
kind_v, text = ev.verdict_detection(dt)
getattr(st, kind_v)(text)

# Unregelmäßig und Zeit-Frequenz
st.markdown("#### Unregelmäßige Messzeiten und zeitabhängige Frequenz")
ir = ev.exp_irregular()
st.caption(f"Lomb-Scargle bei n = {C.N_E} Messzeiten, {C.RUNS_E} Läufe je Szenario: Anteil erkannter Linien (in %) je Art der Messzeiten; das Interpolieren aufs Gitter nur unter der Nyquist-Grenze.")
_table(ev.table_rows("E", ir["e"]))
st.plotly_chart(build_sweep("E", ir["e"], "snr", sorted(C.E_SNRS), "line", list(C.E_LINES)), width="stretch", key="exp_sweep_e")
st.caption(f"Frequenz über der Zeit (n = {C.N_F}), {C.RUNS_F} Läufe je verrauschtes Szenario: quadratischer Mittelwert des Frequenzfehlers im Fenster ohne die Ränder (Zyklen je Abtastwert).")
_table(ev.table_rows("F", ir["f"]))
st.plotly_chart(build_sweep("F", ir["f"], "snr", list(C.F_SNRS), "signal", list(C.F_SIGNALS)), width="stretch", key="exp_sweep_f")
kind_v, text = ev.verdict_irregular(ir)
getattr(st, kind_v)(text)

# Entrauschen
st.markdown("#### Entrauschen")
dn = ev.exp_denoise()
st.caption(f"Vier Testsignale (n = {C.N_G}, Rauschabstand {C.SNR_G:g}), {C.RUNS_G} Läufe je Signal: mittlere quadratische Abweichung vom wahren Signal; rechts die Bezüge (verrauscht, beste Fourier-Schwelle als Orakel, bester gleitender Mittelwert als Orakel).")
_table(ev.table_rows("G", dn["cells"], with_refs=True))
st.plotly_chart(build_sweep("G", dn["cells"], "signal", list(C.G_SIGNALS)), width="stretch", key="exp_sweep_g")
kind_v, text = ev.verdict_denoise(dn)
getattr(st, kind_v)(text)

st.markdown("---")

# --- Entscheidungshilfe --------------------------------------------------------------------------------------------------------------------

st.subheader("🧭 Entscheidungshilfe")
st.caption("Für die gewählte Aufgabe und das gewählte Szenario aus der Seitenleiste. Die Empfehlung gilt nur für die gemessenen Szenarien dieser Tabelle, nie allgemein.")
known_choice = st.radio("Kennen Sie den Typ des Szenarios (Hintergrund, Signal)?", list(C.KNOWN_CHOICES), key="known_select", horizontal=True,
                        help="Bekannt: der Sieger des nächstgelegenen Szenarios. Unbekannt: das Verfahren mit dem kleinsten größten Reuegrad über alle Szenarien der Aufgabe.")
adv = ADV.advise(task, settings.params, known_choice == C.KNOWN_YES)
st.success(f"Empfehlung für {C.TASK_LABELS[task]}: **{adv.recommendation}** – {adv.basis}.")
if adv.note:
    st.caption(adv.note)
_table(ev.advice_rows(task, adv))
st.plotly_chart(build_advice(adv.rows, task), width="stretch", key="advice")
st.caption("Balken: Reuegrad im nächstgelegenen Szenario; Rauten: größter Reuegrad über alle Szenarien der Aufgabe – je kleiner, desto robuster.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Aussagen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Szenarien sind repräsentativ** | Sie sind es nicht: Sie sind gewählt. Mit anderen Hintergründen, Längen und Rauschabständen kippt der Sieger; der Reuegrad gilt nur für diese Zellen. | eigene Szenarien aus den Anwendungsdaten, Sensitivitätsanalyse des Mix |
| **Der Mix ist bekannt** | Die Gewichte der Szenarien bestimmen den mittleren Reuegrad; der größte Reuegrad (Minimax) hängt für Anteile zwischen 0 und 100 % nicht davon ab (an den Rändern fallen Zellen mit Gewicht null heraus), wohl aber von der Wahl der Zellen. | Minimax-Auswahl, mehrere Mischungen vergleichen |
| **Ein Verfahren, eine Einstellung** | Jedes Verfahren läuft mit festen Einstellungen (Fenster, Ordnung, Taper; MUSIC und ESPRIT bekommen die wahre Linienzahl vorgegeben, nur ESPRIT mit MDL-Zählung schätzt sie); mit besserer Einstellung kippt mancher Abstand. | Einstellungen je Szenario abstimmen, auf Daten prüfen |
| **Die Wahrheit ist bekannt** | Nur in der Simulation. Mit echten Daten fehlt der Vergleichswert, dann bleiben Schwellen aus Rauschläufen und Kreuzvalidierung. | simulierte Referenz, Rauschläufe |
| **Kalibrierung gilt** | Die Schwellen der Tests sind für das gewählte Rauschen kalibriert; mit anderem Rauschen schlagen sie anders an. | Kalibrierung am eigenen Rauschmodell |
""",
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Reuegrad.** Für Verfahren $m$ im Szenario $s$ mit Fehler $e_{m,s} \ge 0$ ist $\rho_{m,s} = e_{m,s} / \min_{m'} e_{m',s}$ (Quotient zum besten, $\rho \ge 1$). Bei Erfolgsanteilen $r_{m,s}$ in Prozent ist
$\rho_{m,s} = \max_{m'} r_{m',s} - r_{m,s}$ (Prozentpunkte, $\rho \ge 0$).

**Mittel, Maximum, Siege.** Mit Gewichten $w_s \ge 0$, $\sum_s w_s = 1$: $\bar\rho_m = \sum_s w_s \rho_{m,s}$, $\rho^{\max}_m = \max_{s: w_s > 0} \rho_{m,s}$ und der Siegesanteil $\sum_s w_s \mathbf 1[\rho_{m,s} = \min_{m'} \rho_{m',s}]$.
Minimax-Wahl: $\arg\min_m \rho^{\max}_m$.

**Szenarienmix.** Aufgabe A: Anteil $q$ der Szenarien mit autoregressivem oder weißem Hintergrund, verteilt gleich auf diese Zellen; der Rest $1 - q$ gleich auf Kerbe und Bandpass.

**Fehler Spektrum.** $e = \sqrt{\overline{\left(10 \log_{10}(\hat S(f_k)/S(f_k))\right)^2}}$ über die inneren Bins $k$ (vier an jedem Rand abgezogen).

**Cramér-Rao-Schranke.** Für eine Linie der Amplitude $A$ in weißem Rauschen der Varianz $\sigma^2$ mit $n$ Abtastwerten: $\operatorname{var}(\hat f) \ge 6 \sigma^2 / (\pi^2 A^2 n (n^2 - 1))$ (Frequenz in Zyklen je Abtastwert, Signal-Rausch-Verhältnis $A^2/(2\sigma^2)$ je Abtastwert); in Bins mal $n$.

**Kalibrierung.** Die Schwelle eines Tests ist das 95-%-Quantil der Prüfgröße in $N$ Rauschläufen; Erkennungsrate = Anteil der Läufe mit Linie, in denen die Prüfgröße die Schwelle überschreitet.

**Quellen.** Die Verfahren und ihre Formeln stammen aus den Demos 1 bis 15 der Reihe; der Reuegrad ist das Standardmaß der Entscheidungstheorie (Savage 1951) und der Online-Optimierung, hier nur als Vergleichsmaß verwendet.
Die Quellen wurden nicht gelesen; die Formeln sind per Handrechnung, Simulation und Gegenproben geprüft.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Spektren und Wavelets: von Fourier zum Vergleich](https://sebastianhanisch.net/konzepte-spektral-wavelet-analyse.html)."
)
