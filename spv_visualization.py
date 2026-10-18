"""Plotly-Figuren. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import spv_constants as C
import spv_regret as R
from spv_evaluation import Analysis, de


def _fixed(fig: go.Figure, height: int = 340, legend_y: float = -0.3, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=legend_y), plot_bgcolor="rgba(0,0,0,0)", **layout)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def color_of(task: str, method: str) -> str:
    """Feste Farbe je Verfahren einer Aufgabe (Reihenfolge in C.METHODS); Bezüge und Unbekanntes grau."""
    methods = C.METHODS[task]
    if method in methods:
        return C.PALETTE[methods.index(method) % len(C.PALETTE)]
    return C.COLOR_NOISY


X_TITLES = {"bg": "Hintergrund", "delta": "Abstand (Bins)", "snr": "Rauschabstand (dB)", "n": "Länge n", "signal": "Signal", "line": "Linie"}


def xlabel(v) -> str:
    """Achsenbeschriftung eines Parameterwerts: None = rauschfrei, Zahlen mit Dezimalkomma und Minuszeichen."""
    if v is None:
        return "rauschfrei"
    if isinstance(v, str):
        return v
    return f"{v:g}".replace(".", ",").replace("-", "−")


# --- Live-Ansicht -------------------------------------------------------------------------------------------------------------------------


def build_values(a: Analysis) -> go.Figure:
    """Kennzahl je Verfahren in dieser Realisierung (Balken), bei Aufgabe C mit der Cramér-Rao-Schranke."""
    t = a.settings.task
    names = list(a.values)
    fig = go.Figure(go.Bar(x=names, y=[a.values[k] for k in names], marker_color=[color_of(t, k) for k in names], hovertemplate="%{x}: %{y:.4g}<extra></extra>"))
    if t == "C":
        fig.add_hline(y=a.extra["crb"], line=dict(color=C.COLOR_TRUE, width=1.4, dash="dash"), annotation_text="Cramér-Rao-Schranke", annotation_position="top left")
    fig.update_yaxes(title_text=a.unit, rangemode="tozero")
    fig.update_xaxes(tickangle=-25)
    return _fixed(fig, height=320, showlegend=False)


def build_live(a: Analysis) -> go.Figure:
    """Kurven der Realisierung: Spektren (A, B, C, D), Lomb-Scargle-Leistung (E), Frequenzverläufe (F) oder Signale (G)."""
    t = a.settings.task
    fig = go.Figure()
    if t == "A":
        f, true = a.truth_curve
        fig.add_trace(go.Scatter(x=f, y=true, mode="lines", line=dict(color=C.COLOR_TRUE, width=2, dash="dash"), name="wahres Spektrum"))
    if t == "F" or t == "G":
        x, y = a.truth_curve
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C.COLOR_TRUE, width=1.8, dash="dash"), name="wahr"))
    if t == "G":
        x, y = a.curves["verrauscht"]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=C.COLOR_NOISY, width=1), name="verrauscht"))
    for k, (x, y) in a.curves.items():
        if t == "G" and k == "verrauscht":
            continue
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color_of(t, k), width=1.4), name=k, hovertemplate=f"{k}: " + "%{y:.4g}<extra></extra>"))
    for pos in a.truth_lines:
        fig.add_vline(x=pos, line=dict(color=C.COLOR_TRUE, width=1.4, dash="dash"))
    if t == "B":
        for k, v in a.points.items():
            fig.add_trace(go.Scatter(x=v, y=[0.0] * len(v), mode="markers", marker=dict(color=color_of("B", "ESPRIT (p = 4)"), size=11, symbol="x"), name="ESPRIT-Schätzungen"))
    if t == "F":
        lo, hi = a.extra["window"]
        fig.add_vrect(x0=lo, x1=hi, fillcolor=C.COLOR_AUX, opacity=0.08, line_width=0)
    xt = {"A": "Frequenz (Zyklen je Abtastwert)", "B": "Frequenz (Bins von n = 64)", "C": "Frequenz (Bins)", "D": "Frequenz (Zyklen je Abtastwert)", "E": "Frequenz (Zyklen je mittlerem Abstand)",
          "F": "Zeit (Abtastwerte)", "G": "Zeit (Abtastwerte)"}[t]
    yt = {"A": "Spektrum (dB)", "B": "dB gegen Maximum", "C": "dB gegen Maximum", "D": "Periodogramm (dB)", "E": "normierte Leistung", "F": "Frequenz (Zyklen je Abtastwert)", "G": "Wert"}[t]
    fig.update_xaxes(title_text=xt)
    fig.update_yaxes(title_text=yt)
    if t in ("B", "C"):
        fig.update_yaxes(range=[-60, 3])
    return _fixed(fig, height=380, legend_y=-0.28)


def build_d_stats(a: Analysis) -> go.Figure:
    """Aufgabe D: Prüfgröße geteilt durch die kalibrierte Schwelle (über 1 heißt erkannt) mit den Theorieschwellen als Rauten."""
    names = list(a.values)
    stats, cal = a.extra["stats"], a.extra["calibrated"]
    ratio = [stats[i] / cal[i] for i in range(3)]
    thr_g, thr_f = a.extra["nominal"]
    fig = go.Figure(go.Bar(x=names, y=ratio, marker_color=[color_of("D", k) for k in names], name="Prüfgröße / kalibrierte Schwelle", hovertemplate="%{x}: %{y:.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[names[0], names[2]], y=[thr_g / cal[0], thr_f / cal[2]], mode="markers", marker=dict(color="black", size=11, symbol="diamond"), name="Theorieschwelle (nominell 5 %)",
                             hovertemplate="%{x}: Schwelle %{y:.3g} (in kalibrierten Schwellen)<extra></extra>"))
    fig.add_hline(y=1.0, line=dict(color=C.COLOR_TRUE, width=1.4, dash="dash"), annotation_text="kalibrierte Schwelle (5 % Falschalarm)", annotation_position="top left")
    fig.update_yaxes(title_text="Prüfgröße / kalibrierte Schwelle", rangemode="tozero")
    fig.update_xaxes(tickangle=-15)
    return _fixed(fig, height=320)


# --- Experimente --------------------------------------------------------------------------------------------------------------------------


def build_mix(curves: dict, share: float) -> go.Figure:
    """Aufgabe A: mittlerer Reuegrad und Siegesanteil gegen den Anteil autoregressiver Hintergründe (Prozent); senkrecht der eingestellte Mix."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("mittlerer Reuegrad (1 = bester)", "Anteil der Siege"), horizontal_spacing=0.12)
    x = curves["shares"]
    for k in C.METHODS["A"]:
        col = color_of("A", k)
        fig.add_trace(go.Scatter(x=x, y=curves["mean"][k], mode="lines", line=dict(color=col, width=2), name=k, legendgroup=k), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=curves["wins"][k], mode="lines", line=dict(color=col, width=2), name=k, legendgroup=k, showlegend=False), row=1, col=2)
    for c in (1, 2):
        fig.add_vline(x=100 * share, line=dict(color="black", width=1, dash="dot"), row=1, col=c)
        fig.update_xaxes(title_text="Anteil autoregressiver Hintergründe (%)", row=1, col=c)
    fig.update_yaxes(type="log", row=1, col=1)
    fig.update_yaxes(range=[0, 1.02], row=1, col=2)
    return _fixed(fig, height=380, legend_y=-0.3)


def build_heatmap(task: str, cells: list) -> go.Figure:
    """Reuegrad je Verfahren (Zeilen) und Szenario (Spalten): Farbe = Reuegrad (bei Fehlern logarithmisch), Beschriftung = Reuegrad."""
    methods = [m for m in C.METHODS[task] if all(m in c["values"] for c in cells)]
    labels = [R.cell_label(task, c["params"]) for c in cells]
    reg = [R.cell_regret(task, c) for c in cells]
    ratio = C.TASK_KIND[task] == C.KIND_ERROR
    z = [[(np.log10(reg[j][m]) if ratio else reg[j][m]) for j in range(len(cells))] for m in methods]
    text = [[de(reg[j][m], 1 if ratio else 0) for j in range(len(cells))] for m in methods]
    hover = [[f"{m}<br>{labels[j]}<br>{de(cells[j]['values'][m], 2)} {C.TASK_UNIT[task]}<br>Reuegrad {text[i][j]}" for j in range(len(cells))] for i, m in enumerate(methods)]
    fig = go.Figure(go.Heatmap(z=z, x=labels, y=methods, text=text, texttemplate="%{text}", customdata=hover, hovertemplate="%{customdata}<extra></extra>", colorscale="YlOrRd", showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-35)
    return _fixed(fig, height=max(220, 60 + 38 * len(methods)) + 60)


def build_sweep(task: str, cells: list, x_key: str, x_values: list, split_key: str | None = None, split_values: list | None = None, nominal: bool = False) -> go.Figure:
    """Kennzahl je Verfahren über einem Parameter (x_key) in einem Feld je Wert von split_key; bei Fehlern logarithmische Achse; nominal=True ergänzt die Raten der Theorieschwellen (Aufgabe D)."""
    splits = split_values if split_key else [None]
    fig = make_subplots(rows=1, cols=len(splits), subplot_titles=[xlabel(v) if split_key else "" for v in splits], horizontal_spacing=0.08)
    methods = C.METHODS[task]
    for ci, sv in enumerate(splits, start=1):
        for k in methods:
            ys, xs = [], []
            for xv in x_values:
                hit = [c for c in cells if c["params"][x_key] == xv and (split_key is None or c["params"][split_key] == sv) and k in c["values"]]
                if hit:
                    xs.append(xlabel(xv))
                    ys.append(hit[0]["values"][k])
            if ys:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color=color_of(task, k), width=2), name=k, legendgroup=k, showlegend=ci == 1,
                                         hovertemplate=f"{k}<br>" + "%{x}: %{y:.4g}<extra></extra>"), row=1, col=ci)
        if nominal:
            for k in ("Fisher-g (weißer Hintergrund)", "Harmonischer F-Test (NW = 4)"):
                xs = [xlabel(xv) for xv in x_values]
                ys = [next(c for c in cells if c["params"][x_key] == xv and c["params"][split_key] == sv)["extra"]["nominal_rate"][k] for xv in x_values]
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color_of(task, k), width=1.4, dash="dot"), name=k + " (Theorieschwelle)", legendgroup=k + "n", showlegend=ci == 1,
                                         hovertemplate=f"{k}, Theorieschwelle<br>" + "%{x}: %{y:.4g}<extra></extra>"), row=1, col=ci)
        fig.update_xaxes(title_text=X_TITLES[x_key], type="category", row=1, col=ci)
    if C.TASK_KIND[task] == C.KIND_ERROR:
        fig.update_yaxes(type="log")
    else:
        fig.update_yaxes(range=[-2, 102])
    fig.update_yaxes(title_text=C.TASK_UNIT[task], row=1, col=1)
    return _fixed(fig, height=420, legend_y=-0.75 if x_key in ("bg", "signal") else -0.3)


def build_advice(rows: tuple, task: str) -> go.Figure:
    """Entscheidungshilfe: Reuegrad im gewählten Szenario (Balken) und größter Reuegrad über alle Szenarien (Rauten)."""
    names = [r["method"] for r in rows]
    ratio = C.TASK_KIND[task] == C.KIND_ERROR
    fig = go.Figure()
    fig.add_trace(go.Bar(x=names, y=[r["regret"] for r in rows], marker_color=[color_of(task, k) for k in names], name="Reuegrad in diesem Szenario", hovertemplate="%{x}: %{y:.3g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=names, y=[r["max_regret"] for r in rows], mode="markers", marker=dict(color="black", size=11, symbol="diamond"), name="größter Reuegrad über alle Szenarien",
                             hovertemplate="%{x}: %{y:.3g}<extra></extra>"))
    fig.update_yaxes(title_text="Reuegrad (1 = bester)" if ratio else "Reuegrad (Prozentpunkte hinter dem besten)", rangemode="tozero")
    fig.update_xaxes(tickangle=-25)
    return _fixed(fig, height=340)
