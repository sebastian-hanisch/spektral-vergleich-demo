"""Reuegrad (Regret) und Szenarienmix: Vergleich mehrerer Verfahren über mehrere Szenarien mit einem Maß.

Fehleraufgaben (kleiner ist besser): Reuegrad = Fehler / bester Fehler im Szenario (1 = bester). Erfolgsaufgaben (größer ist besser): Reuegrad = bester Erfolgsanteil minus eigener (Prozentpunkte, 0 = bester).
Je Verfahren: mittlerer Reuegrad (gewichtet), größter Reuegrad (Minimax-Kriterium) und Zahl der Siege. Der Szenarienmix gewichtet die Szenarien (nur Aufgabe A: Anteil der autoregressiven Hintergründe).
"""

import math

import spv_constants as C


def regret_ratio(values: dict) -> dict:
    """Fehler durch besten Fehler (alle Werte nichtnegativ); bei bestem Fehler null: 1 für Verfahren mit Fehler null, sonst unendlich."""
    best = min(values.values())
    return {k: (v / best if best > 0 else (1.0 if v == best else math.inf)) for k, v in values.items()}


def regret_points(values: dict) -> dict:
    """Bester Erfolgsanteil minus eigener (Punkte)."""
    best = max(values.values())
    return {k: best - v for k, v in values.items()}


def cell_regret(task: str, cell: dict) -> dict:
    """Reuegrad je Verfahren in einer Zelle (Bezüge ausgenommen)."""
    values = {k: cell["values"][k] for k in C.METHODS[task] if k in cell["values"]}
    return regret_ratio(values) if C.TASK_KIND[task] == C.KIND_ERROR else regret_points(values)


def winners(task: str, cell: dict) -> list:
    """Verfahren mit dem besten Wert der Zelle (bei Gleichstand alle)."""
    values = {k: cell["values"][k] for k in C.METHODS[task] if k in cell["values"]}
    best = min(values.values()) if C.TASK_KIND[task] == C.KIND_ERROR else max(values.values())
    return [k for k, v in values.items() if v == best]


def summarize(task: str, cells: list, weights=None) -> dict:
    """{Verfahren: {'mean', 'max', 'wins', 'wins_weighted', 'n'}} über die Zellen; weights: Gewichte je Zelle (Summe beliebig, wird normiert), Standard gleich."""
    if not cells:
        raise ValueError("keine Zellen")
    w = [1.0] * len(cells) if weights is None else [float(x) for x in weights]
    if len(w) != len(cells) or min(w) < 0 or sum(w) <= 0:
        raise ValueError("ungültige Gewichte")
    total = sum(w)
    out = {}
    for cell, wi in zip(cells, w):
        reg = cell_regret(task, cell)
        win = winners(task, cell)
        for k, r in reg.items():
            d = out.setdefault(k, {"mean": 0.0, "max": 0.0, "wins": 0, "wins_weighted": 0.0, "n": 0})
            d["mean"] += wi * r / total
            if wi > 0:
                d["max"] = max(d["max"], r)
            d["wins"] += k in win
            d["wins_weighted"] += (wi / total) * (k in win)
            d["n"] += 1
    return out


def minimax_choice(summary: dict) -> str:
    """Verfahren mit dem kleinsten größten Reuegrad (bei Gleichstand der kleinere mittlere, dann die Reihenfolge)."""
    return min(summary, key=lambda k: (summary[k]["max"], summary[k]["mean"]))


def mix_weights(cells: list, ar_share: float) -> list:
    """Gewichte der Aufgabe-A-Zellen: ar_share (0 bis 1) verteilt sich gleich auf die Zellen mit autoregressivem (oder weißem) Hintergrund, der Rest gleich auf Kerbe und Bandpass."""
    if not 0.0 <= ar_share <= 1.0:
        raise ValueError("Anteil außerhalb von 0 bis 1")
    ar = [c["params"]["bg"] in C.AR_TYPE for c in cells]
    n_ar, n_non = sum(ar), len(ar) - sum(ar)
    if n_ar == 0 or n_non == 0:
        raise ValueError("beide Gruppen nötig")
    return [ar_share / n_ar if a else (1.0 - ar_share) / n_non for a in ar]


def cell_label(task: str, params: dict) -> str:
    """Kurzbeschreibung eines Szenarios für Texte und Tabellen."""
    p = params
    if task == "A":
        return f"{p['bg']}, n = {p['n']}"
    if task == "B":
        return f"Abstand {p['delta']:g} Bins, {p['snr']:g} dB".replace(".", ",")
    if task == "C":
        return f"n = {p['n']}, {p['snr']:g} dB"
    if task == "D":
        return f"{p['bg']}, Linie {p['snr']:g} dB".replace(".", ",")
    if task == "E":
        return f"Linie bei {p['line']:g}, {p['snr']:g} dB".replace(".", ",")
    if task == "F":
        return f"{p['signal']}, {'rauschfrei' if p['snr'] is None else format(p['snr'], 'g') + ' dB'}"
    if task == "G":
        return p["signal"]
    raise ValueError(f"unbekannte Aufgabe: {task}")
