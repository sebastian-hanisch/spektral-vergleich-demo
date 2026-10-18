"""Entscheidungshilfe: Empfehlung aus den gemessenen Zellen (nur diese Szenarien, nie allgemein).

Kennt man den Szenariotyp, empfiehlt die Hilfe den Sieger der nächstgelegenen Zelle; kennt man ihn nicht, das Verfahren mit dem kleinsten größten Reuegrad über alle Zellen der Aufgabe (Minimax).
Je Verfahren steht dabei sein Wert in der Zelle, sein Reuegrad dort, sein größter und mittlerer Reuegrad über alle Zellen und der Warnhinweis, wo es am schlechtesten ist.
"""

import math
from dataclasses import dataclass

import spv_constants as C
import spv_regret as R
import spv_results as RES


@dataclass(frozen=True)
class Advice:
    task: str
    cell: dict                  # nächstgelegene Zelle
    exact: bool                 # liegt der gewählte Fall genau auf einer Zelle?
    known: bool                 # Szenariotyp bekannt?
    recommendation: str
    basis: str
    rows: tuple                 # je Verfahren: dict(method, value, regret, max_regret, mean_regret, worst_cell); value und regret sind nan, wenn die Zelle das Verfahren nicht enthält
    note: str


def _distance(task: str, params: dict, cell_params: dict) -> float:
    """Abstand zwischen dem gewählten Fall und einer Zelle: Kategorien müssen stimmen (sonst unendlich), Zahlen nach relativem Abstand."""
    d = 0.0
    for key, val in params.items():
        cv = cell_params[key]
        if isinstance(val, str) or val is None or cv is None:
            if val != cv:
                return float("inf")
        else:
            d += abs(val - cv) / (abs(cv) + 1.0)
    return d


def nearest_cell(task: str, params: dict, path: str = RES.PATH):
    """(Zelle, genau?): die Zelle mit dem kleinsten Abstand zu params; Kategorien (Hintergrund, Signal) müssen passen."""
    best, best_d = None, float("inf")
    for cell in RES.cells(task, path):
        dist = _distance(task, params, cell["params"])
        if dist < best_d:
            best, best_d = cell, dist
    if best is None:
        raise ValueError("keine passende Zelle")
    return best, best_d == 0.0


def worst_cell(task: str, method: str, cells: list) -> tuple:
    """(größter Reuegrad, Zelle) eines Verfahrens über die Zellen."""
    worst, where = -1.0, None
    for cell in cells:
        reg = R.cell_regret(task, cell)
        if method in reg and reg[method] > worst:
            worst, where = reg[method], cell
    return worst, where


def advise(task: str, params: dict, known: bool, path: str = RES.PATH) -> Advice:
    cells = RES.cells(task, path)
    cell, exact = nearest_cell(task, params, path)
    summary = R.summarize(task, cells)
    local = R.cell_regret(task, cell)
    if known:
        rec = R.winners(task, cell)[0]
        basis = f"Sieger im nächstgelegenen Szenario ({R.cell_label(task, cell['params'])})"
    else:
        rec = R.minimax_choice(summary)
        basis = f"kleinster größter Reuegrad über alle {len(cells)} Szenarien der Aufgabe"
    rows = []
    for m in C.METHODS[task]:
        if m not in summary:
            continue
        wmax, wcell = worst_cell(task, m, cells)
        rows.append({"method": m, "value": cell["values"].get(m, math.nan), "regret": local.get(m, math.nan), "max_regret": summary[m]["max"], "mean_regret": summary[m]["mean"], "worst_cell": R.cell_label(task, wcell["params"])})
    note = "" if exact else "Der gewählte Fall liegt zwischen den gemessenen Szenarien; gezeigt wird das nächstgelegene."
    return Advice(task, cell, exact, known, rec, basis, tuple(rows), note)
