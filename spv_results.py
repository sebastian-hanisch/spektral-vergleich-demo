"""Vorgerechnete Tabelle der Experimente (spv_results.json): alle Zellen aller Aufgaben mit festen Zufallsströmen.

Die Tabelle ist deterministisch aus den Aufgabenfunktionen in spv_tasks.py reproduzierbar (`python spv_results.py`, mehrere Minuten); die Tests rechnen Stichproben nach. Zelle:
{'task', 'params', 'values', 'refs', 'extra'}.
"""

import argparse
import functools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import spv_constants as C
import spv_tasks as T

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spv_results.json")
VERSION = 1


def specs() -> list:
    """Alle Zellen als (Aufgabe, Parameter) in fester Reihenfolge; D rechnet je Hintergrund alle Linienstärken zusammen (Eintrag mit snr = None)."""
    out = []
    out += [("A", {"bg": bg, "n": n}) for bg in C.BACKGROUNDS_A for n in C.N_A]
    out += [("B", {"snr": snr, "delta": d}) for snr in C.B_SNRS for d in C.B_DELTAS]
    out += [("C", {"n": n, "snr": snr}) for n in C.C_NS for snr in C.C_SNRS]
    out += [("D", {"bg": bg}) for bg in C.D_BGS]
    out += [("E", {"line": line, "snr": snr}) for line in C.E_LINES for snr in C.E_SNRS]
    out += [("F", {"signal": sig, "snr": snr}) for sig in C.F_SIGNALS for snr in C.F_SNRS]
    out += [("G", {"signal": sig}) for sig in C.G_SIGNALS]
    return out


def compute(task: str, params: dict) -> list:
    """Rechnet die Zelle(n) eines Eintrags aus specs(): eine Liste von Zellen."""
    if task == "A":
        return [{"task": "A", "params": params, **T.cell_a(params["bg"], params["n"])}]
    if task == "B":
        return [{"task": "B", "params": params, **T.cell_b(params["delta"], params["snr"])}]
    if task == "C":
        return [{"task": "C", "params": params, **T.cell_c(params["n"], params["snr"])}]
    if task == "D":
        return [{"task": "D", "params": {"bg": params["bg"], "snr": snr}, **cell} for snr, cell in T.cells_d(params["bg"]).items()]
    if task == "E":
        return [{"task": "E", "params": params, **T.cell_e(params["line"], params["snr"])}]
    if task == "F":
        return [{"task": "F", "params": params, **T.cell_f(params["signal"], params["snr"])}]
    if task == "G":
        return [{"task": "G", "params": params, **T.cell_g(params["signal"])}]
    raise ValueError(f"unbekannte Aufgabe: {task}")


def _job(item):
    return compute(*item)


def generate(jobs: int = 4, only=None, path: str = PATH) -> dict:
    items = [it for it in specs() if only is None or it[0] in only]
    start = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        parts = list(pool.map(_job, items))
    cells = [c for part in parts for c in part]
    data = {"version": VERSION, "cells": cells}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=0)
    print(f"{len(cells)} Zellen in {time.time() - start:.0f} s nach {path}")
    return data


@functools.lru_cache(maxsize=1)
def load(path: str = PATH) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("version") != VERSION:
        raise ValueError("Version der Ergebnistabelle passt nicht")
    return data


def cells(task: str, path: str = PATH) -> list:
    return [c for c in load(path)["cells"] if c["task"] == task]


def find(task: str, path: str = PATH, **params):
    """Die Zelle mit genau diesen Parametern (None bei fehlender)."""
    for c in cells(task, path):
        if all(c["params"].get(k) == v for k, v in params.items()):
            return c
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ergebnistabelle neu rechnen")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only", default="", help="Aufgaben, z. B. A,B (Standard: alle)")
    a = ap.parse_args()
    generate(a.jobs, set(a.only.split(",")) if a.only else None)
