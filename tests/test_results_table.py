"""Vorgerechnete Tabelle (spv_results.json): Aufbau, Vollständigkeit und Stichproben, die gegen die Aufgabenfunktionen nachgerechnet werden."""

import json

import pytest

import spv_constants as C
import spv_results as RES

EXPECTED_COUNTS = {"A": 21, "B": 8, "C": 6, "D": 10, "E": 6, "F": 8, "G": 4}


def test_the_table_has_every_cell_of_every_task():
    data = RES.load()
    assert data["version"] == RES.VERSION == 1
    assert {t: len(RES.cells(t)) for t in C.TASKS} == EXPECTED_COUNTS and len(data["cells"]) == 63


def test_specs_enumerate_the_cells_in_a_fixed_order():
    specs = RES.specs()
    assert len(specs) == 21 + 8 + 6 + 2 + 6 + 8 + 4 and specs[0] == ("A", {"bg": "Resonanz", "n": 64}) and specs[-1] == ("G", {"signal": "Doppler"})
    assert [s for s in specs if s[0] == "D"] == [("D", {"bg": "weiß"}), ("D", {"bg": "AR(1) 0,8"})]
    stored = [(c["task"], c["params"]) for c in RES.load()["cells"]]
    expanded = []
    for task, params in specs:
        expanded += [(task, params)] if task != "D" else [("D", {"bg": params["bg"], "snr": s}) for s in C.D_SNRS]
    assert stored == expanded


def test_every_cell_has_all_its_methods_and_finite_numbers():
    for c in RES.load()["cells"]:
        methods = C.METHODS[c["task"]]
        if c["task"] == "E" and c["params"]["line"] > 0.5:
            methods = tuple(m for m in methods if m != C.E_INTERP)         # das Interpolieren aufs Gitter kennt nur Linien unter der Nyquist-Grenze
        assert list(c["values"]) == list(methods), c["params"]
        assert all(v == v and abs(v) < float("inf") and v >= 0 for v in c["values"].values()), c["params"]
        if C.TASK_KIND[c["task"]] == C.KIND_RATE:
            assert all(0.0 <= v <= 100.0 for v in c["values"].values()), c["params"]


def test_references_and_extras_are_present_where_promised():
    assert all("AR (Burg, feste Ordnung)" in c["refs"] for c in RES.cells("A") if c["params"]["bg"] != "weiß")
    assert all(list(c["refs"]) == list(C.REFERENCES["G"]) for c in RES.cells("G"))
    assert all({"thresholds", "nominal_thresholds", "false_alarm_nominal", "nominal_rate"} <= set(c["extra"]) for c in RES.cells("D"))
    assert all(set(c["extra"]["thresholds"]) >= set(C.E_KINDS) for c in RES.cells("E")) and all("crb" in c["extra"] for c in RES.cells("C"))


def test_find_and_cells():
    c = RES.find("A", bg="Kerbe", n=256)
    assert c["params"] == {"bg": "Kerbe", "n": 256} and RES.find("A", bg="Unsinn") is None and RES.find("D", bg="weiß", snr=-6.0)["params"]["snr"] == -6.0
    assert RES.cells("A")[0]["params"]["bg"] == "Resonanz" and all(x["task"] == "B" for x in RES.cells("B"))


def test_load_rejects_a_wrong_version(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"version": 99, "cells": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        RES.load(str(path))


def test_the_stored_numbers_are_reproducible_for_every_task():
    """Je Aufgabe eine Zelle neu rechnen und mit der Tabelle vergleichen (Fehlerwerte relativ, Erfolgsanteile bis auf einen Lauf)."""
    samples = [("A", {"bg": "Kerbe", "n": 64}), ("B", {"snr": 10.0, "delta": 1.0}), ("C", {"n": 64, "snr": 10.0}), ("E", {"line": 0.2, "snr": -3.0}), ("F", {"signal": "Chirp", "snr": 10.0}),
               ("G", {"signal": "Blocks"})]
    for task, params in samples:
        fresh = RES.compute(task, params)[0]
        stored = RES.find(task, **params)
        tol = dict(abs=1.0) if C.TASK_KIND[task] == C.KIND_RATE else dict(rel=1e-6)
        assert fresh["values"] == pytest.approx(stored["values"], **tol), (task, params)


def test_the_detection_cells_are_reproducible():
    fresh = {c["params"]["snr"]: c for c in RES.compute("D", {"bg": "AR(1) 0,8"})}
    for snr, c in fresh.items():
        stored = RES.find("D", bg="AR(1) 0,8", snr=snr)
        assert c["values"] == pytest.approx(stored["values"], abs=1.0) and c["extra"]["thresholds"] == pytest.approx(stored["extra"]["thresholds"], rel=1e-6)


def test_compute_rejects_unknown_tasks():
    with pytest.raises(ValueError):
        RES.compute("X", {})


def test_generate_only_the_asked_tasks(tmp_path):
    path = str(tmp_path / "g.json")
    data = RES.generate(jobs=1, only={"G"}, path=path)
    assert [c["task"] for c in data["cells"]] == ["G"] * 4 and data["version"] == RES.VERSION
    assert json.loads((tmp_path / "g.json").read_text(encoding="utf-8")) == data
    assert RES.load(path)["cells"][0]["values"] == pytest.approx(RES.find("G", signal="Blocks")["values"])
