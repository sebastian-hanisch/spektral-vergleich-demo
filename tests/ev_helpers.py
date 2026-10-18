"""Hilfen der Auswertungstests: eine Analyse mit vorgegebenen Kennzahlen bauen (für Verdikt- und Spitzentests)."""

import spv_evaluation as ev


def analysis_with(task, values, title="Test", refs=None, extra=None, **settings):
    """Analysis mit handgesetzten Kennzahlen, ohne Kurven (die Verdikte und `extremes` lesen nur Einstellungen, Kennzahlen, Bezüge und Zusatzwerte)."""
    return ev.Analysis(ev.Settings(task=task, **settings), title, "Einheit", dict(values), {}, refs=refs or {}, extra=extra or {})
