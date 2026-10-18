"""Die Experimente je Prozess nur einmal rechnen (feste Ströme und feste Tabelle: das Ergebnis ist immer dasselbe)."""

import functools

import spv_evaluation as ev


@functools.lru_cache(maxsize=None)
def get(name: str):
    return getattr(ev, "exp_" + name)()
