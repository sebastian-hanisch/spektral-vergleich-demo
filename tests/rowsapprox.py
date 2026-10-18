"""Vergleich von Zeilenlisten (Listen von Tupeln) mit Toleranz: pytest.approx kennt keine verschachtelten Strukturen."""

import pytest


class rows_approx:                                           # noqa: N801 (liest sich wie pytest.approx)
    def __init__(self, want, abs=1e-6):
        self.want = [v for row in want for v in row]
        self.abs = abs

    def __eq__(self, other):
        got = [v for row in other for v in row]
        return len(got) == len(self.want) and got == pytest.approx(self.want, abs=self.abs)

    def __repr__(self):
        return f"rows_approx({self.want}, abs={self.abs})"
