"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import spv_constants as C
from spv_evaluation import LIVE_DIGITS, Analysis, de

_REPLACE = {"–": "-", "—": "-", "−": "-", "²": "2", "€": "EUR", "→": "->", "σ": "sigma", "Σ": "Summe", "±": "+-", "√": "Wurzel aus ", "×": "x", "é": "e", "„": '"', "“": '"', "π": "pi", "≥": ">=",
            "λ": "lambda", "ν": "nu", "γ": "gamma", "ρ": "rho", "φ": "phi", "₀": "0", "₁": "1", "₂": "2"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Verfahrensvergleich: Szenario und Kennzahlen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    head = f"Aufgabe {C.TASK_LABELS[s.task]}; Szenario {a.title}; Seed {s.seed}. Eine Realisierung, alles simuliert; die Wahrheit ist bekannt ({C.TASK_TRUTH[s.task]})."
    pdf.multi_cell(0, 5, _s(head), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    digits = LIVE_DIGITS[s.task]
    pdf.multi_cell(0, 5, _s(f"Kennzahl: {a.unit}."), new_x="LMARGIN", new_y="NEXT")
    for name, value in a.values.items():
        pdf.multi_cell(0, 5, _s(f"- {name}: {de(value, digits)}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: spektral-vergleich-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
