"""Die Verfahren in spv_estimators.py, spv_timefreq.py und spv_denoise.py sind wortgleich aus den Vorgängerdemos übernommen.

Liegen die Quellrepos neben diesem Repo (lokal ja, in der CI nein), wird jede Funktion mit ihrer Quelle verglichen (nach denselben Ersetzungen wie beim Übernehmen); sonst bleibt der Test übersprungen.
In den Quellrepos sind die Funktionen mit eigenen Orakeltests gesichert und mutationsgetestet."""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT.parent

SPECS = {
    "spv_estimators.py": ({}, [
        ("multitaper-demo/mtp_taper.py", ["_CACHE", "_tridiagonal", "_sturm_count", "_inverse_iteration", "dpss"]),
        ("multitaper-demo/mtp_spectrum.py", ["eigenspectra", "multitaper_equal", "f_statistic", "f_threshold", "lines_resolved"]),
        ("ar-spektrum-demo/arm_estimators.py", ["burg", "ar_spectrum"]),
        ("ar-spektrum-demo/arm_order.py", ["criteria", "select_order", "pmax_for"]),
        ("bartlett-welch-demo/wbl_welch.py", ["n_segments", "hop_for", "welch"]),
        ("unterraum-demo/ure_subspace.py", ["snapshots", "covariance", "eigendecomposition", "music_spectrum", "local_maxima", "peak_frequencies", "music_frequencies", "esprit", "estimate_count",
                                          "refine_parabola", "periodogram_frequencies", "crb_bins"]),
        ("fenster-linienlage-demo/fnl_estimators.py", ["peak_bins", "three_bins", "est_jacobsen", "est_candan"]),
        ("peak-oder-rauschen-demo/pkr_tests.py", ["fisher_p", "fisher_crit"]),
        ("peak-oder-rauschen-demo/pkr_prewhiten.py", ["residuals", "prewhiten"]),
        ("lomb-scargle-demo/lsp_lomb.py", ["Sums", "CHUNK", "sums_naive", "power_classical", "_reduction", "power_floating", "lomb_scargle", "grid"]),
        ("lomb-scargle-demo/lsp_times.py", ["KINDS", "JITTER", "PERIOD", "WIDTH", "END_FRACTION", "times"])]),
    "spv_timefreq.py": ({"T.stft3(": "stft3(", "T.freq_axis(": "freq_axis(", "T.gauss_windows(": "gauss_windows(", "C.NFFT": "NFFT", "C.DEFAULT_FLOOR": "DEFAULT_FLOOR", "C.W0": "W0", "C.FMIN": "0.02", "C.FMAX": "0.2", "C.VOICES": "12"}, [
        ("reassignment-demo/rea_stft.py", ["gauss_windows", "stft3", "freq_axis"]),
        ("reassignment-demo/rea_reassign.py", ["reassign_fields", "_accumulate", "reassigned_from", "squeezed_from"]),
        ("reassignment-demo/rea_signals.py", ["analytic", "noise_sigma", "TestSignal", "linear_chirp", "sine_fm"]),
        ("kohaerenz-demo/koh_wavelet.py", ["scale_of", "freq_grid", "cwt"]),
        ("wavelet-skalogramm-demo/cwt_morlet.py", ["power", "ridge_frequencies"])]),
    "spv_denoise.py": ({"W.dwt": "dwt", "W.idwt": "idwt", "W.": "W_"}, [
        ("wavelet-schrumpfung-demo/shr_wavelet.py", ["daubechies", "qmf", "check_levels", "dwt", "idwt"]),
        ("wavelet-schrumpfung-demo/shr_shrink.py", ["soft", "hard", "apply_rule", "universal_threshold", "mad_sigma", "sure_risk", "sure_threshold", "level_threshold", "shrink_details", "shrink"]),
        ("wavelet-schrumpfung-demo/shr_signals.py", ["BLOCKS_H", "BUMPS_H", "BUMPS_W", "KNOTS", "_grid", "blocks_raw", "bumps_raw", "heavisine_raw", "doppler_raw", "RAW", "shape", "signal"])]),
}


def top_level(path):
    """{Name: Quelltext} der Funktionen, Klassen und Zuweisungen auf oberster Ebene (mit Dekoratoren)."""
    src = path.read_text(encoding="utf-8")
    lines = src.split("\n")
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            name = node.name
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name = node.target.id
        else:
            continue
        start = node.lineno - 1
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.decorator_list:
            start = node.decorator_list[0].lineno - 1
        out[name] = "\n".join(lines[start:node.end_lineno])
    return out


def normalise(text, replace):
    for a, b in replace.items():
        text = text.replace(a, b)
    return text


@pytest.mark.parametrize("module", list(SPECS))
def test_every_function_is_word_for_word_from_its_source(module):
    replace, specs = SPECS[module]
    here = top_level(ROOT / module)
    checked = 0
    for rel, names in specs:
        source = PARENT / rel
        if not source.exists():
            pytest.skip(f"Quellrepo nicht vorhanden: {rel}")
        there = top_level(source)
        for name in names:
            assert name in here, f"{name} fehlt in {module}"
            assert here[name] == normalise(there[name], replace), f"{name} weicht von {rel} ab"
            checked += 1
    assert checked == sum(len(n) for _, n in specs)


@pytest.mark.parametrize("module", list(SPECS))
def test_no_function_without_a_source(module):
    """Jede Funktion und Klasse der Module stammt aus der Liste (nichts Eigenes versteckt sich zwischen den übernommenen Funktionen)."""
    _, specs = SPECS[module]
    listed = {n for _, names in specs for n in names}
    own = {name for name, src in top_level(ROOT / module).items() if src.startswith(("def ", "class ", "@")) and name not in listed}
    assert own == set(), own
