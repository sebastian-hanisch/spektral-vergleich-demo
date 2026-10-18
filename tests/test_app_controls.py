"""AppTest: Aufgaben, Permalink, bedingte Regler, Mix, Entscheidungshilfe, PDF (die Rauchtests in test_app.py decken nur Knöpfe und den Mix-Regler ab)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import spv_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = 300
BG_A, N_A, DELTA, SNR_B, N_C, BG_D, LINE, SIG_F, SIG_G = "Hintergrund", "Länge n", "Abstand der Linien (Bins)", "Rauschabstand je Linie (dB)", "Länge n", "Hintergrund", "Lage der Linie", "Signal", "Signal"
MIX = "Anteil der autoregressiven Hintergründe im Mix (%)"


def _run(params=None):
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    if params:
        at.query_params.update(params)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _set_task(at, task):
    at.selectbox(key="task_select").set_value(task).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]


def _sidebar_labels(at):
    return [x.label for x in at.sidebar.selectbox] + [x.label for x in at.sidebar.slider] + [x.label for x in at.sidebar.select_slider] + [x.label for x in at.sidebar.radio]


def _widget(at, label):
    pool = list(at.sidebar.selectbox) + list(at.sidebar.slider) + list(at.sidebar.select_slider) + list(at.sidebar.radio)
    return {x.label: x for x in pool}[label]


@pytest.mark.parametrize("task", list(C.TASKS))
def test_every_task_runs(task):
    at = _run()
    _set_task(at, task)
    assert [m.label for m in at.metric][:3] == [("Bestes Verfahren" if C.TASK_KIND[task] == C.KIND_ERROR else "Erfolgreichstes Verfahren"), "Schlechtestes Verfahren", "Nächstes Tabellenszenario"]


RADIOS = {"D": ("Hintergrund", C.D_BGS), "E": ("Lage der Linie", C.E_LINE_CHOICES), "F": ("Signal", C.F_SIGNALS), "G": ("Signal", C.G_SIGNALS)}


@pytest.mark.parametrize("task", list(C.TASKS))
def test_every_sidebar_control_at_its_limits(task):
    at = _run()
    _set_task(at, task)
    controls = [("slider", x.label) for x in at.sidebar.slider] + [("select_slider", x.label) for x in at.sidebar.select_slider]
    for kind, label in controls:
        for edge in (0, -1):
            at = _run()
            _set_task(at, task)
            w = _widget(at, label)
            if kind == "slider":
                value = (w.min, w.max)[edge]
                value = int(value) if isinstance(w.value, int) else value
            else:
                value = w.options[edge]
            w.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (task, label, edge, [str(e) for e in at.exception])
    if task in RADIOS:
        label, options = RADIOS[task]
        for value in (options[0], options[-1]):
            at = _run()
            _set_task(at, task)
            _widget(at, label).set_value(value).run(timeout=TIMEOUT)
            assert not at.exception and _widget(at, label).value == value, (task, value, [str(e) for e in at.exception])


def test_the_controls_follow_the_task():
    at = _run()
    assert {BG_A, N_A} <= set(_sidebar_labels(at)) and not {DELTA, SNR_B, LINE, SIG_F} & set(_sidebar_labels(at))
    _set_task(at, "B")
    assert {DELTA, SNR_B} <= set(_sidebar_labels(at)) and not {BG_A, LINE} & set(_sidebar_labels(at))
    _set_task(at, "E")
    assert {LINE, "Rauschabstand (dB)"} <= set(_sidebar_labels(at)) and not {DELTA, BG_A} & set(_sidebar_labels(at))
    _set_task(at, "F")
    assert {SIG_F, "Rauschabstand (dB)"} <= set(_sidebar_labels(at)) and LINE not in _sidebar_labels(at)
    _set_task(at, "G")
    assert SIG_G in _sidebar_labels(at) and "Rauschabstand (dB)" not in _sidebar_labels(at)


def test_the_metric_columns_follow_the_task():
    at = _run()
    assert [m.label for m in at.metric] == ["Bestes Verfahren", "Schlechtestes Verfahren", "Nächstes Tabellenszenario"]
    _set_task(at, "C")
    assert [m.label for m in at.metric] == ["Bestes Verfahren", "Schlechtestes Verfahren", "Nächstes Tabellenszenario", "Cramér-Rao-Schranke"]
    _set_task(at, "G")
    assert [m.label for m in at.metric][-1] == "Verrauscht"
    _set_task(at, "B")
    assert [m.label for m in at.metric] == ["Erfolgreichstes Verfahren", "Schlechtestes Verfahren", "Nächstes Tabellenszenario"]
    assert at.metric[0].value == "4 gleichauf" and at.metric[2].value == "Abstand 1 Bins, 10 dB"


def test_conditional_values_survive_the_runs_without_their_widget():
    at = _run()
    _widget(at, BG_A).set_value("Kerbe").run(timeout=TIMEOUT)
    _widget(at, N_A).set_value(1024).run(timeout=TIMEOUT)
    _set_task(at, "B")
    _widget(at, DELTA).set_value(2.5).run(timeout=TIMEOUT)
    _set_task(at, "E")
    _widget(at, LINE).set_value(0.2).run(timeout=TIMEOUT)
    _set_task(at, "A")
    assert not at.exception and _widget(at, BG_A).value == "Kerbe" and _widget(at, N_A).value == 1024
    _set_task(at, "B")
    assert _widget(at, DELTA).value == 2.5
    _set_task(at, "E")
    assert _widget(at, LINE).value == 0.2


def test_permalink_is_applied_and_invalid_values_are_ignored():
    at = _run({"tk": "B", "bd": "2.5", "bs": "20", "seed": "42", "mix": "30", "kn": C.KNOWN_NO})
    assert at.selectbox(key="task_select").value == "B" and _widget(at, DELTA).value == 2.5 and _widget(at, SNR_B).value == 20.0
    assert at.number_input(key="seed_input").value == 42 and at.slider(key="mix_input").value == 30 and at.radio(key="known_select").value == C.KNOWN_NO
    at2 = _run({"tk": "Z", "ab": "Unsinn", "an": "100", "bd": "99", "bs": "-5", "mix": "500", "ds": "-99", "el": "0.7", "kn": "vielleicht"})
    assert at2.selectbox(key="task_select").value == C.DEFAULT_TASK and at2.slider(key="mix_input").value == C.MIX_MAX and at2.radio(key="known_select").value == C.KNOWN_YES
    assert _widget(at2, BG_A).value == "Resonanz" and _widget(at2, N_A).value == 256
    _set_task(at2, "B")
    assert _widget(at2, DELTA).value == C.B_DELTA_MAX and _widget(at2, SNR_B).value == C.B_SNR_MIN
    _set_task(at2, "D")
    assert _widget(at2, "Rauschabstand der Linie (dB)").value == C.D_SNR_MIN
    _set_task(at2, "E")
    assert _widget(at2, LINE).value == 1.3


def test_preset_button_sets_the_widgets():
    at = _run()
    names = list(C.PRESETS)
    at.button(key=f"preset_{names[2]}").click().run(timeout=TIMEOUT)
    assert not at.exception and at.selectbox(key="task_select").value == "B" and _widget(at, DELTA).value == 1.0 and _widget(at, SNR_B).value == 10.0
    at.button(key=f"preset_{names[3]}").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="task_select").value == "D" and _widget(at, BG_D).value == "AR(1) 0,8" and _widget(at, "Rauschabstand der Linie (dB)").value == -6.0
    at.button(key=f"preset_{names[4]}").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="task_select").value == "E" and _widget(at, LINE).value == 1.3
    at.button(key=f"preset_{names[1]}").click().run(timeout=TIMEOUT)
    assert at.selectbox(key="task_select").value == "A" and _widget(at, BG_A).value == "Kerbe" and _widget(at, N_A).value == 1024


def test_the_mix_slider_changes_the_summary():
    at = _run()
    caption = [c.value for c in at.caption if c.value.startswith("Zusammenfassung über den Mix")]
    assert caption == ["Zusammenfassung über den Mix 71 % autoregressiv (15 Szenarien) und 29 % Kerbe oder Bandpass (6 Szenarien); gleich gewichtet daneben."]
    at.slider(key="mix_input").set_value(10).run(timeout=TIMEOUT)
    assert not at.exception and any(c.value.startswith("Zusammenfassung über den Mix 10 % autoregressiv") for c in at.caption)
    assert any("Unter dem eingestellten Mix (10 % autoregressiv)" in w.value for w in at.warning)


def test_the_advice_follows_the_known_answer():
    at = _run({"tk": "A", "ab": "Kerbe", "an": "1024"})
    assert any("**Welch (Hann, 50 %, n/4)**" in s.value for s in at.success)
    at.radio(key="known_select").set_value(C.KNOWN_NO).run(timeout=TIMEOUT)
    assert not at.exception and any("**AR (Burg, BIC)**" in s.value and "kleinster größter Reuegrad über alle 21 Szenarien" in s.value for s in at.success)


def test_the_advice_notes_a_scenario_between_the_measured_ones():
    at = _run({"tk": "D", "ds": "-4"})
    assert any(c.value == "Der gewählte Fall liegt zwischen den gemessenen Szenarien; gezeigt wird das nächstgelegene." for c in at.caption)
    at = _run({"tk": "D", "ds": "-6"})
    assert not any("zwischen den gemessenen Szenarien" in c.value for c in at.caption)


def test_pdf_button_builds_a_download():
    at = _run()
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert any(d.proto.label.startswith("⬇️") for d in at.get("download_button"))


def test_the_dice_button_changes_the_seed():
    at = _run()
    before = at.number_input(key="seed_input").value
    [b for b in at.button if "würfeln" in b.label][0].click().run(timeout=TIMEOUT)
    assert not at.exception and at.number_input(key="seed_input").value != before
