"""Presets: Vollständigkeit, gültige Werte, Standardwerte, Parameterschlüssel."""

import spv_constants as C
import spv_evaluation as ev
import spv_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_choices_and_bounds():
    for name, p in C.PRESETS.items():
        assert p["task"] in C.TASKS and p["a_bg"] in C.BACKGROUNDS_A and p["a_n"] in C.N_A and p["c_n"] in C.C_NS and p["d_bg"] in C.D_BGS and p["e_line"] in C.E_LINE_CHOICES, name
        assert p["f_sig"] in C.F_SIGNALS and p["g_sig"] in C.G_SIGNALS and p["known"] in C.KNOWN_CHOICES, name
        assert C.B_DELTA_MIN <= p["b_delta"] <= C.B_DELTA_MAX and C.B_SNR_MIN <= p["b_snr"] <= C.B_SNR_MAX and C.C_SNR_MIN <= p["c_snr"] <= C.C_SNR_MAX, name
        assert C.D_SNR_MIN <= p["d_snr"] <= C.D_SNR_MAX and C.E_SNR_MIN <= p["e_snr"] <= C.E_SNR_MAX and C.F_SNR_MIN <= p["f_snr"] <= C.F_SNR_MAX, name
        assert C.MIX_MIN <= p["mix"] <= C.MIX_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert all(isinstance(p[k], float) for k in ("b_delta", "b_snr", "c_snr", "d_snr", "e_line", "e_snr", "f_snr")) and all(isinstance(p[k], int) for k in ("a_n", "c_n", "mix", "seed")), name


def test_presets_cover_the_tasks_they_promise():
    assert [p["task"] for p in C.PRESETS.values()] == ["A", "A", "B", "D", "E"]


def test_default_settings_are_valid_choices():
    s = ev.Settings()
    assert s.task in C.TASKS and s.a_bg in C.BACKGROUNDS_A and s.a_n in C.N_A and s.c_n in C.C_NS and s.d_bg in C.D_BGS and s.e_line in C.E_LINE_CHOICES and s.f_sig in C.F_SIGNALS and s.g_sig in C.G_SIGNALS
    assert C.B_DELTA_MIN <= s.b_delta <= C.B_DELTA_MAX and C.B_SNR_MIN <= s.b_snr <= C.B_SNR_MAX and C.C_SNR_MIN <= s.c_snr <= C.C_SNR_MAX and C.D_SNR_MIN <= s.d_snr <= C.D_SNR_MAX
    assert C.E_SNR_MIN <= s.e_snr <= C.E_SNR_MAX and C.F_SNR_MIN <= s.f_snr <= C.F_SNR_MAX and C.DEFAULT_MIX in range(C.MIX_MIN, C.MIX_MAX + 1)


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    s = ev.Settings()
    defaults = {"task": s.task, "a_bg": s.a_bg, "a_n": s.a_n, "b_delta": s.b_delta, "b_snr": s.b_snr, "c_n": s.c_n, "c_snr": s.c_snr, "d_bg": s.d_bg, "d_snr": s.d_snr, "e_line": s.e_line, "e_snr": s.e_snr,
                "f_sig": s.f_sig, "f_snr": s.f_snr, "g_sig": s.g_sig, "mix": C.DEFAULT_MIX, "known": C.KNOWN_YES, "seed": s.seed}
    for key, state_key in P.PRESET_KEYS.items():
        assert P.SETTING_SPECS[state_key].default == defaults[key], key


def test_url_params_are_unique_and_not_password_like():
    params = [spec.url_param for spec in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params)) and max(len(n) for n in C.PRESETS) <= 70
    assert not {"pw", "pass", "password", "pwd", "token", "key"} & set(params)


def test_bounds_and_the_constant_relations():
    assert P.bounds("seed_input") == (0, C.SEED_MAX) and P.bounds("b_delta_kept") == (C.B_DELTA_MIN, C.B_DELTA_MAX) and P.bounds("mix_input") == (C.MIX_MIN, C.MIX_MAX) and P.bounds("task_select") == (None, None)
    assert C.DEFAULT_TASK in C.TASKS and C.DEFAULTS["a_bg"] in C.BACKGROUNDS_A and set(C.AR_TYPE) | set(C.NON_AR_TYPE) == set(C.BACKGROUNDS_A) and not set(C.AR_TYPE) & set(C.NON_AR_TYPE)
    assert set(C.METHODS) == set(C.TASKS) == set(C.TASK_LABELS) == set(C.TASK_KIND) and all(len(set(m)) == len(m) >= 3 for m in C.METHODS.values())
    assert set(C.TASK_KIND.values()) == {C.KIND_ERROR, C.KIND_RATE} and C.E_INTERP in C.METHODS["E"] and C.E_LINE_CHOICES == C.E_LINES
    assert C.B_DELTA_MIN < min(C.B_DELTAS) and max(C.B_DELTAS) < C.B_DELTA_MAX and C.F_SNR_MAX > max(s for s in C.F_SNRS if s is not None)
