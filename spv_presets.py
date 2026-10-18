"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Bedingt gezeigte Regler haben keinen eigenen Widget-Schlüssel: Ihr Wert steht unter `*_kept` und überlebt so die Läufe, in denen sie nicht erscheinen (Streamlit räumt den Zustand nicht gezeigter Widgets ab).
"""

import random
from dataclasses import dataclass

import streamlit as st

import spv_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


D = C.DEFAULTS
SETTING_SPECS = {
    "task_select": SettingSpec("tk", str, C.DEFAULT_TASK, choices=C.TASKS),
    "a_bg_kept": SettingSpec("ab", str, D["a_bg"], choices=C.BACKGROUNDS_A),
    "a_n_kept": SettingSpec("an", int, D["a_n"], choices=C.N_A),
    "b_delta_kept": SettingSpec("bd", float, D["b_delta"], C.B_DELTA_MIN, C.B_DELTA_MAX),
    "b_snr_kept": SettingSpec("bs", float, D["b_snr"], C.B_SNR_MIN, C.B_SNR_MAX),
    "c_n_kept": SettingSpec("cn", int, D["c_n"], choices=C.C_NS),
    "c_snr_kept": SettingSpec("cs", float, D["c_snr"], C.C_SNR_MIN, C.C_SNR_MAX),
    "d_bg_kept": SettingSpec("db", str, D["d_bg"], choices=C.D_BGS),
    "d_snr_kept": SettingSpec("ds", float, D["d_snr"], C.D_SNR_MIN, C.D_SNR_MAX),
    "e_line_kept": SettingSpec("el", float, D["e_line"], choices=C.E_LINE_CHOICES),
    "e_snr_kept": SettingSpec("es", float, D["e_snr"], C.E_SNR_MIN, C.E_SNR_MAX),
    "f_sig_kept": SettingSpec("fs", str, D["f_sig"], choices=C.F_SIGNALS),
    "f_snr_kept": SettingSpec("fn", float, D["f_snr"], C.F_SNR_MIN, C.F_SNR_MAX),
    "g_sig_kept": SettingSpec("gs", str, D["g_sig"], choices=C.G_SIGNALS),
    "mix_input": SettingSpec("mix", int, C.DEFAULT_MIX, C.MIX_MIN, C.MIX_MAX),
    "known_select": SettingSpec("kn", str, C.KNOWN_YES, choices=C.KNOWN_CHOICES),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
}
PRESET_KEYS = {"task": "task_select", "a_bg": "a_bg_kept", "a_n": "a_n_kept", "b_delta": "b_delta_kept", "b_snr": "b_snr_kept", "c_n": "c_n_kept", "c_snr": "c_snr_kept", "d_bg": "d_bg_kept",
               "d_snr": "d_snr_kept", "e_line": "e_line_kept", "e_snr": "e_snr_kept", "f_sig": "f_sig_kept", "f_snr": "f_snr_kept", "g_sig": "g_sig_kept", "mix": "mix_input", "known": "known_select",
               "seed": "seed_input"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt bzw. auf gültige Auswahlwerte geprüft."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        if spec.choices is not None and value not in spec.choices:
            continue
        if spec.lo is not None:
            value = max(spec.lo, value)
        if spec.hi is not None:
            value = min(spec.hi, value)
        st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    preset = C.PRESETS[name]
    for key, state_key in PRESET_KEYS.items():
        if key in preset:
            st.session_state[state_key] = preset[key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
