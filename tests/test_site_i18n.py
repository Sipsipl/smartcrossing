"""The website is German + English (site/src/js/i18n*.js): both languages must cover the same texts."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "site" / "src"
sys.path[:0] = [str(ROOT), str(ROOT / "site")]


def load(lang: str) -> dict:
    text = (SRC / "js" / f"i18n-{lang}.js").read_text(encoding="utf-8")
    return json.loads(text[text.index("= {") + 2:].rstrip().rstrip(";"))


def html_keys() -> set:
    keys = set()
    for f in SRC.glob("*.html"):
        t = f.read_text(encoding="utf-8")
        keys |= set(re.findall(r'data-i18n="([^"]+)"', t))
        for attrs in re.findall(r'data-i18n-attr="([^"]+)"', t):
            keys |= {pair.split("=")[1] for pair in attrs.split(",")}
    return keys


def js_keys() -> set:
    keys = set()
    for f in (SRC / "js").glob("*.js"):
        if f.name.startswith("i18n"):
            continue
        keys |= set(re.findall(r'(?<![\w.$])(?:tx|SCX\.t)\("([\w.]+)"', f.read_text(encoding="utf-8")))
    return keys


def test_same_keys_in_both_languages():
    de, en = load("de"), load("en")
    assert not set(de) - set(en), f"no English text for: {sorted(set(de) - set(en))}"
    # English may have more: the static html texts are German in the html itself
    assert not html_keys() - set(en), f"html keys without English: {sorted(html_keys() - set(en))}"
    unused = set(en) - set(de) - html_keys() - {"footer"} - {k for k in en if k.startswith("nav.")}
    assert not unused, f"English texts nobody uses: {sorted(unused)}"


def test_js_keys_exist():
    de, en = load("de"), load("en")
    dyn = {"obsrow.", "nb."}  # keys built at runtime from feature names
    missing = {k for k in js_keys() if k not in de and k not in en and not any(k.startswith(p) for p in dyn)}
    assert not missing, f"keys used in js but not defined: {sorted(missing)}"
    for k in js_keys():
        if k in en and k not in de:
            raise AssertionError(f"{k} used in js needs a German text in i18n-de.js")


def test_placeholders_match():
    de, en = load("de"), load("en")
    ph = lambda s: set(re.findall(r"\{(\w+)\}", s))  # noqa: E731
    bad = [k for k in de if ph(de[k]) != ph(en[k])]
    assert not bad, f"placeholders differ in de / en: {bad}"


def test_build_texts_have_english():
    """site/sitebuild/texts.py: one entry per run with both languages, same {placeholders} in both."""
    from obs_labels import obs_layout
    from sitebuild.texts import RUN_INFO, STEP_INFO, run_info
    from smartcrossing.sim import SimConfig

    ph = lambda s: set(re.findall(r"\{(\w+)\}", s))  # noqa: E731
    for run, e in RUN_INFO.items():
        assert set(e) == {"short", "de", "en"} and e["short"], run
        for lang in ("de", "en"):
            label, note = e[lang]
            assert label and note, (run, lang)
        assert ph(e["de"][1]) == ph(e["en"][1]) and ph(e["de"][0]) == ph(e["en"][0]) == set(), run
    for run, e in STEP_INFO.items():
        assert set(e) == {"de", "en"} and e["de"] and e["en"] and ph(e["de"]) == ph(e["en"]), run
    assert all(run_info(r)["note_en"] and run_info(r)["label_en"] for r in RUN_INFO)
    info = run_info("r5_K_s1", lambda s: s and s.replace("{stop}", "70").replace("{safe}", "99"))
    assert "70 %" in info["step"] and "70%" in info["step_en"] and "{" not in info["note"] + info["note_en"]
    assert run_info("unknown_run") == {"short": "unknown_run", "label": "unknown_run", "label_en": "unknown_run",
                                       "note": None, "note_en": None, "step": None, "step_en": None}
    for legacy in (False, True):
        lay = obs_layout(SimConfig(brake_light=not legacy, n_signal=2 if legacy else 1))
        assert len(lay["actions_en"]) == len(lay["actions"])
        for g in lay["groups"]:
            assert g["name_en"]
            assert all(it["label_en"] for it in g["items"])
