"""
test_label_mapper.py
────────────────────
Parametrised unit tests for the Unified Attack Mapping Engine.
Covers every raw label variant in RAW_LABEL_MAP + edge cases.

Run:  pytest tests/unit/test_label_mapper.py -v
"""
import pytest
import sys
sys.path.insert(0, "src/data")

from label_mapper import (
    CLASS_BOTNET,
    CLASS_BRUTE_FORCE,
    CLASS_DDOS,
    CLASS_DOS,
    CLASS_NORMAL,
    CLASS_PORT_SCAN,
    CLASS_WEB_ATTACK,
    UNIFIED_CLASSES,
    map_label,
    map_label_to_int,
    validate_mapping_coverage,
)


# ── Happy-path parametrised tests ────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    # Normal
    ("BENIGN",          CLASS_NORMAL),
    ("benign",          CLASS_NORMAL),
    ("  BENIGN  ",      CLASS_NORMAL),
    ("Normal",          CLASS_NORMAL),

    # DoS
    ("DoS Hulk",        CLASS_DOS),
    ("DOS HULK",        CLASS_DOS),
    ("Heartbleed",      CLASS_DOS),
    ("DoS slowloris",   CLASS_DOS),
    ("DoS Slowhttptest",CLASS_DOS),

    # DDoS
    ("DDoS",            CLASS_DDOS),
    ("DDOS",            CLASS_DDOS),
    ("DDoS-LOIC-HTTP",  CLASS_DDOS),

    # Port Scan
    ("PortScan",        CLASS_PORT_SCAN),
    ("port scan",       CLASS_PORT_SCAN),
    ("Infiltration",    CLASS_PORT_SCAN),

    # Brute Force
    ("FTP-Patator",     CLASS_BRUTE_FORCE),
    ("SSH-Patator",     CLASS_BRUTE_FORCE),
    ("Brute Force",     CLASS_BRUTE_FORCE),

    # Botnet
    ("Bot",             CLASS_BOTNET),
    ("Botnet ARES",     CLASS_BOTNET),

    # Web Attack
    ("Web Attack \u2013 Brute Force", CLASS_WEB_ATTACK),
    ("Web Attack \u2013 XSS",         CLASS_WEB_ATTACK),
    ("Web Attack \u2013 Sql Injection",CLASS_WEB_ATTACK),
    ("Web Attack XSS",                CLASS_WEB_ATTACK),
])
def test_map_label_happy_path(raw: str, expected: str) -> None:
    assert map_label(raw) == expected


# ── Edge cases ───────────────────────────────────────────────────────────────

def test_non_string_raises_type_error() -> None:
    with pytest.raises(TypeError):
        map_label(42)  # type: ignore[arg-type]

def test_strict_mode_raises_for_unknown() -> None:
    with pytest.raises(ValueError, match="Unmapped label"):
        map_label("totally_unknown_label_xyz", strict=True)

def test_fallback_returns_normal_for_unknown() -> None:
    result = map_label("totally_unknown_label_xyz", strict=False)
    assert result == CLASS_NORMAL

def test_custom_fallback() -> None:
    result = map_label("unknown_thing", fallback=CLASS_DOS, strict=False)
    assert result == CLASS_DOS

def test_map_label_to_int_range() -> None:
    for cls in UNIFIED_CLASSES:
        idx = map_label_to_int(cls)
        assert 0 <= idx <= 6

def test_validate_mapping_coverage_full() -> None:
    raw_labels = list(["BENIGN", "DoS Hulk", "DDoS", "PortScan",
                        "FTP-Patator", "Bot", "Web Attack XSS"])
    report = validate_mapping_coverage(raw_labels)
    assert report["unmapped"] == []
    assert report["mapped"] == 7

def test_validate_mapping_coverage_with_unknowns() -> None:
    report = validate_mapping_coverage(["BENIGN", "unknown_xyz"])
    assert "unknown_xyz" in report["unmapped"]
    assert report["mapped"] == 1
