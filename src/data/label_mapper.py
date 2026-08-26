"""
label_mapper.py
───────────────
Unified Attack Mapping Engine for CICIDS2017 → 7-Class Taxonomy.

Handles:
  - Non-printable / invisible Unicode character stripping
  - Irregular casing normalisation
  - Leading / trailing whitespace cleanup
  - Exhaustive raw-label → unified-class mapping
  - Unknown label detection with configurable fallback

Taxonomy:
    0 → Normal
    1 → DoS
    2 → DDoS
    3 → Port Scan
    4 → Brute Force
    5 → Botnet
    6 → Web Attack

Author : Person 1 — Data & Classical ML
Project: Enterprise NIDS — Week 1 Foundation
"""

from __future__ import annotations

import re
import logging
import unicodedata
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# 7-Class Taxonomy Constants
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NORMAL      = "Normal"
CLASS_DOS         = "DoS"
CLASS_DDOS        = "DDoS"
CLASS_PORT_SCAN   = "Port Scan"
CLASS_BRUTE_FORCE = "Brute Force"
CLASS_BOTNET      = "Botnet"
CLASS_WEB_ATTACK  = "Web Attack"

UNIFIED_CLASSES: list[str] = [
    CLASS_NORMAL,
    CLASS_DOS,
    CLASS_DDOS,
    CLASS_PORT_SCAN,
    CLASS_BRUTE_FORCE,
    CLASS_BOTNET,
    CLASS_WEB_ATTACK,
]

CLASS_TO_INT: Dict[str, int] = {cls: idx for idx, cls in enumerate(UNIFIED_CLASSES)}
INT_TO_CLASS: Dict[int, str] = {idx: cls for idx, cls in enumerate(UNIFIED_CLASSES)}

# ──────────────────────────────────────────────────────────────────────────────
# Raw CICIDS2017 Label → Unified Class Mapping
# ──────────────────────────────────────────────────────────────────────────────
# Keys MUST be lowercase-stripped. The normalise() function converts raw labels
# before lookup — so all keys here are pre-normalised (lowercase, no extra WS).
# ──────────────────────────────────────────────────────────────────────────────

RAW_LABEL_MAP: Dict[str, str] = {

    # ── Normal / Benign ──────────────────────────────────────────────────────
    "benign"                            : CLASS_NORMAL,
    "normal"                            : CLASS_NORMAL,

    # ── DoS (Denial-of-Service) ──────────────────────────────────────────────
    "dos hulk"                          : CLASS_DOS,
    "dos goldeneye"                     : CLASS_DOS,
    "dos slowloris"                     : CLASS_DOS,
    "dos slowhttptest"                  : CLASS_DOS,
    "dos-hulk"                          : CLASS_DOS,
    "dos-goldeneye"                     : CLASS_DOS,
    "dos-slowloris"                     : CLASS_DOS,
    "dos-slowhttptest"                  : CLASS_DOS,
    "heartbleed"                        : CLASS_DOS,   # OpenSSL memory DoS
    "hulk"                              : CLASS_DOS,
    "goldeneye"                         : CLASS_DOS,
    "slowloris"                         : CLASS_DOS,
    "slowhttptest"                      : CLASS_DOS,

    # ── DDoS (Distributed Denial-of-Service) ────────────────────────────────
    "ddos"                              : CLASS_DDOS,
    "ddos attack"                       : CLASS_DDOS,
    "ddos-loic-http"                    : CLASS_DDOS,
    "ddos loic http"                    : CLASS_DDOS,
    "loic"                              : CLASS_DDOS,

    # ── Port Scan ────────────────────────────────────────────────────────────
    "portscan"                          : CLASS_PORT_SCAN,
    "port scan"                         : CLASS_PORT_SCAN,
    "port-scan"                         : CLASS_PORT_SCAN,
    "scanning"                          : CLASS_PORT_SCAN,
    "infiltration"                      : CLASS_PORT_SCAN,  # recon phase

    # ── Brute Force ──────────────────────────────────────────────────────────
    "brute force"                       : CLASS_BRUTE_FORCE,
    "brute-force"                       : CLASS_BRUTE_FORCE,
    "bruteforce"                        : CLASS_BRUTE_FORCE,
    "ftp-patator"                       : CLASS_BRUTE_FORCE,
    "ftp patator"                       : CLASS_BRUTE_FORCE,
    "ssh-patator"                       : CLASS_BRUTE_FORCE,
    "ssh patator"                       : CLASS_BRUTE_FORCE,
    "patator"                           : CLASS_BRUTE_FORCE,

    # ── Botnet ───────────────────────────────────────────────────────────────
    "botnet"                            : CLASS_BOTNET,
    "bot"                               : CLASS_BOTNET,
    "botnet ares"                       : CLASS_BOTNET,
    "botnet-ares"                       : CLASS_BOTNET,

    # ── Web Attack ───────────────────────────────────────────────────────────
    "web attack brute force"            : CLASS_WEB_ATTACK,
    "web attack xss"                    : CLASS_WEB_ATTACK,
    "web attack sql injection"          : CLASS_WEB_ATTACK,
    "web-attack brute force"            : CLASS_WEB_ATTACK,
    "web-attack xss"                    : CLASS_WEB_ATTACK,
    "web-attack sql injection"          : CLASS_WEB_ATTACK,
    "web attack – brute force"          : CLASS_WEB_ATTACK,
    "web attack – xss"                  : CLASS_WEB_ATTACK,
    "web attack – sql injection"        : CLASS_WEB_ATTACK,
    "web attacks"                       : CLASS_WEB_ATTACK,
    "xss"                               : CLASS_WEB_ATTACK,
    "sql injection"                     : CLASS_WEB_ATTACK,
    "sqli"                              : CLASS_WEB_ATTACK,
}


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _strip_non_printable(text: str) -> str:
    """Remove all non-printable / invisible Unicode codepoints."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch) not in {"Cc", "Cf", "Cs", "Co", "Cn"}
    )


def _normalise_label(raw: str) -> str:
    """
    Canonical normalisation pipeline:
      1. Strip non-printable Unicode
      2. Replace Unicode dashes / hyphens → ASCII hyphen
      3. Collapse multiple internal whitespace → single space
      4. Strip leading / trailing whitespace
      5. Lowercase
    """
    text = _strip_non_printable(raw)
    # Replace typographic dashes (–, —, −) with space for uniform splitting
    text = re.sub(r"[\u2013\u2014\u2212]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lower()
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def map_label(
    raw_label: str,
    fallback: Optional[str] = None,
    strict: bool = False,
) -> str:
    """
    Map a raw CICIDS2017 label string to the unified 7-class taxonomy.

    Parameters
    ----------
    raw_label : str
        The raw label string as it appears in the CSV dataset.
    fallback : str, optional
        Value returned for unmapped labels when ``strict=False``.
        Defaults to ``CLASS_NORMAL`` (conservative assumption).
    strict : bool
        If ``True``, raises ``ValueError`` for unmapped labels instead of
        returning the fallback. Useful for dataset validation runs.

    Returns
    -------
    str
        One of the 7 unified class strings.

    Raises
    ------
    ValueError
        Only when ``strict=True`` and the label is unmapped.
    TypeError
        If ``raw_label`` is not a string.
    """
    if not isinstance(raw_label, str):
        raise TypeError(
            f"raw_label must be str, got {type(raw_label).__name__!r}"
        )

    normalised = _normalise_label(raw_label)
    unified = RAW_LABEL_MAP.get(normalised)

    if unified is not None:
        return unified

    # ── Fuzzy prefix matching as a secondary safety net ─────────────────────
    for key, cls in RAW_LABEL_MAP.items():
        if normalised.startswith(key) or key.startswith(normalised):
            logger.debug(
                "Label %r matched via prefix on key %r → %s", raw_label, key, cls
            )
            return cls

    # ── Unmapped label handling ──────────────────────────────────────────────
    if strict:
        raise ValueError(
            f"Unmapped label: {raw_label!r} (normalised: {normalised!r}). "
            f"Add it to RAW_LABEL_MAP in label_mapper.py."
        )

    effective_fallback = fallback if fallback is not None else CLASS_NORMAL
    logger.warning(
        "Unmapped label %r → falling back to %r. "
        "Consider adding it to RAW_LABEL_MAP.",
        raw_label,
        effective_fallback,
    )
    return effective_fallback


def map_label_to_int(raw_label: str, **kwargs) -> int:
    """Convenience wrapper — returns integer class index (0–6)."""
    return CLASS_TO_INT[map_label(raw_label, **kwargs)]


def validate_mapping_coverage(raw_labels: list[str]) -> dict:
    """
    Run a coverage audit on a list of raw labels.

    Returns
    -------
    dict with keys:
        mapped   — labels successfully mapped
        unmapped — labels that fell back (potential gaps in RAW_LABEL_MAP)
        counts   — {unified_class: count} distribution
    """
    from collections import Counter

    mapped, unmapped = [], []
    class_counts: Counter = Counter()

    for lbl in raw_labels:
        try:
            unified = map_label(lbl, strict=True)
            mapped.append(lbl)
            class_counts[unified] += 1
        except ValueError:
            unmapped.append(lbl)

    return {
        "total"   : len(raw_labels),
        "mapped"  : len(mapped),
        "unmapped": unmapped,
        "counts"  : dict(class_counts),
    }
