"""Bias mitigation: blind the text that is used for scoring.

Removes direct identifiers (name, contact details, links) and attributes that
should not influence a hiring decision (gender, age, DOB, marital status,
nationality, religion, photo mentions, gendered pronouns).

NOTE: this reduces - it does not eliminate - bias. Things like university names
or employment gaps can still act as proxies. Keep a human in the loop.
"""
from __future__ import annotations

import re

from .parser import _EMAIL_RE, _PHONE_RE, _URL_RE

_SENSITIVE_LINE = re.compile(
    r"^\s*(gender|sex|dob|d\.o\.b|date of birth|age|marital status|nationality|religion|"
    r"father'?s? name|mother'?s? name|address|permanent address|photo|passport)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)
_PRONOUNS = re.compile(r"\b(he|she|his|her|hers|him|himself|herself|mr|mrs|ms|miss)\b\.?", re.IGNORECASE)


def anonymize(text: str, name: str = "") -> str:
    text = _SENSITIVE_LINE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = _PHONE_RE.sub(" ", text)
    if name:
        for part in name.split():
            if len(part) > 1:
                text = re.sub(rf"\b{re.escape(part)}\b", " ", text, flags=re.IGNORECASE)
    text = _PRONOUNS.sub(" ", text)
    return re.sub(r"[ \t]+", " ", text).strip()
