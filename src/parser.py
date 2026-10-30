"""Step 2/3 - Information extraction from resumes and job descriptions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

from .skills_db import SKILLS

# ---------------------------------------------------------------- skills
def _build_patterns() -> dict[str, re.Pattern]:
    patterns = {}
    for canonical, synonyms in SKILLS.items():
        alts = "|".join(re.escape(s) for s in sorted(set(synonyms), key=len, reverse=True))
        patterns[canonical] = re.compile(
            rf"(?<![A-Za-z0-9+#]){alts}(?![A-Za-z0-9+#])", re.IGNORECASE
        )
    return patterns


_SKILL_PATTERNS = _build_patterns()


def extract_skills(text: str) -> set[str]:
    """Return the set of canonical skills mentioned in `text`."""
    return {skill for skill, pat in _SKILL_PATTERNS.items() if pat.search(text)}


# --------------------------------------------------------------- sections
_SECTION_ALIASES = {
    "summary": ["summary", "profile", "objective", "career objective", "professional summary", "about me"],
    "skills": ["skills", "technical skills", "key skills", "core competencies", "skills & tools"],
    "experience": ["experience", "work experience", "professional experience", "employment history",
                   "work history", "internships", "internship experience"],
    "education": ["education", "academic background", "academics", "qualifications", "education & certifications"],
    "projects": ["projects", "academic projects", "personal projects", "key projects"],
    "certifications": ["certifications", "certificates", "licenses", "courses", "training"],
}
_ALIAS_TO_SECTION = {a: sec for sec, al in _SECTION_ALIASES.items() for a in al}


def split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {"header": []}
    current = "header"
    for line in text.splitlines():
        key = re.sub(r"[:\-–_=*#\s]+$", "", line.strip().lower())
        key = re.sub(r"^[#*\-\s]+", "", key)
        if key in _ALIAS_TO_SECTION:
            current = _ALIAS_TO_SECTION[key]
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


# --------------------------------------------------------------- contact
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3,5}\)?[\s-]?)?\d{3,5}[\s-]?\d{4,5}(?!\d)")
_URL_RE = re.compile(r"(?:https?://|www\.|linkedin\.com|github\.com)\S+", re.IGNORECASE)


def extract_email(text: str) -> str:
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else ""


def extract_phone(text: str) -> str:
    for m in _PHONE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 13:
            return m.group(0).strip()
    return ""


def extract_name(text: str, fallback: str = "Unknown") -> str:
    for line in text.splitlines()[:6]:
        line = line.strip()
        if not line or "@" in line or any(ch.isdigit() for ch in line):
            continue
        if line.lower().rstrip(":") in _ALIAS_TO_SECTION:
            continue
        words = line.split()
        if 1 < len(words) <= 4 and all(re.fullmatch(r"[A-Za-z.'\-]+", w) for w in words):
            return line.title() if line.isupper() else line
    return fallback


# -------------------------------------------------------------- education
EDU_LEVELS = {0: "Not specified", 1: "Diploma", 2: "Bachelor's", 3: "Master's", 4: "PhD"}
_EDU_PATTERNS = [
    (4, re.compile(r"\b(ph\.?\s?d|doctorate|doctoral)\b", re.I)),
    (3, re.compile(r"\b(m\.?tech|m\.?sc|m\.e\b|mba|mca|m\.?s\b|master'?s?|masters)\b", re.I)),
    (2, re.compile(r"\b(b\.?tech|b\.?sc|b\.e\b|bca|b\.?com|b\.?a\b|bachelor'?s?|bachelors|undergraduate)\b", re.I)),
    (1, re.compile(r"\b(diploma|polytechnic)\b", re.I)),
]


def extract_education_level(text: str) -> int:
    for level, pat in _EDU_PATTERNS:
        if pat.search(text):
            return level
    return 0


# ------------------------------------------------------------- experience
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_RANGE_RE = re.compile(
    rf"(?:{_MONTH}\s+)?((?:19|20)\d{{2}})\s*(?:-|–|—|to)\s*"
    rf"(?:(?:{_MONTH}\s+)?((?:19|20)\d{{2}})|(present|current|now|till date|ongoing))",
    re.IGNORECASE,
)
_YEARS_RE = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)


def _merge_intervals(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total, (cs, ce) = 0, intervals[0]
    for s, e in intervals[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            total += ce - cs
            cs, ce = s, e
    return total + (ce - cs)


def extract_experience_years(text: str, sections: dict[str, str] | None = None) -> float:
    """Estimate total years of experience.

    1. Sum (de-overlapped) date ranges inside the EXPERIENCE section.
    2. Fall back to explicit statements like '5 years of experience'.
    Takes the larger of the two to be fair to both resume styles.
    """
    sections = sections or split_sections(text)
    this_year = date.today().year
    intervals = []
    exp_text = sections.get("experience", "")
    for m in _RANGE_RE.finditer(exp_text):
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else this_year
        if 1980 <= start <= end <= this_year + 1:
            intervals.append((start, end))
    from_dates = float(_merge_intervals(intervals))

    stated = 0.0
    scan = text if not exp_text else (sections.get("summary", "") + "\n" + sections.get("header", ""))
    for m in _YEARS_RE.finditer(scan):
        val = float(m.group(1))
        if val <= 40:
            stated = max(stated, val)
    return max(from_dates, stated)


# ---------------------------------------------------- certifications etc.
def count_certifications(sections: dict[str, str]) -> int:
    block = sections.get("certifications", "")
    return len([ln for ln in block.splitlines() if ln.strip()])


def count_projects(sections: dict[str, str]) -> int:
    block = sections.get("projects", "")
    return len([ln for ln in block.splitlines() if ln.strip().startswith(("-", "•", "*")) or len(ln.strip()) > 25])


# ----------------------------------------------------------- data classes
@dataclass
class ResumeInfo:
    filename: str
    text: str
    name: str = ""
    email: str = ""
    phone: str = ""
    skills: set[str] = field(default_factory=set)
    evidence_skills: set[str] = field(default_factory=set)  # skills mentioned OUTSIDE the skills list
    education_level: int = 0
    experience_years: float = 0.0
    n_certifications: int = 0
    n_projects: int = 0


@dataclass
class JobInfo:
    text: str
    required_skills: set[str] = field(default_factory=set)
    preferred_skills: set[str] = field(default_factory=set)
    min_years: float = 0.0
    education_level: int = 0


def parse_resume(filename: str, text: str) -> ResumeInfo:
    sections = split_sections(text)
    # Skills mentioned in experience/projects/summary prove usage; a bare skills list can be keyword stuffing.
    evidence_text = "\n".join(v for k, v in sections.items() if k != "skills")
    return ResumeInfo(
        filename=filename,
        text=text,
        name=extract_name(text, fallback=filename.rsplit(".", 1)[0]),
        email=extract_email(text),
        phone=extract_phone(text),
        skills=extract_skills(text),
        evidence_skills=extract_skills(evidence_text),
        education_level=extract_education_level(sections.get("education", "") or text),
        experience_years=extract_experience_years(text, sections),
        n_certifications=count_certifications(sections),
        n_projects=count_projects(sections),
    )


_PREFERRED_MARKER = re.compile(r"^\s*(preferred|nice to have|good to have|bonus|plus|desirable)", re.I)


def parse_job_description(text: str) -> JobInfo:
    """Skills listed after a 'Preferred / Nice to have' heading are treated as optional."""
    required_lines, preferred_lines, in_preferred = [], [], False
    for line in text.splitlines():
        if _PREFERRED_MARKER.match(line):
            in_preferred = True
        elif re.match(r"^\s*(required|requirements|responsibilities|must have|qualifications)", line, re.I):
            in_preferred = False
        (preferred_lines if in_preferred else required_lines).append(line)

    required = extract_skills("\n".join(required_lines))
    preferred = extract_skills("\n".join(preferred_lines)) - required

    min_years = 0.0
    m = re.search(r"(\d{1,2})\s*(?:-|–|to)\s*\d{1,2}\s*\+?\s*(?:years?|yrs?)", text, re.I)
    if m:
        min_years = float(m.group(1))
    else:
        vals = [float(v) for v in _YEARS_RE.findall(text)]
        min_years = min(vals) if vals else 0.0

    return JobInfo(
        text=text,
        required_skills=required,
        preferred_skills=preferred,
        min_years=min_years,
        education_level=extract_education_level(text),
    )
