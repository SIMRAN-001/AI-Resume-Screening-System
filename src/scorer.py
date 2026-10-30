"""Step 4 - Matching and scoring (hybrid: semantic + skills + experience + education + extras)."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .parser import EDU_LEVELS, JobInfo, ResumeInfo


@dataclass
class Weights:
    semantic: float = 0.40
    skills: float = 0.30
    experience: float = 0.15
    education: float = 0.10
    extras: float = 0.05

    def normalized(self) -> "Weights":
        d = asdict(self)
        total = sum(d.values()) or 1.0
        return Weights(**{k: v / total for k, v in d.items()})


# ------------------------------------------------------------ semantic part
_SBERT_MODEL = None


def _load_sbert():
    global _SBERT_MODEL
    if _SBERT_MODEL is None:
        from sentence_transformers import SentenceTransformer

        _SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _SBERT_MODEL


def semantic_scores(jd_text: str, resume_texts: list[str], use_embeddings: bool = False) -> tuple[list[float], str]:
    """Return (scores in 0..1, engine name). Falls back to TF-IDF if SBERT is unavailable."""
    if use_embeddings:
        try:
            model = _load_sbert()
            emb = model.encode([jd_text] + resume_texts, normalize_embeddings=True)
            sims = emb[1:] @ emb[0]
            # MiniLM cosine for a good match is usually ~0.5-0.7; stretch to 0..1
            return [float(np.clip(s / 0.7, 0, 1)) for s in sims], "sentence-transformers (all-MiniLM-L6-v2)"
        except Exception as exc:  # not installed / no internet to download model
            print(f"[warn] embeddings unavailable ({exc.__class__.__name__}); using TF-IDF instead")

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    matrix = vec.fit_transform([jd_text] + resume_texts)
    sims = cosine_similarity(matrix[0], matrix[1:])[0]
    # TF-IDF cosine for a strong match is usually ~0.3-0.5; stretch to 0..1
    return [float(np.clip(s / 0.5, 0, 1)) for s in sims], "TF-IDF (scikit-learn)"


# --------------------------------------------------------- rule-based parts
UNVERIFIED_CREDIT = 0.6  # skill only appears in the skills list, never in experience/projects


def skill_score(resume: ResumeInfo, job: JobInfo) -> tuple[float, list[str], list[str], list[str]]:
    """Weighted skill coverage. Skills backed by experience/projects earn full credit; skills that
    only appear in a bare skills list earn UNVERIFIED_CREDIT (anti keyword-stuffing)."""
    req, pref = job.required_skills, job.preferred_skills

    def credit(skill: str) -> float:
        return 1.0 if skill in resume.evidence_skills else UNVERIFIED_CREDIT

    matched_req = sorted(resume.skills & req)
    missing_req = sorted(req - resume.skills)
    matched_pref = sorted(resume.skills & pref)
    denom = len(req) + 0.5 * len(pref)
    if denom == 0:
        return 0.5, [], [], []  # JD lists no recognizable skills -> neutral
    earned = sum(credit(s) for s in matched_req) + 0.5 * sum(credit(s) for s in matched_pref)
    return earned / denom, matched_req + matched_pref, missing_req, sorted(resume.skills - req - pref)


def experience_score(resume: ResumeInfo, job: JobInfo) -> float:
    if job.min_years <= 0:
        return 1.0
    return float(min(1.0, resume.experience_years / job.min_years))


def education_score(resume: ResumeInfo, job: JobInfo) -> float:
    if job.education_level <= 0:
        return 1.0
    if resume.education_level >= job.education_level:
        return 1.0
    return resume.education_level / job.education_level


def extras_score(resume: ResumeInfo) -> float:
    """Up to 0.5 for certifications (2+ lines) and 0.5 for projects (3+ lines)."""
    certs = min(resume.n_certifications, 2) / 2
    projects = min(resume.n_projects, 3) / 3
    return 0.5 * certs + 0.5 * projects


# --------------------------------------------------------------- main API
@dataclass
class CandidateResult:
    rank: int
    filename: str
    name: str
    email: str
    phone: str
    score: float                 # 0..100
    semantic: float
    skills: float
    experience: float
    education: float
    extras: float
    experience_years: float
    education_level: str
    matched_skills: list[str]
    missing_skills: list[str]
    extra_skills: list[str]
    explanation: str
    shortlisted: bool = False


def explain(r: ResumeInfo, job: JobInfo, matched: list[str], missing: list[str], comps: dict[str, float]) -> str:
    parts = []
    n_req = len(job.required_skills)
    if n_req:
        n_m = len([s for s in matched if s in job.required_skills])
        parts.append(f"Matches {n_m}/{n_req} required skills.")
    if missing:
        parts.append("Missing: " + ", ".join(missing[:6]) + ("..." if len(missing) > 6 else "") + ".")
    if job.min_years > 0:
        verdict = "meets" if r.experience_years >= job.min_years else "below"
        parts.append(f"Experience {r.experience_years:g} yrs ({verdict} the {job.min_years:g}-yr requirement).")
    else:
        parts.append(f"Experience {r.experience_years:g} yrs.")
    if job.education_level:
        verdict = "meets" if r.education_level >= job.education_level else "below"
        parts.append(f"Education {EDU_LEVELS[r.education_level]} ({verdict} requirement: {EDU_LEVELS[job.education_level]}).")
    if len(r.skills) >= 8 and len(r.evidence_skills) / len(r.skills) < 0.3:
        parts.append(f"Caution: only {len(r.evidence_skills)} of {len(r.skills)} listed skills appear in experience/projects (possible keyword stuffing).")
    parts.append(f"Text similarity to JD: {comps['semantic'] * 100:.0f}%.")
    return " ".join(parts)


def score_candidates(
    job: JobInfo,
    resumes: list[ResumeInfo],
    scoring_texts: list[str],
    weights: Weights | None = None,
    use_embeddings: bool = False,
    top_n: int | None = None,
    threshold: float | None = None,
) -> tuple[list[CandidateResult], str]:
    w = (weights or Weights()).normalized()
    sem, engine = semantic_scores(job.text, scoring_texts, use_embeddings)

    results: list[CandidateResult] = []
    for r, s_sem in zip(resumes, sem):
        s_skill, matched, missing, extra = skill_score(r, job)
        s_exp = experience_score(r, job)
        s_edu = education_score(r, job)
        s_ext = extras_score(r)
        comps = {"semantic": s_sem, "skills": s_skill, "experience": s_exp, "education": s_edu, "extras": s_ext}
        total = (w.semantic * s_sem + w.skills * s_skill + w.experience * s_exp
                 + w.education * s_edu + w.extras * s_ext) * 100
        results.append(CandidateResult(
            rank=0, filename=r.filename, name=r.name, email=r.email, phone=r.phone,
            score=round(total, 1),
            semantic=round(s_sem * 100, 1), skills=round(s_skill * 100, 1),
            experience=round(s_exp * 100, 1), education=round(s_edu * 100, 1),
            extras=round(s_ext * 100, 1),
            experience_years=r.experience_years, education_level=EDU_LEVELS[r.education_level],
            matched_skills=matched, missing_skills=missing, extra_skills=extra,
            explanation=explain(r, job, matched, missing, comps),
        ))

    results.sort(key=lambda c: c.score, reverse=True)
    for i, c in enumerate(results, 1):
        c.rank = i
        if top_n is not None and threshold is not None:
            c.shortlisted = i <= top_n and c.score >= threshold
        elif top_n is not None:
            c.shortlisted = i <= top_n
        elif threshold is not None:
            c.shortlisted = c.score >= threshold
    return results, engine
