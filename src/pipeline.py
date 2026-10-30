"""End-to-end pipeline: raw texts in -> ranked, explained shortlist out."""
from __future__ import annotations

import pandas as pd

from .anonymizer import anonymize
from .parser import parse_job_description, parse_resume
from .scorer import CandidateResult, Weights, score_candidates


def screen(
    jd_text: str,
    resumes: list[tuple[str, str]],
    weights: Weights | None = None,
    use_embeddings: bool = False,
    blind: bool = True,
    top_n: int | None = 5,
    threshold: float | None = None,
) -> tuple[list[CandidateResult], str]:
    """resumes: list of (filename, text). Returns (results sorted best-first, engine name)."""
    job = parse_job_description(jd_text)
    infos = [parse_resume(fn, txt) for fn, txt in resumes]
    texts = [anonymize(i.text, i.name) if blind else i.text for i in infos]
    return score_candidates(job, infos, texts, weights, use_embeddings, top_n, threshold)


def to_dataframe(results: list[CandidateResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Rank": r.rank, "Shortlisted": "Yes" if r.shortlisted else "No",
            "Candidate": r.name, "File": r.filename, "Email": r.email, "Phone": r.phone,
            "Score": r.score, "Semantic": r.semantic, "Skills": r.skills,
            "Experience": r.experience, "Education": r.education, "Extras": r.extras,
            "Exp (yrs)": r.experience_years, "Highest Degree": r.education_level,
            "Matched Skills": ", ".join(r.matched_skills),
            "Missing Skills": ", ".join(r.missing_skills),
            "Explanation": r.explanation,
        })
    return pd.DataFrame(rows)
