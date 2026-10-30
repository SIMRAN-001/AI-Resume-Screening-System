"""Command-line interface.

Example:
    python cli.py --jd data/sample_jd.txt --resumes data/sample_resumes --top 5
"""
import argparse
import csv
from pathlib import Path

from src.evaluate import evaluate
from src.extractor import extract_text, load_resumes_from_folder
from src.pipeline import screen, to_dataframe
from src.scorer import Weights


def main():
    ap = argparse.ArgumentParser(description="AI-based resume screening & shortlisting")
    ap.add_argument("--jd", required=True, help="Path to job description (.txt/.pdf/.docx)")
    ap.add_argument("--resumes", required=True, help="Folder containing resumes (.pdf/.docx/.txt)")
    ap.add_argument("--top", type=int, default=5, help="Shortlist the top N candidates")
    ap.add_argument("--threshold", type=float, default=None, help="Minimum score (0-100) to shortlist")
    ap.add_argument("--embeddings", action="store_true", help="Use sentence-transformers (needs optional install)")
    ap.add_argument("--no-blind", action="store_true", help="Disable anonymization before scoring")
    ap.add_argument("--out", default="outputs/shortlist.csv", help="CSV output path")
    ap.add_argument("--evaluate", default=None, help="CSV with columns filename,relevance to compute metrics")
    args = ap.parse_args()

    jd_text = extract_text(args.jd)
    resumes = load_resumes_from_folder(args.resumes)
    if not resumes:
        raise SystemExit("No readable resumes found in the folder.")

    results, engine = screen(
        jd_text, resumes, Weights(), use_embeddings=args.embeddings,
        blind=not args.no_blind, top_n=args.top, threshold=args.threshold,
    )

    print(f"\nScored {len(results)} resumes | similarity engine: {engine} | blind scoring: {not args.no_blind}\n")
    print(f"{'Rank':<5}{'Candidate':<20}{'Score':>6}  {'Skills':>6} {'Exp':>5} {'Edu':>5}  Shortlisted")
    print("-" * 72)
    for r in results:
        print(f"{r.rank:<5}{r.name[:19]:<20}{r.score:>6.1f}  {r.skills:>6.0f} {r.experience_years:>4g}y {r.education_level[:5]:>5}  "
              f"{'YES' if r.shortlisted else '-'}")
    print()
    for r in results[: args.top]:
        print(f"#{r.rank} {r.name}: {r.explanation}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    to_dataframe(results).to_csv(out, index=False)
    print(f"\nSaved full report to {out}")

    if args.evaluate:
        with open(args.evaluate, newline="", encoding="utf-8") as f:
            relevance = {row["filename"]: int(row["relevance"]) for row in csv.DictReader(f)}
        metrics = evaluate([r.filename for r in results], relevance, k=min(args.top, len(results)))
        print("Evaluation vs human labels:", metrics)


if __name__ == "__main__":
    main()
