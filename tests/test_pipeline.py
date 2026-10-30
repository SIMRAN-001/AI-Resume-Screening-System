"""Run with:  python -m unittest discover -s tests -v"""
import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.anonymizer import anonymize
from src.evaluate import evaluate
from src.extractor import extract_text, load_resumes_from_folder
from src.parser import extract_skills, extract_education_level, parse_job_description, parse_resume
from src.pipeline import screen


class TestParser(unittest.TestCase):
    def test_skill_boundaries(self):
        self.assertIn("javascript", extract_skills("I know JavaScript"))
        self.assertNotIn("java", extract_skills("I know JavaScript"))
        self.assertIn("c++", extract_skills("Experienced in C++ and C#"))
        self.assertIn("machine learning", extract_skills("ML engineer"))

    def test_education(self):
        self.assertEqual(extract_education_level("M.Tech in CS"), 3)
        self.assertEqual(extract_education_level("PhD in Statistics"), 4)
        self.assertEqual(extract_education_level("B.Tech"), 2)

    def test_experience_from_dates(self):
        text = "Name Here\n\nEXPERIENCE\nDev, X  2019 - 2022\nDev, Y  2021 - 2023\n\nEDUCATION\nB.Tech 2015 - 2019"
        info = parse_resume("a.txt", text)
        self.assertEqual(info.experience_years, 4.0)  # 2019-2023 merged, education dates ignored

    def test_job_description(self):
        job = parse_job_description(extract_text(ROOT / "data/sample_jd.txt"))
        self.assertIn("python", job.required_skills)
        self.assertIn("sql", job.required_skills)
        self.assertIn("aws", job.preferred_skills)
        self.assertEqual(job.min_years, 3.0)


class TestAnonymizer(unittest.TestCase):
    def test_removes_identifiers(self):
        out = anonymize("Jane Doe\njane@x.com\nGender: Female\nShe built models in Python.", "Jane Doe")
        for bad in ("Jane", "jane@x.com", "Female", "She"):
            self.assertNotIn(bad, out)
        self.assertIn("Python", out)


class TestEndToEnd(unittest.TestCase):
    def test_sample_ranking(self):
        jd = extract_text(ROOT / "data/sample_jd.txt")
        resumes = load_resumes_from_folder(ROOT / "data/sample_resumes")
        self.assertEqual(len(resumes), 10)
        results, _ = screen(jd, resumes, top_n=5)
        ranked = [r.filename for r in results]
        # irrelevant profiles must not be in the top 5
        self.assertNotIn("05_karan_singh.txt", ranked[:5])
        self.assertNotIn("07_vikram_joshi.txt", ranked[:5])
        # keyword-stuffed resume (skills list only, no experience) must be penalised
        self.assertNotIn("09_rahul_khanna.txt", ranked[:5])
        with open(ROOT / "data/ground_truth.csv", newline="") as f:
            rel = {r["filename"]: int(r["relevance"]) for r in csv.DictReader(f)}
        m = evaluate(ranked, rel, k=5)
        self.assertGreaterEqual(m["precision@5"], 0.6)
        self.assertGreaterEqual(m["ndcg@5"], 0.8)


if __name__ == "__main__":
    unittest.main()
