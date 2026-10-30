🌐 Live Demo
🔗 Live Application: https://simran-001-ai-resume-screening-system-app-rxhd1y.streamlit.app/



# AI-Based Resume Screening & Candidate Shortlisting System

Upload a job description and a batch of resumes (PDF / DOCX / TXT). The system parses them, scores every
candidate with a transparent hybrid model, ranks them, shortlists the best, and **explains** each decision.
It is a decision-support tool: a human recruiter makes the final call.

## Features
- Resume ingestion: PDF (pdfplumber), DOCX (python-docx), TXT
- Information extraction: name, email, phone, skills (150+ synonyms), education level, years of experience, projects, certifications
- Job-description analysis: required vs "nice to have" skills, minimum years, education requirement
- Hybrid scoring (weights adjustable in the UI):
  semantic similarity 40% + required skills 30% + experience 15% + education 10% + projects/certs 5%
- Two similarity engines: TF-IDF (default, no download) or Sentence-BERT embeddings (optional)
- Blind screening: removes name, contact info, gender, age, DOB, pronouns before scoring
- Anti keyword-stuffing: skills that appear only in a bare skills list get reduced credit
- Explainable output: matched / missing skills and a plain-English reason per candidate
- Evaluation: precision@k, recall@k, NDCG@k against human-labelled rankings
- Streamlit dashboard + command-line tool + CSV export

## Project structure
```
resume_screening/
├── app.py                  # Streamlit dashboard
├── cli.py                  # Command-line runner
├── requirements.txt        # Required packages
├── requirements-optional.txt  # sentence-transformers (optional)
├── src/
│   ├── extractor.py        # PDF/DOCX/TXT -> text
│   ├── parser.py           # skills, education, experience, contact extraction
│   ├── skills_db.py        # skill dictionary (edit to add skills)
│   ├── anonymizer.py       # bias mitigation
│   ├── scorer.py           # hybrid scoring + explanations
│   ├── pipeline.py         # end-to-end screen()
│   └── evaluate.py         # precision@k, recall@k, NDCG@k
├── data/
│   ├── sample_jd.txt
│   ├── sample_resumes/     # 10 fictional resumes (txt, pdf, docx)
│   └── ground_truth.csv    # human relevance labels (0-3) for evaluation
├── tests/test_pipeline.py
└── .vscode/launch.json     # F5 run/debug configurations
```

## How to run in VS Code

### 1. Prerequisites
- Python 3.9 or newer (https://www.python.org/downloads/). On Windows, tick **"Add Python to PATH"** during install.
- VS Code (https://code.visualstudio.com/) with the **Python** extension (publisher: Microsoft).

### 2. Open the project
Unzip the folder, then in VS Code: **File > Open Folder...** and choose `resume_screening`.
Open the built-in terminal with **Ctrl + `** (backtick) (Terminal > New Terminal).

### 3. Create a virtual environment
Windows (PowerShell):
```
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell says scripts are disabled, run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
(or use Command Prompt and run `.venv\Scripts\activate.bat`).

macOS / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```
You should see `(.venv)` at the start of the prompt. If VS Code asks "We noticed a new environment", click **Yes**.
Otherwise press **Ctrl+Shift+P > "Python: Select Interpreter"** and pick the one inside `.venv`.

### 4. Install dependencies
```
pip install -r requirements.txt
```

### 5. Run the web dashboard
```
streamlit run app.py
```
Your browser opens at http://localhost:8501. Tick **"Use the bundled sample resumes + JD"** and click
**Screen candidates**, or upload your own JD and resumes.
(If Streamlit asks for an email, just press Enter.) Stop the server with **Ctrl+C**.

### 6. Or run from the command line
```
python cli.py --jd data/sample_jd.txt --resumes data/sample_resumes --top 5 --evaluate data/ground_truth.csv
```
Results are printed and saved to `outputs/shortlist.csv`.
Use your own data with `--jd my_jd.txt --resumes my_resumes_folder`.

Useful options: `--threshold 50` (min score), `--no-blind` (turn off anonymization), `--embeddings` (see below).

### 7. Run the tests
```
python -m unittest discover -s tests -v
```

### 8. Run / debug with F5
Open the Run and Debug panel (Ctrl+Shift+D) and choose **"Run CLI on sample data"** or **"Run Streamlit app"**.

## Optional: better semantic matching with Sentence-BERT
```
pip install -r requirements-optional.txt
python cli.py --jd data/sample_jd.txt --resumes data/sample_resumes --embeddings
```
In the dashboard, tick **"Use sentence-transformers embeddings"**. The first run downloads a ~90 MB model, so it
needs internet once. If the package or model is unavailable the app automatically falls back to TF-IDF.

## How scoring works
| Component | Weight | How it is computed |
|---|---|---|
| Semantic similarity | 40% | Cosine similarity of resume vs JD (TF-IDF or SBERT), scaled to 0-100 |
| Required skills | 30% | Matched required skills / required skills (+ half weight for "nice to have"). Skills not backed by experience/projects get 60% credit |
| Experience | 15% | min(1, candidate years / required years). Years come from the Experience section date ranges (overlaps merged) or explicit "N years" statements |
| Education | 10% | Full marks if the candidate's highest degree meets the JD requirement, else proportional |
| Projects / certifications | 5% | Up to 50% for 2+ certifications and 50% for 3+ project lines |

All weights can be changed in the dashboard sidebar and are re-normalised to sum to 1.

## Adapting it
- **Add skills:** edit `src/skills_db.py` (canonical name -> list of spellings).
- **JD format:** put optional skills under a heading such as "Nice to have" or "Preferred" and they are treated as optional.
- **Resume format:** works best when resumes have headings like Skills, Experience, Education, Projects.
- **Scanned (image-only) PDFs** have no text layer and are skipped with a warning. To support them, add OCR with
  `pytesseract` + `pdf2image` (requires installing Tesseract) inside `src/extractor.py`.

## Evaluation
`data/ground_truth.csv` holds human relevance labels (0 = irrelevant ... 3 = excellent) for the sample resumes.
`--evaluate` reports precision@k, recall@k and NDCG@k. For your report, label your own set of resumes
(ideally by 2-3 reviewers) and compare, then try different weights and TF-IDF vs SBERT.

## Limitations (mention these in your report)
- Rule-based parsing misses unusual resume layouts, tables and multi-column PDFs.
- Skill matching only knows the skills in `skills_db.py`.
- Keyword stuffing is mitigated, not eliminated.
- Blind screening reduces bias but proxies (university name, gaps, location) can remain. Audit regularly.
- Automated hiring tools are regulated in some regions (e.g. EU AI Act, NYC Local Law 144). Use as decision support with human review.
- The sample resumes are fictional; all names and details are made up.

## Troubleshooting
| Problem | Fix |
|---|---|
| `python` not found | Windows: use `py -m venv .venv`; macOS/Linux: use `python3` |
| `streamlit` not recognised | Activate the venv first, or run `python -m streamlit run app.py` |
| `ModuleNotFoundError: src` | Run commands from the project root folder (the one containing `app.py`) |
| "no readable text" warning | The PDF is a scanned image; see the OCR note above |
| Port 8501 busy | `streamlit run app.py --server.port 8502` |
