"""Streamlit dashboard.  Run with:  streamlit run app.py"""
import streamlit as st

from src.extractor import extract_text, extract_text_from_bytes
from src.pipeline import screen, to_dataframe
from src.scorer import Weights

st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
st.title("📄 AI-Based Resume Screening & Candidate Shortlisting")
st.caption("Decision-support tool: it ranks and explains, a human makes the final call.")

# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Shortlist top N", 1, 30, 5)
    threshold = st.slider("Minimum score to shortlist", 0, 100, 0)
    blind = st.checkbox("Blind screening (remove name, contact, gender, age before scoring)", value=True)
    use_emb = st.checkbox("Use sentence-transformers embeddings (optional install)", value=False)
    st.subheader("Score weights")
    w_sem = st.slider("Semantic similarity", 0.0, 1.0, 0.40, 0.05)
    w_skill = st.slider("Required skills", 0.0, 1.0, 0.30, 0.05)
    w_exp = st.slider("Experience", 0.0, 1.0, 0.15, 0.05)
    w_edu = st.slider("Education", 0.0, 1.0, 0.10, 0.05)
    w_ext = st.slider("Projects / certifications", 0.0, 1.0, 0.05, 0.05)

# --------------------------------------------------------------- inputs
col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Job description")
    jd_file = st.file_uploader("Upload JD (.txt, .pdf, .docx)", type=["txt", "pdf", "docx"], key="jd")
    jd_text = st.text_area("...or paste it here", height=220)
    if jd_file is not None:
        jd_text = extract_text_from_bytes(jd_file.getvalue(), jd_file.name)
with col2:
    st.subheader("2. Resumes")
    files = st.file_uploader("Upload resumes (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"],
                             accept_multiple_files=True, key="resumes")
    use_sample = st.checkbox("Use the bundled sample resumes + JD instead")

run = st.button("Screen candidates", type="primary")

if run:
    resumes = []
    if use_sample:
        from src.extractor import load_resumes_from_folder
        resumes = load_resumes_from_folder("data/sample_resumes")
        if not jd_text.strip():
            jd_text = extract_text("data/sample_jd.txt")
    else:
        for f in files or []:
            try:
                txt = extract_text_from_bytes(f.getvalue(), f.name)
                if len(txt) < 30:
                    st.warning(f"{f.name}: no readable text (scanned image?) - skipped")
                    continue
                resumes.append((f.name, txt))
            except Exception as exc:
                st.warning(f"Could not read {f.name}: {exc}")

    if not jd_text.strip():
        st.error("Please provide a job description.")
    elif not resumes:
        st.error("Please upload at least one resume (or tick the sample-data box).")
    else:
        weights = Weights(w_sem, w_skill, w_exp, w_edu, w_ext)
        with st.spinner("Parsing and scoring..."):
            results, engine = screen(jd_text, resumes, weights, use_emb, blind, top_n,
                                     threshold if threshold > 0 else None)
        df = to_dataframe(results)

        st.success(f"Scored {len(results)} resumes  |  engine: {engine}  |  blind screening: {'on' if blind else 'off'}")
        shortlisted = df[df["Shortlisted"] == "Yes"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Resumes screened", len(df))
        m2.metric("Shortlisted", len(shortlisted))
        m3.metric("Top score", f"{df['Score'].max():.1f}")

        st.subheader("Ranked candidates")
        st.bar_chart(df.set_index("Candidate")["Score"])
        st.dataframe(df[["Rank", "Shortlisted", "Candidate", "Score", "Semantic", "Skills",
                         "Experience", "Education", "Exp (yrs)", "Highest Degree"]], hide_index=True)

        st.subheader("Why each candidate ranked where they did")
        for r in results:
            badge = "✅" if r.shortlisted else "▫️"
            with st.expander(f"{badge} #{r.rank}  {r.name}  -  {r.score:.1f}"):
                st.write(r.explanation)
                c1, c2, c3 = st.columns(3)
                c1.markdown("**Matched skills**\n\n" + (", ".join(r.matched_skills) or "-"))
                c2.markdown("**Missing required skills**\n\n" + (", ".join(r.missing_skills) or "-"))
                c3.markdown("**Other skills**\n\n" + (", ".join(r.extra_skills) or "-"))
                st.progress(min(int(r.score), 100))
                st.caption(f"Semantic {r.semantic} | Skills {r.skills} | Experience {r.experience} | "
                           f"Education {r.education} | Extras {r.extras}  (each 0-100)")
                st.caption(f"{r.email}  {r.phone}  ({r.filename})")

        st.download_button("Download full report (CSV)", df.to_csv(index=False).encode("utf-8"),
                           "shortlist.csv", "text/csv")
        st.download_button("Download shortlist only (CSV)", shortlisted.to_csv(index=False).encode("utf-8"),
                           "shortlist_only.csv", "text/csv")
