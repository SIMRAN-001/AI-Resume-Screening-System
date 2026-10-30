"""Step 1 - Resume ingestion: turn PDF / DOCX / TXT files into plain text."""
from __future__ import annotations

import io
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def _read_pdf(data: bytes) -> str:
    import pdfplumber

    pages = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _read_docx(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    parts = [p.text for p in document.paragraphs]
    # resumes often put content in tables
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return "\n".join(parts)


def _read_txt(data: bytes) -> str:
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeError:
            continue
    return data.decode("utf-8", errors="ignore")


def extract_text_from_bytes(data: bytes, filename: str) -> str:
    """Extract text from raw file bytes (used by the Streamlit uploader)."""
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        text = _read_pdf(data)
    elif ext == ".docx":
        text = _read_docx(data)
    elif ext == ".txt":
        text = _read_txt(data)
    else:
        raise ValueError(f"Unsupported file type: {ext} ({filename})")
    return text.strip()


def extract_text(path: str | Path) -> str:
    """Extract text from a file on disk."""
    path = Path(path)
    return extract_text_from_bytes(path.read_bytes(), path.name)


def load_resumes_from_folder(folder: str | Path) -> list[tuple[str, str]]:
    """Return [(filename, text), ...] for every supported file in a folder."""
    folder = Path(folder)
    resumes = []
    for f in sorted(folder.iterdir()):
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                text = extract_text(f)
            except Exception as exc:  # corrupt file, etc.
                print(f"[warn] could not read {f.name}: {exc}")
                continue
            if len(text) < 30:
                print(f"[warn] {f.name} has almost no text (scanned image? needs OCR) - skipped")
                continue
            resumes.append((f.name, text))
    return resumes
