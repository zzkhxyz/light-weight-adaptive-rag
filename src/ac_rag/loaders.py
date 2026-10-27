from pathlib import Path


def load_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md"}:
        text = path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    elif suffix == ".docx":
        from docx import Document
        text = "\n".join(paragraph.text for paragraph in Document(path).paragraphs)
    else:
        raise ValueError(f"Unsupported document format: {suffix}")
    if not text.strip():
        raise ValueError(f"No extractable text in {path}; scanned files need OCR first")
    return text
