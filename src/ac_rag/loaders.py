from io import BytesIO
from pathlib import Path


def load_document(path: Path) -> str:
    return extract_document(path.name, path.read_bytes())


def extract_document(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        text = content.decode("utf-8-sig")
    elif suffix == ".pdf":
        from pypdf import PdfReader
        text = "\n".join(page.extract_text() or "" for page in PdfReader(BytesIO(content)).pages)
    elif suffix == ".docx":
        from docx import Document
        text = "\n".join(paragraph.text for paragraph in Document(BytesIO(content)).paragraphs)
    else:
        raise ValueError(f"Unsupported document format: {suffix}")
    if not text.strip():
        raise ValueError("Документ не содержит доступного текста. Для сканированного PDF сначала выполните OCR.")
    return text
