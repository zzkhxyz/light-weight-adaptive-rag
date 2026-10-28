from pathlib import Path

import pymupdf


def load_document(path: Path) -> str:
    return extract_document(path.name, path.read_bytes())


def extract_document(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        text = content.decode("utf-8-sig")
    elif suffix == ".pdf":
        text = "\n".join(extract_pdf_pages(content))
    elif suffix == ".docx":
        from docx import Document
        from io import BytesIO
        text = "\n".join(paragraph.text for paragraph in Document(BytesIO(content)).paragraphs)
    else:
        raise ValueError(f"Unsupported document format: {suffix}")
    if not text.strip():
        raise ValueError("Документ не содержит доступного текста. Для сканированного PDF сначала выполните OCR.")
    return text


def extract_pdf_pages(content: bytes) -> list[str]:
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        return [page.get_text() for page in pdf]


def render_highlighted_page(content: bytes, page_number: int, snippet: str) -> tuple[bytes, bool]:
    with pymupdf.open(stream=content, filetype="pdf") as pdf:
        if page_number < 1 or page_number > len(pdf):
            raise ValueError("Page not found")
        page = pdf[page_number - 1]
        words = snippet.split()
        matches = page.search_for(snippet)
        for size in (18, 12, 8, 5):
            if matches:
                break
            if len(words) < size:
                continue
            matches = page.search_for(" ".join(words[:size]))
            if matches:
                break
        if not matches and words:
            matches = page.search_for(" ".join(words[:min(4, len(words))]))
        if matches:
            annotation = page.add_highlight_annot(matches)
            annotation.set_colors(stroke=(0.27, 0.81, 0.49))
            annotation.update(opacity=0.38)
        return page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).tobytes("png"), bool(matches)
