import pytest
import pymupdf

from ac_rag.loaders import extract_document, extract_pdf_pages, render_highlighted_page


def test_text_upload_decodes_bom():
    assert extract_document("memo.txt", b"\xef\xbb\xbfA short memo.") == "A short memo."


def test_empty_document_is_rejected():
    with pytest.raises(ValueError, match="не содержит"):
        extract_document("memo.txt", b"  ")


def test_unsupported_file_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        extract_document("memo.exe", b"content")


def test_pdf_page_extraction_and_highlight_render():
    pdf = pymupdf.open()
    pdf.new_page().insert_text((60, 80), "First page only.")
    pdf.new_page().insert_text((60, 80), "The launch date is 14 November 2026.")
    content = pdf.tobytes()
    pdf.close()
    assert extract_pdf_pages(content)[1].strip() == "The launch date is 14 November 2026."
    image, highlighted = render_highlighted_page(content, 2, "The launch date is 14 November 2026.")
    assert image.startswith(b"\x89PNG")
    assert highlighted
