import pytest

from ac_rag.loaders import extract_document


def test_text_upload_decodes_bom():
    assert extract_document("memo.txt", b"\xef\xbb\xbfA short memo.") == "A short memo."


def test_empty_document_is_rejected():
    with pytest.raises(ValueError, match="не содержит"):
        extract_document("memo.txt", b"  ")


def test_unsupported_file_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        extract_document("memo.exe", b"content")
