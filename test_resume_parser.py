"""Tests for src/resume_parser.py.

The PDF tests build a real PDF in memory with PyMuPDF and then read it back, so
the whole round trip is covered without any file needing to be committed.
"""

from __future__ import annotations

import io

import pymupdf
import pytest

from src.resume_parser import (
    SUPPORTED_EXTENSIONS,
    ResumeParseError,
    extract_text,
    extract_text_from_bytes,
    extract_text_from_path,
    extract_text_from_pdf,
    extract_text_from_text_file,
    get_extension,
)


def make_pdf(text: str, pages: int = 1) -> bytes:
    """Build a real PDF containing `text` on every page and return its bytes."""
    document = pymupdf.open()
    for index in range(pages):
        page = document.new_page()
        suffix = "" if pages == 1 else f" (page {index + 1})"
        page.insert_text((72, 100), text + suffix)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


class FakeUpload:
    """Stands in for a Streamlit UploadedFile, which has .name and .read()."""

    def __init__(self, data: bytes, name: str) -> None:
        self.name = name
        self._stream = io.BytesIO(data)

    def read(self) -> bytes:
        return self._stream.read()

    def seek(self, offset: int) -> None:
        self._stream.seek(offset)


# ---------------------------------------------------------------------------
# File extensions
# ---------------------------------------------------------------------------


def test_get_extension_lowercases():
    assert get_extension("Resume.PDF") == ".pdf"


def test_get_extension_with_dots_in_the_name():
    assert get_extension("alex.morgan.resume.pdf") == ".pdf"


def test_get_extension_without_a_dot():
    assert get_extension("resume") == ""


# ---------------------------------------------------------------------------
# Plain text files
# ---------------------------------------------------------------------------


def test_reads_utf8_text():
    data = "Python, SQL and Docker".encode("utf-8")
    assert extract_text_from_text_file(data) == "Python, SQL and Docker"


def test_reads_text_with_unicode_characters():
    text = "José García - Analista de datos"
    assert extract_text_from_text_file(text.encode("utf-8")) == text


def test_falls_back_to_windows_encoding():
    # 0x92 is a curly apostrophe in Windows-1252 but is not valid UTF-8.
    data = b"it\x92s a resume"
    assert "it" in extract_text_from_text_file(data)


def test_empty_text_file_raises():
    with pytest.raises(ResumeParseError, match="empty"):
        extract_text_from_text_file(b"")


# ---------------------------------------------------------------------------
# PDF files
# ---------------------------------------------------------------------------


def test_pdf_round_trip():
    text = "Python, pandas and SQL"
    assert text in extract_text_from_pdf(make_pdf(text))


def test_pdf_with_several_pages_reads_them_all():
    extracted = extract_text_from_pdf(make_pdf("Data Analyst", pages=3))
    assert "page 1" in extracted
    assert "page 3" in extracted


def test_pdf_without_any_text_raises_a_helpful_error():
    document = pymupdf.open()
    document.new_page()  # a blank page - nothing to read
    blank_pdf = document.tobytes()
    document.close()

    with pytest.raises(ResumeParseError, match="No selectable text"):
        extract_text_from_pdf(blank_pdf)


def test_empty_pdf_bytes_raise():
    with pytest.raises(ResumeParseError, match="empty"):
        extract_text_from_pdf(b"")


def test_password_protected_pdf_raises():
    document = pymupdf.open()
    document.new_page().insert_text((72, 100), "secret skills")
    locked = document.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="hunter2", owner_pw="hunter2"
    )
    document.close()

    with pytest.raises(ResumeParseError, match="password"):
        extract_text_from_pdf(locked)


def test_damaged_pdf_raises():
    with pytest.raises(ResumeParseError):
        extract_text_from_pdf(b"%PDF-1.4 this is not really a pdf")


# ---------------------------------------------------------------------------
# Dispatching on the file name
# ---------------------------------------------------------------------------


def test_bytes_dispatch_picks_the_pdf_parser():
    text = "Kubernetes and Terraform"
    assert text in extract_text_from_bytes(make_pdf(text), "resume.pdf")


def test_bytes_dispatch_picks_the_text_parser():
    assert extract_text_from_bytes(b"Kubernetes", "notes.txt") == "Kubernetes"


def test_bytes_dispatch_accepts_markdown():
    assert extract_text_from_bytes(b"# Skills\nPython", "resume.md") == "# Skills\nPython"


def test_unsupported_extension_raises_and_lists_the_supported_ones():
    with pytest.raises(ResumeParseError) as error:
        extract_text_from_bytes(b"anything", "resume.docx")
    message = str(error.value)
    assert ".docx" in message
    for extension in SUPPORTED_EXTENSIONS:
        assert extension in message


def test_missing_extension_raises():
    with pytest.raises(ResumeParseError, match="Unsupported file type"):
        extract_text_from_bytes(b"anything", "resume")


def test_extract_text_reads_a_streamlit_style_upload():
    uploaded = FakeUpload(make_pdf("Tableau and Power BI"), "resume.pdf")
    extracted = extract_text(uploaded)
    assert "Tableau" in extracted


def test_extract_text_can_be_called_twice_on_the_same_upload():
    # Streamlit reuses the same object across reruns, so the stream has to be
    # rewound before each read.
    uploaded = FakeUpload(b"python sql", "resume.txt")
    assert extract_text(uploaded) == "python sql"
    assert extract_text(uploaded) == "python sql"


def test_extract_text_rejects_an_unsupported_upload():
    with pytest.raises(ResumeParseError):
        extract_text(FakeUpload(b"not a pdf", "resume.pdf"))


def test_extract_text_from_path(tmp_path):
    path = tmp_path / "resume.txt"
    path.write_text("python and sql", encoding="utf-8")
    assert extract_text_from_path(str(path)) == "python and sql"


def test_extract_text_from_path_reads_a_pdf(tmp_path):
    path = tmp_path / "resume.pdf"
    path.write_bytes(make_pdf("Docker and AWS"))
    assert "Docker" in extract_text_from_path(str(path))
