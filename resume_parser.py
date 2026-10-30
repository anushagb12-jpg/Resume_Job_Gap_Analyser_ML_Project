"""Read a resume file and return the plain text inside it.

Streamlit hands us an UploadedFile object, which behaves like an open Python
file: it has a `.name` (with the extension) and a `.read()` method that gives
back raw bytes. This module turns those bytes into a string.

Supported formats
-----------------
.pdf  -> parsed page by page with PyMuPDF
.txt  -> decoded as text
.md   -> decoded as text (Markdown resumes work fine)
"""

from __future__ import annotations

from typing import Any

import pymupdf

# File extensions we know how to handle. The keys are what the app's file
# uploader accepts, and `SUPPORTED_EXTENSIONS` is also used for error messages.
TEXT_EXTENSIONS = (".txt", ".md", ".markdown", ".rst", ".csv")
PDF_EXTENSION = ".pdf"
SUPPORTED_EXTENSIONS = (PDF_EXTENSION, ".txt", ".md")


class ResumeParseError(Exception):
    """Raised when a resume cannot be read.

    We use our own exception type so the app can catch exactly this problem
    and show a friendly message, instead of catching every possible error and
    hiding real bugs.
    """


def get_extension(filename: str) -> str:
    """Return the lowercase file extension, e.g. 'Resume.PDF' -> '.pdf'."""
    if "." not in filename:
        return ""
    return "." + filename.rsplit(".", 1)[-1].lower()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Pull the text out of every page of a PDF and join it with newlines.

    Raises ResumeParseError for encrypted or damaged files, because those
    really cannot be read and the user needs to be told.
    """
    if not file_bytes:
        raise ResumeParseError("The PDF file is empty (0 bytes).")

    try:
        document = pymupdf.open(stream=file_bytes, filetype="pdf")
    except Exception as error:  # PyMuPDF raises several different error types
        raise ResumeParseError(f"Could not open the PDF: {error}") from error

    # `with` makes sure the document is closed even if a page fails to read.
    with document:
        if document.needs_pass:
            raise ResumeParseError(
                "This PDF is password protected. Please upload an unlocked copy."
            )

        pages = []
        for page in document:
            # page.get_text() returns the text in reading order, including the
            # line breaks, which is what we want.
            pages.append(page.get_text())

    text = "\n".join(pages).strip()
    if not text:
        raise ResumeParseError(
            "No selectable text was found in this PDF. It may be a scanned "
            "image - try exporting it as text, or paste your resume instead."
        )
    return text


def extract_text_from_text_file(file_bytes: bytes) -> str:
    """Decode plain-text bytes into a string.

    We try UTF-8 first (the modern default) and fall back to Windows-1252,
    which is what many older word processors produce. `errors="replace"` on the
    fallback guarantees we always return *something* rather than crashing on a
    single odd byte.
    """
    if not file_bytes:
        raise ResumeParseError("The text file is empty (0 bytes).")

    try:
        return file_bytes.decode("utf-8").strip()
    except UnicodeDecodeError:
        return file_bytes.decode("windows-1252", errors="replace").strip()


def extract_text_from_bytes(file_bytes: bytes, filename: str) -> str:
    """Extract text from raw bytes, picking a parser based on the file name.

    This is the core of the module. It takes bytes rather than a file object so
    it is easy to cache, easy to test, and independent of Streamlit.
    """
    extension = get_extension(filename)

    if extension != PDF_EXTENSION and extension not in TEXT_EXTENSIONS:
        supported = ", ".join(SUPPORTED_EXTENSIONS)
        raise ResumeParseError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            f"Please upload one of: {supported}."
        )

    if extension == PDF_EXTENSION:
        return extract_text_from_pdf(file_bytes)
    return extract_text_from_text_file(file_bytes)


def extract_text(uploaded_file: Any) -> str:
    """Extract text from a Streamlit UploadedFile (or any file-like object).

    This is the function the app calls.
    """
    filename = getattr(uploaded_file, "name", "") or ""
    return extract_text_from_bytes(_read_bytes(uploaded_file), filename)


def extract_text_from_path(file_path: str) -> str:
    """Convenience helper: extract text from a file path on disk.

    Useful for testing the pipeline from a script without going through
    Streamlit's upload widget.
    """
    with open(file_path, "rb") as handle:
        return extract_text_from_bytes(handle.read(), file_path)


def _read_bytes(uploaded_file: Any) -> bytes:
    """Read all bytes from a file-like object, rewinding it first if possible."""
    # Streamlit file objects may already have been read once, so go back to the
    # start. Plain files and BytesIO objects support seek() too.
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    return uploaded_file.read()
