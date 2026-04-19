from __future__ import annotations

from fpdf import FPDF


def _normalize_text_for_pdf(text: str) -> str:
    """
    Keep PDF text robust with core Helvetica font.
    Replaces common unicode punctuation and strips unsupported chars.
    """
    normalized = (
        text.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
    )
    # Core fonts support latin-1; replace unsupported chars to avoid crashes.
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


def _break_long_tokens(text: str, max_token_len: int = 60) -> str:
    """
    Insert spaces inside very long unbroken tokens (e.g., URLs) so FPDF can wrap lines.
    """
    out: list[str] = []
    for token in text.split(" "):
        if len(token) <= max_token_len:
            out.append(token)
            continue
        chunks = [token[i : i + max_token_len] for i in range(0, len(token), max_token_len)]
        out.append(" ".join(chunks))
    return " ".join(out)


def report_markdown_to_pdf_bytes(title: str, markdown_text: str) -> bytes:
    """
    Minimal Markdown-to-PDF: we render as wrapped plain text to keep deps tiny.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 14)
    pdf.multi_cell(0, 8, title)
    pdf.ln(2)

    pdf.set_font("Helvetica", "", 11)
    for line in markdown_text.splitlines():
        safe = line.replace("\t", "    ")
        safe = _normalize_text_for_pdf(safe)
        safe = _break_long_tokens(safe)
        pdf.multi_cell(0, 6, safe)

    return bytes(pdf.output(dest="S"))

