from __future__ import annotations

import re
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
    # Remove zero-width and non-breaking whitespace that can break wrapping.
    normalized = (
        normalized.replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
    )
    # Collapse control chars except newline/tab.
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", normalized)
    # Core fonts support latin-1; replace unsupported chars to avoid crashes.
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


def _break_long_tokens(text: str, max_token_len: int = 40) -> str:
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


def _safe_wrapped_lines(pdf: FPDF, text: str, max_width: float) -> list[str]:
    """
    Wrap text with explicit width checks to avoid fpdf line-break edge cases.
    """
    if not text:
        return [""]

    words = text.split(" ")
    lines: list[str] = []
    current = ""

    for word in words:
        candidate = word if not current else f"{current} {word}"
        if pdf.get_string_width(candidate) <= max_width:
            current = candidate
            continue

        if current:
            lines.append(current)
            current = ""

        # Word itself may still exceed line width; split by characters.
        if pdf.get_string_width(word) <= max_width:
            current = word
            continue

        chunk = ""
        for ch in word:
            probe = f"{chunk}{ch}"
            if probe and pdf.get_string_width(probe) <= max_width:
                chunk = probe
            else:
                if chunk:
                    lines.append(chunk)
                chunk = ch
        current = chunk

    if current or not lines:
        lines.append(current)
    return lines


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
    max_width = pdf.w - pdf.l_margin - pdf.r_margin
    for line in markdown_text.splitlines():
        safe = line.replace("\t", "    ")
        safe = _normalize_text_for_pdf(safe)
        safe = _break_long_tokens(safe)
        wrapped = _safe_wrapped_lines(pdf, safe, max_width=max_width)
        for wrapped_line in wrapped:
            pdf.cell(0, 6, wrapped_line, ln=True)

    return bytes(pdf.output(dest="S"))

