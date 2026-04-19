from __future__ import annotations

from fpdf import FPDF


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
        # basic cleanup
        safe = line.replace("\t", "    ")
        pdf.multi_cell(0, 6, safe)

    return bytes(pdf.output(dest="S"))

