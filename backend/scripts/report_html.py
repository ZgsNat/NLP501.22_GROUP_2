"""docs/FINAL_REPORT.md → print-ready HTML with the SVG figures inlined; print it to PDF with a browser.

    uv run --with markdown python scripts/report_html.py ../docs/FINAL_REPORT.md /tmp/report.html
    "<Chrome>" --headless=new --print-to-pdf=../docs/FINAL_REPORT.pdf --no-pdf-header-footer file:///tmp/report.html
"""
import sys, markdown, re, pathlib
src = pathlib.Path(sys.argv[1]); out = pathlib.Path(sys.argv[2])


def nested_lists(md: str) -> str:
    """Python-Markdown nests a list only at 4-space indents; the report uses 2 (under "- ") and 3 (under "1. "), which
    it would flatten into the parent list. Re-indent list items, outside code fences, to 4 spaces per level."""
    out, fence = [], False
    for line in md.splitlines():
        fence ^= line.lstrip().startswith("```")
        m = None if fence else re.match(r"^( +)([-*+]|\d+\.) ", line)
        out.append(" " * 4 * ((len(m.group(1)) + 3) // 4) + line.lstrip() if m else line)
    return "\n".join(out)


body = markdown.markdown(nested_lists(src.read_text(encoding="utf-8")), extensions=["tables", "fenced_code"])
body = re.sub(r'<img alt="[^"]*" src="figures/([^"]+\.svg)" */?>',
              lambda m: (src.parent / "figures" / m.group(1)).read_text(encoding="utf-8").replace("<svg ", "<svg style='max-width:100%;height:auto' ", 1), body)
css = """
@page { size: A4; margin: 18mm 16mm; }
body { font: 10.5pt/1.45 -apple-system, "Segoe UI", system-ui, sans-serif; color: #1c1b18; max-width: 780px; margin: 0 auto; }
h1 { font-size: 20pt; margin: 0 0 6pt; } h2 { font-size: 14pt; margin: 18pt 0 6pt; border-bottom: 1px solid #dedbd2; padding-bottom: 3pt; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; } p, li { margin: 4pt 0; }
table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 6pt 0 10pt; page-break-inside: avoid; }
th, td { border-bottom: 1px solid #dedbd2; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #f3f2ee; } code { font-size: 8.8pt; background: #f3f2ee; padding: 0 2pt; }
pre { background: #f6f5f1; padding: 8pt; font-size: 8pt; white-space: pre-wrap; page-break-inside: avoid; }
img { max-width: 100%; page-break-inside: avoid; } em { color: #5f5e5a; }
"""
out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Final report · NLP501 Group 2</title><style>{css}</style></head><body>{body}</body></html>", encoding="utf-8")
