"""Typeset the report: Markdown exported from the report doc -> A4 PDF (headless Chrome) + DOCX (pandoc).

Usage: python src/build_report.py report/report.md
  - images in the Markdown may point to blob/... ids from the doc; they are mapped to figures/*.png
    by their caption ("Figure 1." -> FIGURES[0], ...)
  - writes report/<stem>.html, report/<stem>.pdf and report/<stem>.docx
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FIGURES = [ROOT / "figures/fig_learning_curve.png", ROOT / "figures/fig_confusion.png"]


def prepare(md: str) -> str:
    # doc chips arrive as "Oct 12, 2026 · @Name": drop the "@"
    md = re.sub(r"^(\w{3} \d{1,2}, \d{4}) · @(\S+)", r"\2 · \1", md, count=1, flags=re.M)

    # images: point "Figure N." captions at the local PNGs
    def image(m):
        caption = m.group(1)
        n = re.match(r"Figure (\d+)", caption)
        path = FIGURES[int(n.group(1)) - 1] if n else m.group(2)
        return f"![{caption}]({path})"
    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image, md)
    # italic "Table N. ..." lines become captions
    md = re.sub(r"^\*(Table \d+\..*)\*\s*$", r"::: caption\n\1\n:::", md, flags=re.M)
    return md


def main():
    src = Path(sys.argv[1])
    stem = src.stem
    md = prepare(src.read_text())
    tmp = REPORT / f"{stem}.prepared.md"
    tmp.write_text(md)
    html = REPORT / f"{stem}.html"
    subprocess.run(["pandoc", str(tmp), "-f", "markdown+fenced_divs", "-t", "html5", "-s", "--embed-resources",
                    "--css", str(REPORT / "style.css"), "--metadata", "pagetitle=Bullish or Bearish?",
                    "-o", str(html)], check=True)
    # pandoc wraps fenced divs as <div class="caption"><p>..</p></div>; style them as captions
    text = html.read_text().replace('<div class="caption">\n<p>', '<div class="caption">\n<p class="caption">')
    html.write_text(text)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={REPORT / f'{stem}.pdf'}", html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pandoc", str(tmp), "-f", "markdown+fenced_divs", "-t", "docx",
                    "--resource-path", str(ROOT), "-o", str(REPORT / f"{stem}.docx")], check=True)
    tmp.unlink()
    print("wrote", REPORT / f"{stem}.pdf", "and", REPORT / f"{stem}.docx")


if __name__ == "__main__":
    main()
