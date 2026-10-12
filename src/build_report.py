"""Typeset the report: Markdown exported from the report doc -> A4 PDF (headless Chrome) + DOCX (pandoc).

Usage: python src/build_report.py report/report.md [--author "Name Surname"]
  - images may point to blob/... ids from the doc; an image whose alt text starts with
    "Figure N." is mapped to FIGURES[N-1]
  - italic lines starting with "Table N." or "Figure N." become captions
  - writes report/<stem>.pdf and report/<stem>.docx
"""
import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FIGURES = [ROOT / "figures/fig_learning_curve.png", ROOT / "figures/fig_confusion.png"]


def prepare(md: str, author: str | None) -> str:
    # the doc's byline arrives as "Oct 12, 2026 · @Name"
    md = re.sub(r"^(\w{3} \d{1,2}, \d{4}) · @(\S+)[ \t]*$",
                lambda m: f"{author or m.group(2)} · {m.group(1)}", md, count=1, flags=re.M)

    # images -> local PNGs, without a second caption (the italic caption line follows them).
    # The doc's Markdown export writes an image as "&#91;image: <alt>\]".
    def image(m):
        n = re.match(r"Figure (\d+)", m.group(1))
        return f"![]({FIGURES[int(n.group(1)) - 1] if n else m.group(2)})"
    md = re.sub(r"^&#91;image: (Figure \d+)\..*\\\][ \t]*$", image, md, flags=re.M)
    md = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", image, md)

    # italic "Table N. ..." / "Figure N. ..." lines become captions
    md = re.sub(r"^\*((?:Table|Figure) \d+\..*)\*[ \t]*$", r"::: caption\n\1\n:::", md, flags=re.M)

    # the abstract becomes a compact block without its own heading
    md = re.sub(r"^## Abstract[ \t]*\n\n(.+?)\n", r"::: abstract\n**Abstract.** \1\n:::\n", md, count=1,
                flags=re.M | re.S)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("markdown")
    ap.add_argument("--author", default=None)
    args = ap.parse_args()
    src = Path(args.markdown)
    stem = src.stem
    tmp = REPORT / f"{stem}.prepared.md"
    tmp.write_text(prepare(src.read_text(), args.author))
    html = REPORT / f"{stem}.html"
    subprocess.run(["pandoc", str(tmp), "-f", "markdown+fenced_divs-implicit_figures", "-t", "html5", "-s",
                    "--embed-resources", "--css", str(REPORT / "style.css"),
                    "--metadata", "pagetitle=Bullish or Bearish?", "-o", str(html)], check=True)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={REPORT / f'{stem}.pdf'}", html.as_uri()], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pandoc", str(tmp), "-f", "markdown+fenced_divs-implicit_figures", "-t", "docx",
                    "--resource-path", str(ROOT), "-o", str(REPORT / f"{stem}.docx")], check=True)
    tmp.unlink()
    print("wrote", REPORT / f"{stem}.pdf", "and", REPORT / f"{stem}.docx")


if __name__ == "__main__":
    main()
