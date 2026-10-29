"""Render every slide of the HTML deck to sNN.png (1600x900) for ``build_results_slides.py --images``.

The HTML deck shows one slide at a time; for each slide a temporary copy opens it on load and sets the page
counter to its position in the final PPTX (the closing slide is the last one of the combined deck).

    python scripts/render_html_slides.py NLP501_FinalSlide_v7.html out_dir --total 39
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

CHROME_CANDIDATES = (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                     r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                     "google-chrome", "chromium", "msedge")
WIDTH, HEIGHT = 1600, 900
WAIT_MS = 8000  # Tailwind and the fonts load from a CDN


def find_browser() -> str:
    for cand in CHROME_CANDIDATES:
        if Path(cand).exists() or shutil.which(cand):
            return cand
    raise SystemExit("no Chrome / Edge found for headless rendering")


def opener(index: int, shown: int, total: int) -> str:
    return ("<script>window.addEventListener('load',()=>setTimeout(()=>{goToSlide(%d);"
            "const c=document.getElementById('slide-counter');if(c)c.textContent='%02d / %d';"
            "const b=document.getElementById('progress-bar');if(b)b.style.width='%.2f%%';},300))</script></body>"
            % (index, shown, total, 100 * shown / total))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("html", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--total", type=int, help="slides in the combined PPTX (default: slides in the HTML)")
    args = ap.parse_args()

    src = args.html.read_text(encoding="utf-8")
    n = len(re.findall(r'<section data-slide="\d+"', src))
    total = args.total or n
    browser = find_browser()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            shown = total if i == n - 1 else i + 1
            page = Path(tmp) / f"s{i + 1:02d}.html"
            page.write_text(src.replace("</body>", opener(i, shown, total)), encoding="utf-8")
            png = (args.out_dir / f"s{i + 1:02d}.png").resolve()
            subprocess.run([browser, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            f"--window-size={WIDTH},{HEIGHT}", f"--virtual-time-budget={WAIT_MS}",
                            f"--screenshot={png}", page.resolve().as_uri()],
                           check=True, capture_output=True)
    print(f"rendered {n} slides to {args.out_dir}")


if __name__ == "__main__":
    main()
