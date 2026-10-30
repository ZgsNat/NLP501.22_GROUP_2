"""Report figures, drawn from the generated results (never typed by hand): python -m app.eval.figures

    docs/figures/ladder.svg     NLP501-bench v1 test split: nDCG@10 per system with its 95% CI
    docs/figures/controls.svg   what the agent's own controls change: LLM claims before → after (P6 vs B4)

Static SVG for the markdown report; the same numbers are in the tables next to each figure (the table view).
Palette: reference slot-1 blue for the merged system, a neutral for the others (dataviz skill, validated)."""

import json
from html import escape

from app.config import ROOT
from app.eval import benchmark, metrics
from app.eval.run import result_path

OUT = ROOT / "docs" / "figures"
BLUE, GREY, INK, INK2, GRID, SURFACE = "#2a78d6", "#8a8983", "#0b0b0b", "#5f5e5a", "#e6e5e1", "#fcfcfb"
FONT = "font-family='system-ui, -apple-system, Segoe UI, sans-serif'"
MERGED = {"B3 8-axis ⊕ CE [hybrid]", "B4", "P6"}
LABEL = {"B0 BM25": "B0 BM25", "B2 hybrid": "B2 hybrid (BM25 + dense)", "R structural + CE": "Structural chunks + CE",
         benchmark.DAN: "Đán's core (coding branch)", "B3 rules ⊕ CE": "Our v2 (4 rule criteria ⊕ CE)",
         "B3 8-axis ⊕ CE [hybrid]": "Merged v3.1, deterministic", "B4": "Merged v3.1, fixed pipeline (B4)",
         "P6": "Merged v3.1, full agent (P6)"}


def _svg(w: int, h: int, body: list[str], title: str, note: str) -> str:
    return "\n".join([f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' width='{w}' height='{h}' {FONT}>",
                      f"<rect width='{w}' height='{h}' fill='{SURFACE}'/>",
                      f"<text x='20' y='28' font-size='15' font-weight='600' fill='{INK}'>{escape(title)}</text>",
                      f"<text x='20' y='{h - 12}' font-size='11' fill='{INK2}'>{escape(note)}</text>", *body, "</svg>"])


def ladder() -> str:
    m = json.loads(benchmark.MANIFEST.read_text())
    test = m["splits"]["test"]["ours"] + m["splits"]["test"]["coding"]
    rows = []
    for s in benchmark.SYSTEMS:
        sc = benchmark._scores(s)
        nd = [sc[j]["ndcg@10"] for j in test if j in sc]
        if len(nd) == len(test):
            rows.append((s, *metrics.bootstrap_ci(nd)))
    x0, x1, lo, hi, top, rh = 300, 680, 0.4, 1.0, 56, 30
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
    body = []
    for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        body += [f"<line x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{top - 8}' y2='{top + rh * len(rows)}' stroke='{GRID}' stroke-width='1'/>",
                 f"<text x='{X(t):.1f}' y='{top + rh * len(rows) + 16}' font-size='11' fill='{INK2}' text-anchor='middle'>{t:.1f}</text>"]
    for i, (s, mean, a, b) in enumerate(rows):
        y, c = top + rh * i + rh / 2, BLUE if s in MERGED else GREY
        body += [f"<text x='{x0 - 12}' y='{y + 4:.1f}' font-size='13' fill='{INK}' text-anchor='end'>{escape(LABEL[s])}</text>",
                 f"<line x1='{X(a):.1f}' x2='{X(b):.1f}' y1='{y:.1f}' y2='{y:.1f}' stroke='{c}' stroke-width='2' stroke-linecap='round'/>",
                 f"<circle cx='{X(mean):.1f}' cy='{y:.1f}' r='5' fill='{c}' stroke='{SURFACE}' stroke-width='2'/>",
                 f"<text x='{X(b) + 8:.1f}' y='{y + 4:.1f}' font-size='12' fill='{INK}'>{mean:.3f}</text>"]
    h = top + rh * len(rows) + 48
    return _svg(720, h, body, "NLP501-bench v1, test split (10 JDs): nDCG@10",
                "Dot = mean over test JDs, line = 95% bootstrap CI. Blue = merged v3.1. Source: docs/BENCHMARK_RESULTS.md")


def controls() -> str:
    """Share of LLM outputs failing a check, on the first attempt and after the graph's controls acted."""
    def rate(system: str, part: str, first: bool) -> float:
        qs = [x["quality"] for b in ("ours", "coding") if result_path(system, b).exists()
              for x in json.loads(result_path(system, b).read_text())["jds"].values() if x.get("quality")]
        key = f"{part}_{'first' if first else 'final'}"
        bad = sum((q[key] or {}).get("claims_bad", 0) for q in qs)
        return 100 * bad / max(sum((q[key] or {}).get("claims", 0) for q in qs), 1)
    rows = [("Rationales contradicting the CV (Q4): P6", rate("P6", "verify", True), rate("P6", "verify", False)),
            ("Same, fixed pipeline without verify (B4)", rate("B4", "verify", True), rate("B4", "verify", False)),
            ("Report candidates with unsupported claims (Q6): P6", rate("P6", "report", True), rate("P6", "report", False))]
    x0, x1, lo, hi, top, rh = 330, 680, 0.0, 10.0, 60, 34
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)  # noqa: E731
    body = []
    for t in (0, 2, 4, 6, 8, 10):
        body += [f"<line x1='{X(t):.1f}' x2='{X(t):.1f}' y1='{top - 8}' y2='{top + rh * len(rows)}' stroke='{GRID}' stroke-width='1'/>",
                 f"<text x='{X(t):.1f}' y='{top + rh * len(rows) + 16}' font-size='11' fill='{INK2}' text-anchor='middle'>{t}%</text>"]
    for i, (label, first, final) in enumerate(rows):
        y = top + rh * i + rh / 2
        body += [f"<text x='{x0 - 12}' y='{y + 4:.1f}' font-size='12' fill='{INK}' text-anchor='end'>{escape(label)}</text>",
                 f"<line x1='{X(min(first, final)):.1f}' x2='{X(max(first, final)):.1f}' y1='{y:.1f}' y2='{y:.1f}' stroke='{GREY}' stroke-width='2'/>",
                 f"<circle cx='{X(first):.1f}' cy='{y:.1f}' r='5' fill='{SURFACE}' stroke='{GREY}' stroke-width='2'/>",
                 f"<circle cx='{X(final):.1f}' cy='{y:.1f}' r='5' fill='{BLUE}' stroke='{SURFACE}' stroke-width='2'/>",
                 f"<text x='{X(max(first, final)) + 10:.1f}' y='{y + 4:.1f}' font-size='12' fill='{INK}'>{first:.1f}% → {final:.1f}%"
                 + (" (no retry: unchanged)" if abs(first - final) < 0.05 else "") + "</text>"]
    h = top + rh * len(rows) + 48
    return _svg(760, h, body, "What the agent's controls change: share of LLM outputs failing a check",
                "Hollow = first attempt · filled = after the controls (retry, then label or drop) · all benchmark JDs")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("ladder", ladder), ("controls", controls)):
        (OUT / f"{name}.svg").write_text(fn())
        print(f"→ {OUT / name}.svg")
