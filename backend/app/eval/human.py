"""Q13 / E1: validate the silver labels with people (docs/EVAL.md §2).

    python -m app.eval.human sample   → docs/labeling/index.html: 120 blind (JD, CV) pairs, one page, no server
    python -m app.eval.human score    → docs/HUMAN_LABELS.md from data/eval/human_*.json (the files the page downloads)

The page shows only the JD and the CV text: never the silver grades, a system's rank or another person's labels.
"""

import json
import random
import sys
from collections import defaultdict

from app.config import ROOT
from app.eval import metrics
from app.eval.run import annotators, gold
from app.rag.corpus import load_dir

KIT = ROOT / "docs" / "labeling"
LABELS = ROOT / "data" / "eval"
PER_JD = 10


def sample(seed: int = 13) -> list[dict]:
    """10 pairs per JD, stratified by silver grade, a third of them where the two silver annotators disagree."""
    rnd, a, b, g = random.Random(seed), *annotators().values(), gold()
    pairs = []
    for jd in sorted(g):
        bins = defaultdict(list)
        for cv, grade in sorted(g[jd].items()):
            bins["disagree" if a[jd][cv] != b[jd][cv] else min(int(grade), 3)].append(cv)
        take = rnd.sample(bins["disagree"], min(3, len(bins["disagree"])))
        quota = {3: 2, 2: 2, 1: 2, 0: 1}  # + 3 disagreements = 10; empty bins are topped up below
        for grade, n in quota.items():
            take += rnd.sample([c for c in bins[grade] if c not in take], min(n, len(bins[grade])))
        rest = [c for c in g[jd] if c not in take]
        take += rnd.sample(rest, max(0, PER_JD - len(take)))
        pairs += [{"jd": jd, "cv": cv} for cv in take]
    rnd.shuffle(pairs)
    return pairs


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CV–JD Labelling</title><style>
:root{--bg:#f7f7f5;--card:#fff;--ink:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--acc:#2f5bd3;--on:#e8eefc}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--ink:#ecebe6;--mut:#a3a29c;--line:#33332f;--acc:#7d9cf0;--on:#25304d}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:12px 16px;border-bottom:1px solid var(--line);background:var(--card);position:sticky;top:0}
header b{font-size:16px}.mut{color:var(--mut)}input,button,textarea{font:inherit;color:inherit}
input,textarea{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px 8px}
button{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 12px;cursor:pointer}
button.on{background:var(--on);border-color:var(--acc)}button.pri{background:var(--acc);border-color:var(--acc);color:#fff}
main{display:grid;grid-template-columns:1fr 1fr;gap:12px;padding:12px 16px}
@media (max-width:800px){main{grid-template-columns:1fr}}
section{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;max-height:62vh;overflow:auto}
pre{white-space:pre-wrap;font:14px/1.5 ui-monospace,monospace;margin:0}h2{font-size:13px;text-transform:uppercase;letter-spacing:.04em;margin:0 0 8px;color:var(--mut)}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;padding:0 16px 16px}.grades button{min-width:220px;text-align:left}
details{padding:0 16px 8px}</style></head><body>
<header><b>CV–JD labelling</b><span class="mut" id="prog"></span><span style="flex:1"></span>
<label>Your name <input id="who" size="10" placeholder="e.g. Hoa"></label>
<button id="prev">← Prev</button><button id="next">Next →</button><button class="pri" id="dl">Download labels</button></header>
<details><summary>Guideline (read once)</summary><p>Grade how well the CV fits <i>this</i> JD, on evidence in the CV, as a hiring manager would.
<b>0</b> not relevant: a core must-have is absent, or only claimed or hedged ("familiar with", "currently learning").
<b>1</b> weak or adjacent. <b>2</b> good partial fit: you would phone-screen. <b>3</b> strong fit with demonstrated work: you would interview.
Judge evidence, not keywords (a tool in a skills list is a claim; a project that used it is evidence). Ignore anything that is not job-related.
Grade alone: do not discuss pairs with the other annotator until both of you have downloaded your labels. Keys 0–3 grade, ← → move.</p></details>
<div class="bar grades" id="grades"></div>
<div class="bar"><input id="note" placeholder="optional note (why, or what was unclear)" style="flex:1;min-width:240px"></div>
<main><section><h2 id="jdh">Job description</h2><pre id="jd"></pre></section><section><h2 id="cvh">CV</h2><pre id="cv"></pre></section></main>
<script>
const DATA = __DATA__;
const GR = ['0 · not relevant', '1 · weak / adjacent', '2 · would phone-screen', '3 · would interview'];
const KEY = 'cvjd-labels-v1';
let st = {who: '', i: 0, labels: {}};
try { st = Object.assign(st, JSON.parse(localStorage.getItem(KEY) || '{}')); } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(st)); } catch (e) {} };
const $ = (id) => document.getElementById(id);
const pid = (p) => p.jd + '|' + p.cv;
function show() {
  const p = DATA.pairs[st.i], l = st.labels[pid(p)] || {};
  $('jd').textContent = DATA.jds[p.jd]; $('cv').textContent = DATA.cvs[p.cv];
  $('jdh').textContent = 'Job description · ' + p.jd; $('cvh').textContent = 'CV · ' + p.cv;
  $('note').value = l.note || ''; $('who').value = st.who;
  const done = DATA.pairs.filter((q) => st.labels[pid(q)]?.grade !== undefined).length;
  $('prog').textContent = `pair ${st.i + 1} of ${DATA.pairs.length} · ${done} graded`;
  $('grades').innerHTML = GR.map((g, k) => `<button data-g="${k}" class="${l.grade === k ? 'on' : ''}">${g}</button>`).join('');
}
function grade(k) { const p = DATA.pairs[st.i]; st.labels[pid(p)] = {grade: k, note: $('note').value}; save();
  if (st.i < DATA.pairs.length - 1) st.i++; show(); }
$('grades').onclick = (e) => { const b = e.target.closest('button'); if (b) grade(+b.dataset.g); };
$('note').oninput = () => { const p = DATA.pairs[st.i]; st.labels[pid(p)] = {...(st.labels[pid(p)] || {}), note: $('note').value}; save(); };
$('who').oninput = () => { st.who = $('who').value.trim(); save(); };
$('prev').onclick = () => { st.i = Math.max(0, st.i - 1); save(); show(); };
$('next').onclick = () => { st.i = Math.min(DATA.pairs.length - 1, st.i + 1); save(); show(); };
document.onkeydown = (e) => { if (e.target.tagName === 'INPUT') return;
  if ('0123'.includes(e.key)) grade(+e.key); if (e.key === 'ArrowRight') $('next').click(); if (e.key === 'ArrowLeft') $('prev').click(); };
$('dl').onclick = () => {
  if (!st.who) { alert('Type your name first.'); return; }
  const out = {annotator: st.who, sample: DATA.sample, labels: DATA.pairs.filter((p) => st.labels[pid(p)]?.grade !== undefined)
    .map((p) => ({jd: p.jd, cv: p.cv, ...st.labels[pid(p)]}))};
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: 'application/json'}));
  a.download = `human_${st.who.toLowerCase().replace(/\\W+/g, '_')}.json`; a.click();
};
show();
</script></body></html>"""


def write_kit() -> None:
    pairs = sample()
    jds = {d.id: d.text for d in load_dir("jds")}
    cvs = {d.id: d.text for d in load_dir("cvs")}
    data = {"sample": "v1-seed13", "pairs": pairs, "jds": {p["jd"]: jds[p["jd"]] for p in pairs},
            "cvs": {c: cvs[c] for c in sorted({p["cv"] for p in pairs})}}
    KIT.mkdir(parents=True, exist_ok=True)
    (KIT / "index.html").write_text(PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False)))
    (KIT / "pairs.json").write_text(json.dumps(pairs, indent=1))
    print(f"{len(pairs)} pairs over {len(data['jds'])} JDs and {len(data['cvs'])} CVs → {KIT / 'index.html'}")


def score() -> str:
    files = sorted(LABELS.glob("human_*.json"))
    humans = {json.loads(f.read_text())["annotator"]: {(x["jd"], x["cv"]): x["grade"] for x in json.loads(f.read_text())["labels"]}
              for f in files}
    a, b = annotators().values()
    silver = {"annotator A (Opus)": {(j, c): v for j, d in a.items() for c, v in d.items()},
              "annotator B (Sonnet)": {(j, c): v for j, d in b.items() for c, v in d.items()},
              "silver gold (mean, rounded)": {(j, c): int(v + 0.5) for j, d in gold().items() for c, v in d.items()}}
    lines = ["# Human validation of the silver labels (generated by `python -m app.eval.human score`; do not edit)", "",
             f"Annotators: {', '.join(humans) or 'none yet'}. Quadratic-weighted Cohen's κ on the pairs both sides graded.",
             "Decision rule (EVAL.md §2): κ(human, silver) ≥ 0.8 keeps the silver set as the main test set; ≥ 0.6 keeps "
             "it as a secondary set; below that, the human labels replace it.", "",
             "| Pair | pairs | exact agreement | weighted κ |", "|---|---|---|---|"]

    def row(n1: str, x: dict, n2: str, y: dict) -> None:
        common = sorted(set(x) & set(y))
        if common:
            xs, ys = [int(x[k]) for k in common], [int(y[k]) for k in common]
            lines.append(f"| {n1} vs {n2} | {len(common)} | {sum(p == q for p, q in zip(xs, ys)) / len(common):.1%} | "
                         f"{metrics.weighted_kappa(xs, ys):.3f} |")

    names = list(humans)
    for i, h in enumerate(names):
        for h2 in names[i + 1:]:
            row(h, humans[h], h2, humans[h2])
    for h in names:
        for sname, s in silver.items():
            row(h, humans[h], sname, s)
    text = "\n".join(lines) + "\n"
    (ROOT / "docs" / "HUMAN_LABELS.md").write_text(text)
    return text


if __name__ == "__main__":
    write_kit() if sys.argv[1:2] == ["sample"] else print(score())
