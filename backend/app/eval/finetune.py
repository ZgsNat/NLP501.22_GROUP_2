"""E20: a learned reranker. The cross-encoder fine-tuned on the benchmark's dev JDs, judged on test and on Djinni.

    cd backend && uv run --group ml python -m app.eval.finetune train    → models/ce-ft-dev (gitignored)
    cd backend && uv run --group ml python -m app.eval.finetune score    → results + docs/FINETUNE_RESULTS.md

On a weak machine, train on Colab instead (notebooks/finetune_ce_colab.ipynb, scripts/colab_train_ce.py):
    cd backend && uv run python -m app.eval.finetune export              → .runtime/colab/ce_train.jsonl
    cd backend && uv run python -m app.eval.finetune install ce-ft-dev.zip   the zip Colab returns → models/ce-ft-dev

Input, the same for the zero-shot control and the fine-tuned model: the JD as its requirement queries
(retrieve.queries) against the CV's evidence passages (each query's best structural chunk, as the agent reranks them).
Target: the CV's 0–3 label / 3, binary cross-entropy on the logit. Training data: the 8 dev JDs of NLP501-bench v1
(6 ours × 50 CVs + 2 coding × 200 CVs), never a test JD; fixed hyper-parameters, no search. Test CVs can also appear
under dev JDs (the benchmark splits by JD), so the test split measures transfer to new JDs, Djinni to new CVs and JDs.

CV-disjoint check (G5, `--cv-disjoint` on export / install / score): the same, trained only on the half of each pool's
CVs that `held_out` does not pick, and scored on the test JDs over the other half only. If the +gain of CE-ft on test
comes from having seen the test CVs under dev JDs, it shrinks here. Model: models/ce-ft-dev-cvdisjoint.

Systems saved per bench, so the benchmark reports can show them:
    CE0 (JD, evidence)            the base model on this input: the control for the input format
    CE-ft (JD, evidence)          the fine-tuned model alone
    B3 8-axis ⊕ CE0 [hybrid]      rules fused with the control
    B3 8-axis ⊕ CE-ft [hybrid]    rules fused with the fine-tuned model (what would replace the agent's rerank)
"""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from app.config import ROOT, Settings, get_settings
from app.eval import metrics
from app.eval.bench import Bench, load
from app.eval.benchmark import MANIFEST
from app.eval.run import b3_systems, full_order, requirements, result_path, save, score_system
from app.nlp.pipeline import process
from app.rag import retrieve
from app.rag.corpus import load_yaml
from app.rag.store import KnowledgeStore

from scripts.colab_train_ce import BASE, MAX_LEN, SEED  # one definition, shared with the Colab script

OUT = ROOT / "models" / "ce-ft-dev"
OUT_DISJOINT = ROOT / "models" / "ce-ft-dev-cvdisjoint"
DOC = ROOT / "docs" / "FINETUNE_RESULTS.md"
EXPORT = get_settings().runtime_dir / "colab" / "ce_train.jsonl"
EXPORT_DISJOINT = get_settings().runtime_dir / "colab" / "ce_train_cvdisjoint.jsonl"
WEIGHTS = ("config.json", "train.json")  # files a usable model directory must have (plus the weights)
SYSTEMS = ("CE0 (JD, evidence)", "CE-ft (JD, evidence)", "B3 8-axis ⊕ CE0 [hybrid]", "B3 8-axis ⊕ CE-ft [hybrid]",
           "B3 8-axis ⊕ CE [hybrid]")


def jd_query(qs: list[retrieve.Query]) -> str:
    return " | ".join(q.text for q in qs)


def evidence(hit: retrieve.CVHit | None, chunks: dict) -> str:
    """The CV's best passage per requirement, in query order, without repeats; the first 3 chunks if not retrieved."""
    ids = list(dict.fromkeys(cid for cid, _ in hit.best.values() if cid in chunks)) if hit else []
    return "\n".join(chunks[c].embed_text for c in (ids or list(chunks)[:3]))


class Prepared:
    """One bench, indexed once: per JD the requirements, the retrieval hits and the (query, evidence) inputs."""

    def __init__(self, b: Bench):
        self.b, self.pool = b, [d.id for d in b.cvs]
        self.store = KnowledgeStore(Settings(qdrant_url=":memory:"))
        self.prof = {d.id: process(d.id, d.text, d.meta.get("headline", "")) for d in b.cvs}
        self.store.index_items("chunks_structural", retrieve.chunk_items(list(self.prof.values())))
        weights = load_yaml("criteria_profiles.yaml")["profiles"]
        self.req, self.weights, self.inputs = {}, {}, {}
        for jd in b.jds:
            r = asyncio.run(requirements(jd, "hybrid", b.name))
            qs = retrieve.queries(r["req"], r["terms"], r["subs"])
            hits = {h.cv_id: h for h in retrieve.rank_cvs(self.store, "chunks_structural", qs,
                                                          limit=max(200, 2 * len(self.pool)))}
            self.req[jd.id], self.weights[jd.id] = r, weights[jd.meta.get("company_profile", "default")]["weights"]
            q = jd_query(qs)
            self.inputs[jd.id] = {c: (q, evidence(hits.get(c), {ch.id: ch for ch in self.prof[c].chunks}))
                                  for c in self.pool}


def dev_jds() -> dict[str, list[str]]:
    return json.loads(MANIFEST.read_text())["splits"]["dev"]


def held_out(cv_id: str) -> bool:
    """The CV-disjoint split: a fixed half of every pool (by hash, not by position), never trained on."""
    return hashlib.sha256(cv_id.encode()).digest()[0] % 2 == 0


def examples(prepared: dict[str, Prepared], cv_disjoint: bool = False) -> list[tuple[str, str, float]]:
    out = []
    for src, jd_ids in dev_jds().items():
        p = prepared[src]
        out += [(*p.inputs[j][c], g / 3) for j in jd_ids for c, g in p.b.gold[j].items()
                if not (cv_disjoint and held_out(c))]
    return out


def _model(path: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(SEED)
    return AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path, num_labels=1)


def export(data: list[tuple[str, str, float]], path: Path = EXPORT, cv_disjoint: bool = False) -> Path:
    """The training pairs as JSONL (first line: provenance), for scripts/colab_train_ce.py. The text is the redacted
    CVProfile of synthetic CVs (ours, coding); no Djinni CV and no test JD is in it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"meta": {"dev_jds": dev_jds(), "manifest": MANIFEST.name, "cv_disjoint": cv_disjoint}})]
    lines += [json.dumps({"q": q, "e": e, "y": y}, ensure_ascii=False) for q, e, y in data]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def install(archive: Path, out: Path = OUT) -> Path:
    """Unpack the model zip from Colab into models/ce-ft-dev (whatever folder level the zip has) and check it."""
    import shutil
    import zipfile

    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        cfg = next((n for n in names if n.endswith("config.json") and "tokenizer" not in n), None)
        if not cfg:
            raise SystemExit(f"{archive.name}: no config.json inside; is this the model zip from Colab?")
        prefix = cfg[: -len("config.json")]
        if out.exists():  # keep the previous model next to it, never delete it
            shutil.move(str(out), str(out.with_name(out.name + ".prev")))
        out.mkdir(parents=True)
        for n in names:
            if n.startswith(prefix) and not n.endswith("/"):
                (out / n[len(prefix):]).parent.mkdir(parents=True, exist_ok=True)
                (out / n[len(prefix):]).write_bytes(z.read(n))
    missing = [f for f in WEIGHTS if not (out / f).exists()]
    if missing or not any(out.glob("*.safetensors")) and not any(out.glob("*.bin")):
        raise SystemExit(f"{out}: incomplete model (missing {missing or 'weights'})")
    return out


def logits(path: str, pairs: list[tuple[str, str]]) -> list[float]:
    import torch

    tok, model = _model(path)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), 32):
            q, e = zip(*pairs[i:i + 32])
            enc = tok(list(q), list(e), truncation="only_second", max_length=MAX_LEN, padding=True, return_tensors="pt")
            out += model(**enc).logits.squeeze(-1).tolist()
    return out


def rank(p: Prepared, ce: dict[str, dict[str, float]], name: str) -> dict[str, dict[str, list[str]]]:
    """→ system → JD → ranking, for one model's CV scores."""
    out: dict[str, dict[str, list[str]]] = {f"{name} (JD, evidence)": {}, f"B3 8-axis ⊕ {name} [hybrid]": {}}
    for j in p.req:
        out[f"{name} (JD, evidence)"][j] = sorted(p.pool, key=lambda c: ce[j][c], reverse=True)
        out[f"B3 8-axis ⊕ {name} [hybrid]"][j] = full_order(
            b3_systems(p.store, p.pool, p.prof, p.req[j], p.weights[j], ce=ce[j])["8-axis ⊕ CE"], p.pool)
    return out


def score_bench(p: Prepared, models: tuple[tuple[str, str], ...] = (("CE0", BASE), ("CE-ft", str(OUT)))) -> None:
    keys = [(j, c) for j in p.req for c in p.pool]
    pairs = [p.inputs[j][c] for j, c in keys]
    for name, path in models:
        flat = dict(zip(keys, logits(path, pairs)))
        ce = {j: {c: flat[(j, c)] for c in p.pool} for j in p.req}
        for system, per_jd in rank(p, ce, name).items():
            save(system, per_jd, p.b.name)
    if not result_path("B3 8-axis ⊕ CE [hybrid]", p.b.name).exists():  # `run retrieval` saves it; same code here
        base = {j: full_order(b3_systems(p.store, p.pool, p.prof, p.req[j], p.weights[j])["8-axis ⊕ CE"], p.pool)
                for j in p.req}
        save("B3 8-axis ⊕ CE [hybrid]", base, p.b.name)
    print(f"{p.b.name}: {len(p.req)} JDs scored", flush=True)


def _ndcg(system: str, bench: str, only_held_out: bool = False) -> dict[str, float]:
    """Per-JD nDCG@10; only_held_out scores each ranking over the held-out half of the pool (condensed)."""
    path = result_path(system, bench)
    if not path.exists():
        return {}
    b = load(bench)
    gold = {j: {c: g for c, g in v.items() if held_out(c)} for j, v in b.gold.items()} if only_held_out else b.gold
    sc = score_system(json.loads(path.read_text())["jds"], gold, b.traps)
    return dict(zip(sc["jds"], sc["ndcg@10"]))


def disjoint_section(m: dict) -> list[str]:
    """Test JDs scored over held-out CVs only: the leaked model (CE-ft saw these CVs under dev JDs) against the
    CV-disjoint one (CE-ft⊥ never did), both alone and fused with the rules."""
    if not (OUT_DISJOINT / "train.json").exists():
        return []
    systems = ("CE0 (JD, evidence)", "CE-ft (JD, evidence)", "CE-ft⊥ (JD, evidence)", "B3 8-axis ⊕ CE [hybrid]",
               "B3 8-axis ⊕ CE0 [hybrid]", "B3 8-axis ⊕ CE-ft [hybrid]", "B3 8-axis ⊕ CE-ft⊥ [hybrid]")
    test = {src: m["splits"]["test"][src] for src in ("ours", "coding")}
    scores = {s: {j: v for src, js in test.items() for j, v in _ndcg(s, src, True).items() if j in js} for s in systems}
    meta = json.loads((OUT_DISJOINT / "train.json").read_text())
    lines = ["## Test (v1, 10 JDs) over held-out CVs only: the CV-disjoint check (G5)", "",
             f"CE-ft⊥ was trained on {meta['pairs']} dev pairs whose CV is not held out; every system here is scored "
             "over the held-out half of each pool. CE-ft saw those CVs in training (under dev JDs), CE-ft⊥ did not.", "",
             "| System | JDs | nDCG@10 [95% CI] |", "|---|---|---|"]
    lines += [f"| {s} | {len(per)} | " + "{:.3f} [{:.3f}, {:.3f}]".format(*metrics.bootstrap_ci(list(per.values()))) + " |"
              for s, per in scores.items() if per]
    lines += ["", "| First | second | ΔnDCG@10 | p |", "|---|---|---|---|"]
    for a, b in (("B3 8-axis ⊕ CE-ft⊥ [hybrid]", "B3 8-axis ⊕ CE0 [hybrid]"),
                 ("B3 8-axis ⊕ CE-ft⊥ [hybrid]", "B3 8-axis ⊕ CE [hybrid]"),
                 ("B3 8-axis ⊕ CE-ft [hybrid]", "B3 8-axis ⊕ CE-ft⊥ [hybrid]"), ("CE-ft⊥ (JD, evidence)", "CE0 (JD, evidence)")):
        common = sorted(set(scores[a]) & set(scores[b]))
        if common:
            xs, ys = [scores[a][j] for j in common], [scores[b][j] for j in common]
            lines.append(f"| {a} | {b} | {sum(xs) / len(xs) - sum(ys) / len(ys):+.3f} | {metrics.paired_bootstrap_p(xs, ys):.3f} |")
    return lines + [""]


def report() -> str:
    m = json.loads(MANIFEST.read_text())
    meta = json.loads((OUT / "train.json").read_text())
    from app.eval import djinni

    sets = {"Test (v1, 10 JDs, official)": {"ours": m["splits"]["test"]["ours"], "coding": m["splits"]["test"]["coding"]},
            f"Djinni (real CVs, 8 JDs; labels: {djinni.gold()[1]})": {"djinni": [d.id for d in djinni.jds()]},
            "Dev (v1, 8 JDs: the training JDs, in-sample)": {"ours": m["splits"]["dev"]["ours"],
                                                             "coding": m["splits"]["dev"]["coding"]}}
    lines = ["# Fine-tuned reranker (generated by `python -m app.eval.finetune score`; do not edit by hand)", "",
             f"Base `{meta['base']}`, fine-tuned on {meta['pairs']} (JD, CV) pairs of the v1 dev JDs only "
             f"({meta['epochs']} epochs, lr {meta['lr']}, batch {meta['batch']}, max {meta['max_len']} tokens, seed "
             f"{meta['seed']}; no hyper-parameter search). Input: the JD's requirement queries vs the CV's evidence "
             "passages. CE0 is the untouched base model on the same input. Mean nDCG@10 with 95% bootstrap CI; paired "
             "one-sided bootstrap p for “first > second”. PYTHONHASHSEED=0.", ""]
    pairs = (("CE-ft (JD, evidence)", "CE0 (JD, evidence)"), ("B3 8-axis ⊕ CE-ft [hybrid]", "B3 8-axis ⊕ CE0 [hybrid]"),
             ("B3 8-axis ⊕ CE-ft [hybrid]", "B3 8-axis ⊕ CE [hybrid]"))
    for title, srcs in sets.items():
        scores = {s: {j: v for src, js in srcs.items() for j, v in _ndcg(s, src).items() if j in js} for s in SYSTEMS}
        lines += [f"## {title}", "", "| System | JDs | nDCG@10 [95% CI] |", "|---|---|---|"]
        for s, per in scores.items():
            if per:
                lines.append(f"| {s} | {len(per)} | " + "{:.3f} [{:.3f}, {:.3f}]".format(*metrics.bootstrap_ci(list(per.values()))) + " |")
        lines += ["", "| First | second | ΔnDCG@10 | p |", "|---|---|---|---|"]
        for a, b in pairs:
            common = sorted(set(scores[a]) & set(scores[b]))
            if common:
                xs, ys = [scores[a][j] for j in common], [scores[b][j] for j in common]
                lines.append(f"| {a} | {b} | {sum(xs) / len(xs) - sum(ys) / len(ys):+.3f} | "
                             f"{metrics.paired_bootstrap_p(xs, ys):.3f} |")
        lines.append("")
    lines += disjoint_section(m)
    DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(DOC)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "export", "install", "score", "report"])
    ap.add_argument("archive", nargs="?", type=Path, help="install: the model zip from Colab")
    ap.add_argument("--cv-disjoint", action="store_true", help="the G5 check: train without the held-out CVs")
    a = ap.parse_args()
    if a.what in ("train", "export"):
        data = examples({src: Prepared(load(src)) for src in ("ours", "coding")}, a.cv_disjoint)
        if a.what == "train":
            from scripts.colab_train_ce import train as train_ce
            train_ce(data, OUT_DISJOINT if a.cv_disjoint else OUT, {"dev_jds": dev_jds(), "cv_disjoint": a.cv_disjoint})
        else:
            print(f"{len(data)} pairs → {export(data, EXPORT_DISJOINT if a.cv_disjoint else EXPORT, a.cv_disjoint)}")
    elif a.what == "install":
        print(f"model → {install(a.archive, OUT_DISJOINT if a.cv_disjoint else OUT)}")
    elif a.what == "score" and a.cv_disjoint:
        for src in ("ours", "coding"):
            score_bench(Prepared(load(src)), (("CE-ft⊥", str(OUT_DISJOINT)),))
        print(report())
    elif a.what == "score":
        for src in ("ours", "coding", "djinni"):
            score_bench(Prepared(load(src)))
        print(report())
    else:
        print(report())


if __name__ == "__main__":
    main()
