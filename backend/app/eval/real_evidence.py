"""Evidence on real data: what the NLP pipeline extracts from every real CV on this machine, and the real-data benchmark
recomputed per JD. Answers "are these numbers measured, or generated?" with file paths.

Real, public, anonymised CV sets (not authored by the team; all under gitignored folders):
    Djinni Recruitment Dataset (MIT, UNLP 2024)        .runtime/bench/djinni/cvs/*.txt, jds.yaml
    Kaggle Resume Dataset (24 trades, PDF)               .runtime/bench/kaggle_resume/data/<TRADE>/*.pdf
    resume-job-description-fit (Hugging Face)            data/public/cvs/*.txt, jds.yaml

Per CV the row records only derived facts (sections, dated roles, years, tools with evidence strength, depth, quality
flags), never CV text. "Valid" = the pipeline recovered the structure; "not valid" = a quality flag says the parse is
incomplete (no experience section, no dated roles, garbled or short text, injected instructions).

    python -m app.eval.real_evidence   → docs/REAL_DATA_EVIDENCE.md, docs/real_data_extraction.csv
"""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

from app.config import ROOT, get_settings
from app.eval import djinni_gold
from app.nlp import extract, quality
from app.nlp.text import read_file
from app.nlp.pipeline import process

RUNTIME = get_settings().runtime_dir
DATASETS = {
    "djinni": {"glob": (RUNTIME / "bench" / "djinni" / "cvs", "*.txt"), "jds": RUNTIME / "bench" / "djinni" / "jds.yaml",
               "source": "Djinni Recruitment Dataset (Drushchak & Romanyshyn, UNLP 2024), lang-uk on Hugging Face",
               "license": "MIT", "labels": "424 cặp (JD, CV) có nhãn người: data/eval/human_djinni_gold.json"},
    "kaggle": {"glob": (RUNTIME / "bench" / "kaggle_resume" / "data", "*/*.pdf"), "jds": None,
               "source": "Kaggle Resume Dataset (2.484 PDF, 24 ngành)", "license": "không ghi trên trang dataset",
               "labels": "không có nhãn xếp hạng; Resume.csv là văn bản gốc để so chữ đọc từ PDF"},
    "rjf": {"glob": (ROOT / "data" / "public" / "cvs", "*.txt"), "jds": ROOT / "data" / "public" / "jds.yaml",
            "source": "resume-job-description-fit (cnamuangtoun, Hugging Face), lọc theo tín hiệu QA",
            "license": "không ghi trên trang dataset", "labels": "nhãn No/Potential/Good Fit của tác giả dataset"},
}
INVALID = ("garbled_text", "short_text", "prompt_injection")  # the text itself is unusable or unsafe
PARTIAL = ("no_experience_section", "no_dated_roles")  # readable, but the layout hides part of the structure
OK, PART, BAD = "hợp lệ", "thiếu cấu trúc", "không hợp lệ"
DOC = ROOT / "docs" / "REAL_DATA_EVIDENCE.md"
CSV = ROOT / "docs" / "real_data_extraction.csv"
COLUMNS = ["dataset", "path", "trade", "lang", "words", "sections", "experience", "years", "depth", "tools", "flags", "status"]


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def status(flags: list[str]) -> str:
    """Unusable text wins over a partial layout; few_skills alone is not a defect (most trades name no test tools)."""
    bad, part = [f for f in flags if f in INVALID], [f for f in flags if f in PARTIAL]
    return f"{BAD}: {', '.join(bad)}" if bad else f"{PART}: {', '.join(part)}" if part else OK


def row(dataset: str, path: Path) -> dict:
    raw = read_file(path.name, path.read_bytes()) if path.suffix == ".pdf" else path.read_text(encoding="utf-8")
    p = process(f"{dataset}-{path.stem}", raw)
    flags = quality.flags(p)
    tools = sorted(f"{k}({extract.best(ms).strength})" for k, ms in p.skills.items())
    return {"dataset": dataset, "path": rel(path), "trade": path.parent.name if dataset == "kaggle" else "",
            "lang": p.lang, "words": len(p.text.split()), "sections": "+".join(sorted({s.kind for s in p.sections})),
            "experience": any(s.kind == "experience" for s in p.sections),
            "years": round(p.years_total, 1) if p.years_total is not None else "", "depth": p.depth_cue,
            "tools": " ".join(tools), "flags": " ".join(flags), "status": status(flags)}


def measure() -> list[dict]:
    rows = []
    for name, cfg in DATASETS.items():
        folder, pattern = cfg["glob"]
        for path in sorted(folder.glob(pattern)):
            try:
                rows.append(row(name, path))
            except Exception as e:  # an unreadable file is a result, not a crash
                rows.append({**dict.fromkeys(COLUMNS, ""), "dataset": name, "path": rel(path),
                             "status": f"{BAD}: {type(e).__name__}"})
    return rows


def djinni_per_jd() -> tuple[list[str], list[list[str]]]:
    jds = {j["id"]: j["title"] for j in yaml.safe_load(DATASETS["djinni"]["jds"].read_text(encoding="utf-8"))}
    gold = json.loads((ROOT / "data" / "eval" / "human_djinni_gold.json").read_text(encoding="utf-8"))
    judged = Counter(r["jd"] for r in gold["labels"])
    systems = djinni_gold.evaluate_production()["systems"]
    names = list(systems)
    out = [[jd, jds.get(jd, ""), str(judged[jd])] + [f"{systems[s]['per_jd_ndcg10'][jd]:.3f}" for s in names]
           for jd in sorted(judged)]
    out.append(["**trung bình**", "", str(sum(judged.values()))] + [f"**{systems[s]['ndcg10'][0]:.3f}**" for s in names])
    return names, out


def table(head: list[str], rows: list[list]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def report(rows: list[dict]) -> str:
    out = ["# Evidence on real data (generated by `python -m app.eval.real_evidence`; do not edit by hand)", "",
           "Every row below is a real, public, anonymised CV the team did not write, run through the same NLP pipeline the "
           "app uses. Only derived facts are kept; the files stay under gitignored folders. Full per-file table: "
           "`docs/real_data_extraction.csv`.", "", "## 1. Real datasets on this machine", ""]
    inv = []
    for name, cfg in DATASETS.items():
        folder, pattern = cfg["glob"]
        rs = [r for r in rows if r["dataset"] == name]
        c = Counter(r["status"].split(":")[0] for r in rs)
        inv.append([name, cfg["source"], cfg["license"], f"`{rel(folder)}/{pattern}`", len(rs), c[OK], c[PART], c[BAD],
                    cfg["labels"]])
    out += table(["bộ", "nguồn", "giấy phép", "đường dẫn (tương đối)", "số CV", OK, PART, BAD, "nhãn"], inv)
    out += ["", f"*{OK}*: đọc được và khôi phục đủ cấu trúc (mục kinh nghiệm, vai trò có ngày). *{PART}*: đọc được, "
            "trích xuất được công cụ và số năm, nhưng bố cục không có tiêu đề mục hoặc mốc thời gian (vd hồ sơ Djinni là "
            f"đoạn văn liền). *{BAD}*: chữ bị nhiễu, quá ngắn, có câu chèn lệnh hoặc không đọc được tệp."]
    out += ["", "## 2. Which NLP numbers come from real data", "",
            *table(["số trên deck", "dữ liệu", "loại dữ liệu", "đáp án đúng (gold)"], [
                ["F1 tool extraction 0.978 · MAE 0.79 năm · depth 89% · PII 0% · 87.5% · 0.157",
                 "`.runtime/bench/coding/` (200 CV)", "**tổng hợp** (generator của nhánh coding)", "dữ kiện gốc của generator `latent.json`"],
                ["F1 0.997 sau OCR", "30 CV coding in thành ảnh", "**tổng hợp**, ảnh do nhóm render", "`latent.json`"],
                ["100% đọc được · date range 41.9% → 97.7% · nhận nhầm công cụ 17.2% → 1.6%",
                 "`.runtime/bench/kaggle_resume/data/` (2.484 PDF)", "**thật**", "không có gold trích xuất; so chữ với `Resume.csv`"],
                ["prose precision 184/204 = 90% (REALTEXT)", "datasetmaster/resumes, 217 CV", "**thật**",
                 "các trường có cấu trúc do chính người viết CV điền"],
                ["Djinni nDCG@10 0.422 → 0.643, +0.214", "`.runtime/bench/djinni/` (250 CV, 8 JD)", "**thật**",
                 "nhãn người `data/eval/human_djinni_gold.json` (xem lưu ý ở mục 5)"],
                ["benchmark v1 0.851 (ours + coding)", "`data/cvs/`, `.runtime/bench/coding/`", "**tổng hợp**", "nhãn silver LLM + 120 cặp nhãn người mù"]]),
            "", "## 3. Extraction on real CVs: counts by status", ""]
    for name in DATASETS:
        rs = [r for r in rows if r["dataset"] == name]
        reasons = Counter(f for r in rs if r["status"] != OK for f in r["status"].split(": ", 1)[1].split(", "))
        tools = Counter(t.split("(")[0] for r in rs for t in r["tools"].split())
        out += [f"**{name}** ({len(rs)} CV): lý do: " + (", ".join(f"{k} {v}" for k, v in reasons.most_common()) or "—")
                + f"; công cụ nhận ra nhiều nhất: {', '.join(f'{k} {v}' for k, v in tools.most_common(8)) or '—'}", ""]
    out += ["## 4. Examples (path → what was extracted)", ""]
    head = ["đường dẫn", "ngành", "lang", "sections", "năm KN", "depth", "công cụ (mức bằng chứng)", "cờ", "trạng thái"]
    for name in DATASETS:
        rs = [r for r in rows if r["dataset"] == name]
        by = {k: [r for r in rs if r["status"].split(":")[0] == k] for k in (OK, PART, BAD)}
        picks = ((OK, sorted(by[OK], key=lambda r: -len(r["tools"].split()))[:6]),
                 (PART, sorted(by[PART], key=lambda r: -len(r["tools"].split()))[:6]), (BAD, by[BAD][:10]))
        for label, pick in picks:
            if not pick:
                continue
            out += [f"### {name}: {label} ({len(pick)} ví dụ / {len(by[label])})", ""]
            out += table(head, [[f"`{r['path']}`", r["trade"], r["lang"], r["sections"], r["years"], r["depth"],
                                 r["tools"] or "—", r["flags"] or "—", r["status"]] for r in pick]) + [""]
    names, per_jd = djinni_per_jd()
    out += ["## 5. Real-data benchmark: Djinni, nDCG@10 per JD (recomputed from the stored orders)", "",
            "JD text: `.runtime/bench/djinni/jds.yaml` · CV: `.runtime/bench/djinni/cvs/` · labels: "
            "`data/eval/human_djinni_gold.json` · orders: `.runtime/eval/results_djinni/<system>.json` · protocol: "
            "`data/eval/djinni_production_v1.json`. Judged-only nDCG@10 on the unrounded mean of the two human grades. "
            "The 250 CV files here are byte-identical to the frozen production inputs (SHA-256 check passes). The orders "
            "in `.runtime/eval/results_djinni/` on this machine are the 2026-11-05 local runs, not the official production "
            "run (Gemini P6) behind `docs/DJINNI_PRODUCTION_RESULTS.md`, whose orders are not in git: B0 and B2 match it, "
            "P6 here is the local claude_code run.", "",
            *table(["JD", "tiêu đề", "cặp có nhãn", *names], per_jd), "",
            "Label provenance: the first Đán file in git (095c4ad) records `saw_model: true` on all 424 labels and Vũ's "
            "notes repeat Opus's note on 212 pairs, so these labels are human grades made with the model's suggestion "
            "available, not independent blind grades. The CVs and JDs themselves are real.", ""]
    return "\n".join(out)


def main() -> int:
    rows = measure()
    with CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    DOC.write_text(report(rows), encoding="utf-8")
    (RUNTIME / "real_evidence.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"→ {DOC} · {CSV} ({len(rows)} CVs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
