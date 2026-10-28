"""Synthetic Automation-Test CV generator + simulated two-annotator gold labels.

Purpose: let the pipeline and the evaluation harness run end-to-end before the
real Gold Set (D2) exists — the requirement's fallback plan ("bổ sung CV tổng
hợp, báo cáo tách riêng"). Every generated record is tagged ``synthetic``.
All names / emails / phones are fake.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

TOOLS_WEB = ["Selenium WebDriver", "Playwright", "Cypress", "WebdriverIO"]
TOOLS_MOBILE = ["Appium", "Espresso", "XCUITest"]
TOOLS_API = ["Postman", "Rest Assured"]
TOOLS_PERF = ["JMeter", "k6", "Gatling"]
TOOLS_RUNNER = ["TestNG", "JUnit", "Pytest", "Cucumber", "Robot Framework"]
LANGS = ["Java", "Python", "JavaScript", "TypeScript", "C#"]
CICD = ["Jenkins", "GitLab CI", "GitHub Actions", "Azure DevOps", "Docker"]
SURNAMES = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Vũ", "Đặng", "Bùi", "Đỗ", "Hồ"]
MIDDLE_M, MIDDLE_F = ["Văn", "Đức", "Minh", "Quốc"], ["Thị", "Ngọc", "Thu", "Mai"]
GIVEN_M, GIVEN_F = ["Hùng", "Nam", "Tuấn", "Khoa", "Phúc", "Long"], ["Lan", "Hương", "Trang", "Linh", "Thảo", "Vy"]


@dataclass
class Latent:
    """Ground-truth attributes of a synthetic candidate (never shown to the system)."""
    cv_id: str
    gender: str
    years: int
    depth: int
    tools: List[str]
    langs: List[str]
    test_types: List[str]
    cicd: List[str]
    istqb: bool
    english: bool
    japanese: bool
    lang_cv: str
    low_quality: bool = False
    name: str = ""
    rationale: List[str] = field(default_factory=list)


def _name(rng: random.Random, gender: str) -> str:
    mid, giv = (MIDDLE_M, GIVEN_M) if gender == "M" else (MIDDLE_F, GIVEN_F)
    return f"{rng.choice(SURNAMES)} {rng.choice(mid)} {rng.choice(giv)}"


def sample_latent(rng: random.Random, idx: int) -> Latent:
    gender = rng.choice("MF")
    years = rng.choice([0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 10])
    depth = max(1, min(4, 1 + (years >= 1) + (years >= 3 and rng.random() < 0.7) + (years >= 6 and rng.random() < 0.6)))
    focus = rng.choice(["web", "web", "mobile", "api", "perf"])
    tools = rng.sample(TOOLS_WEB, rng.randint(1, 2)) if focus in {"web", "api"} or rng.random() < 0.5 else []
    tools += rng.sample(TOOLS_MOBILE, 1) if focus == "mobile" or rng.random() < 0.15 else []
    tools += rng.sample(TOOLS_API, rng.randint(1, 2)) if focus == "api" or rng.random() < 0.5 else []
    tools += rng.sample(TOOLS_PERF, 1) if focus == "perf" or rng.random() < 0.15 else []
    tools += rng.sample(TOOLS_RUNNER, rng.randint(1, 2))
    if depth == 1:  # manual testers mention few automation tools
        tools = [t for t in tools if t in TOOLS_API or rng.random() < 0.3]
    test_types = sorted({"ui testing" if any(t in TOOLS_WEB for t in tools) else "manual testing",
                         *(["api testing"] if any(t in TOOLS_API for t in tools) else []),
                         *(["mobile testing"] if any(t in TOOLS_MOBILE for t in tools) else []),
                         *(["performance testing"] if any(t in TOOLS_PERF for t in tools) else [])})
    return Latent(
        cv_id=f"syn_{idx:03d}", gender=gender, years=years, depth=depth, tools=tools,
        langs=rng.sample(LANGS, rng.randint(1, 2)) if depth >= 2 else rng.sample(LANGS, rng.randint(0, 1)),
        test_types=test_types, cicd=rng.sample(CICD, rng.randint(1, 3)) if depth >= 3 else rng.sample(CICD, rng.randint(0, 1)),
        istqb=rng.random() < 0.35, english=rng.random() < 0.6, japanese=rng.random() < 0.15,
        lang_cv="vi" if rng.random() < 0.35 else "en", low_quality=rng.random() < 0.05,
    )


_DEPTH_EN = {1: "Executed manual test cases and reported defects in Jira.",
             2: "Wrote and maintained automated test scripts for regression suites.",
             3: "Designed and built a Page Object Model automation framework and integrated it into the CI pipeline.",
             4: "Defined the test automation strategy, led a team of 5 QA engineers and mentored juniors on automation."}
_DEPTH_VI = {1: "Thực hiện kiểm thử thủ công, viết test case và báo lỗi trên Jira.",
             2: "Viết automation test scripts cho bộ regression, kiểm thử tự động hằng ngày.",
             3: "Xây dựng framework automation theo Page Object Model, data-driven, tích hợp Jenkins pipeline.",
             4: "Trưởng nhóm QA 5 người, định hướng automation strategy và mentor thành viên mới."}


def render_cv(lat: Latent, rng: random.Random) -> str:
    lat.name = _name(rng, lat.gender)
    email = f"{lat.name.split()[-1].lower()}{rng.randint(10, 99)}@example.com".replace("ư", "u").replace("ơ", "o")
    phone = f"09{rng.randint(10_000_000, 99_999_999)}"
    vi = lat.lang_cv == "vi"
    start = 2026 - lat.years
    lines = [lat.name, "Automation Test Engineer" if lat.depth >= 2 else "QA/QC Engineer",
             f"Email: {email}", f"{'Điện thoại' if vi else 'Phone'}: {phone}",
             f"{'Giới tính' if vi else 'Gender'}: {('Nam' if lat.gender == 'M' else 'Nữ') if vi else ('Male' if lat.gender == 'M' else 'Female')}",
             f"{'Ngày sinh' if vi else 'Date of birth'}: {rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1988, 2003)}",
             "", "SUMMARY" if not vi else "TÓM TẮT"]
    years_phrase = (f"{lat.years} năm kinh nghiệm kiểm thử phần mềm." if vi else f"{lat.years} years of experience in software testing.")
    lines.append(years_phrase if lat.years else ("Mới tốt nghiệp, đam mê kiểm thử." if vi else "Fresh graduate passionate about testing."))
    if lat.years:
        lat.rationale.append(years_phrase)
    depth_line = (_DEPTH_VI if vi else _DEPTH_EN)[lat.depth]
    lat.rationale.append(depth_line)
    lines += ["", "KINH NGHIỆM" if vi else "EXPERIENCE"]
    if lat.years:
        lines.append(f"Company {rng.choice('ABCDEFG')} ({start} - {'nay' if vi else 'Present'})")
    lines.append(f"- {depth_line}")
    if lat.tools:
        t_line = (f"- Sử dụng {', '.join(lat.tools)} cho kiểm thử {', '.join(lat.test_types)}." if vi
                  else f"- Used {', '.join(lat.tools)} for {', '.join(lat.test_types)}.")
        lines.append(t_line)
        lat.rationale.append(t_line.strip("- "))
    if lat.cicd:
        c_line = f"- {'Chạy test trên' if vi else 'Ran tests on'} {', '.join(lat.cicd)}."
        lines.append(c_line)
        lat.rationale.append(c_line.strip("- "))
    lines += ["", "KỸ NĂNG" if vi else "SKILLS"]
    if lat.langs:
        lines.append(f"{'Ngôn ngữ lập trình' if vi else 'Programming'}: {', '.join(lat.langs)}")
    if lat.istqb:
        lines.append("Chứng chỉ: ISTQB Foundation Level (CTFL)" if vi else "Certification: ISTQB CTFL")
    langs = (["English (TOEIC 750)"] if lat.english else []) + (["Japanese JLPT N2"] if lat.japanese else [])
    if langs:
        lines.append(f"{'Ngoại ngữ' if vi else 'Languages'}: {', '.join(langs)}")
    lines += ["", f"{'Địa chỉ' if vi else 'Address'}: {rng.randint(1, 300)} Nguyen Trai Street, District {rng.randint(1, 12)}"]
    text = "\n".join(lines)
    if lat.low_quality:  # simulate a badly parsed PDF
        text = "".join(ch if rng.random() > 0.25 else rng.choice("#@%~") for ch in text[:400])
    return text


def gold_relevance(lat: Latent, jd_spec: Dict, rng: random.Random) -> Tuple[int, float]:
    """Hidden annotation rubric -> (grade 0..3, continuous fit). Independent of the system's scorer."""
    must = jd_spec["must"]
    tools_l = [t.lower() for t in lat.tools]
    has_any = lambda opts: any(any(o in t for t in tools_l) for o in opts)  # noqa: E731
    if lat.years < must.get("years", 0) or lat.depth < must.get("depth", 1):
        return 0, 0.0
    if must.get("tools_any") and not has_any(must["tools_any"]):
        return 0, 0.0
    if must.get("english") and not lat.english:
        return 0, 0.0
    nice = jd_spec.get("nice", {})
    fit = 0.3 * min(1, lat.depth / 4) + 0.2 * min(1, lat.years / max(must.get("years", 1) * 2, 1))
    fit += 0.15 * (sum(has_any([t]) for t in nice.get("tools", [])) / max(len(nice.get("tools", [])), 1))
    fit += 0.1 * bool(set(l.lower() for l in lat.langs) & set(nice.get("langs", [])))
    fit += 0.1 * (lat.istqb if nice.get("istqb") else 0.5)
    fit += 0.1 * (lat.japanese if nice.get("japanese") else lat.english)
    fit += 0.05 * bool(lat.cicd)
    fit += rng.gauss(0, 0.03) - 0.25 * lat.low_quality
    grade = 3 if fit >= 0.68 else 2 if fit >= 0.5 else 1
    return grade, round(fit, 4)


def generate(out_dir: Path, jd_specs: Dict[str, Dict], n: int = 200, seed: int = 42) -> Dict:
    """Write CV .txt files, gold.json (two annotators + adjudicated) and latent.json."""
    rng = random.Random(seed)
    cv_dir = Path(out_dir) / "cvs"
    cv_dir.mkdir(parents=True, exist_ok=True)
    latents = [sample_latent(rng, i) for i in range(n)]
    for lat in latents:
        (cv_dir / f"{lat.cv_id}.txt").write_text(render_cv(lat, rng), encoding="utf-8")
    # two exact-ish duplicates to exercise FR-06
    for src, dup in ((latents[0], "syn_dup_a"), (latents[1], "syn_dup_b")):
        (cv_dir / f"{dup}.txt").write_text((cv_dir / f"{src.cv_id}.txt").read_text(encoding="utf-8") + "\n",
                                           encoding="utf-8")
    gold: Dict[str, Dict] = {}
    for jd_id, spec in jd_specs.items():
        ann_a, ann_b, final, fit = {}, {}, {}, {}
        for lat in latents:
            g, f = gold_relevance(lat, spec, rng)
            ann_a[lat.cv_id] = g
            ann_b[lat.cv_id] = max(0, min(3, g + (rng.choice([-1, 1]) if g and rng.random() < 0.12 else 0)))
            final[lat.cv_id] = g  # adjudicated = annotator A after discussion
            fit[lat.cv_id] = f
        hr_rank = sorted((c for c in final if final[c] > 0), key=lambda c: (-final[c], -fit[c], c))
        gold[jd_id] = {"relevance": final, "annotator_a": ann_a, "annotator_b": ann_b, "hr_ranking": hr_rank,
                       "ambiguous": spec.get("ambiguous", False), "split": spec.get("split", "test")}
    meta = {"synthetic": True, "seed": seed, "n_cvs": n,
            "rationale": {lat.cv_id: lat.rationale for lat in latents},
            "groups": {lat.cv_id: {"gender": lat.gender, "lang_cv": lat.lang_cv} for lat in latents},
            "names": {lat.cv_id: lat.name for lat in latents}}
    (Path(out_dir) / "gold.json").write_text(json.dumps(gold, ensure_ascii=False, indent=1), encoding="utf-8")
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (Path(out_dir) / "latent.json").write_text(json.dumps([asdict(l) for l in latents], ensure_ascii=False, indent=1),
                                               encoding="utf-8")
    return {"cvs": n + 2, "jds": len(jd_specs)}
