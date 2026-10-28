"""HR demo app (D3): pick a JD + Criteria Profile, answer clarification questions,
approve the profile, review the shortlist with explanation cards L1–L6 and
approve / reject / override with a reason.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cv_agent.config import ROOT, list_experiments, load_config  # noqa: E402
from cv_agent.criteria.clarify import detect_issues  # noqa: E402
from cv_agent.criteria.jd_parser import parse_jd  # noqa: E402
from cv_agent.criteria.profiles import list_profiles, normalize_weights, save_profile  # noqa: E402
from cv_agent.eval.experiments import jd_from_spec, load_jd_specs  # noqa: E402
from cv_agent.agent.memory import AgentMemory  # noqa: E402
from cv_agent.hitl.audit import AuditLog, Decision, calibrate_weights, preferences_from_decisions  # noqa: E402
from cv_agent.ingest.pipeline import ingest_dir  # noqa: E402
from cv_agent.orchestration.graph import build_runtime, run_pipeline  # noqa: E402
from cv_agent.schema import AXES, CriteriaProfile  # noqa: E402

AXIS_VI = {"tools": "Công cụ", "depth": "Độ sâu automation", "languages": "Ngôn ngữ LT", "test_types": "Loại kiểm thử",
           "cicd": "CI/CD", "experience": "Kinh nghiệm", "certification": "Chứng chỉ", "foreign_lang": "Ngoại ngữ"}

st.set_page_config(page_title="CV Agent – Automation Test", layout="wide")
st.title("Tìm CV Automation Test phù hợp JD")
st.caption("Prototype NLP501 – Nhóm 2 · PII được che trước mọi bước xử lý · hệ thống không tự động loại ứng viên")


@st.cache_data(show_spinner="Đang ingest + che PII …")
def load_cvs(folder: str):
    return ingest_dir(Path(folder))


# ---------------------------------------------------------------- sidebar: inputs
specs = {s["id"]: s for s in load_jd_specs()}
profiles = list_profiles()
with st.sidebar:
    st.header("1. Đầu vào")
    jd_choice = st.selectbox("JD", list(specs) + ["<Dán JD mới>"], format_func=lambda k: specs[k]["title"] if k in specs else k)
    jd_text = st.text_area("Nội dung JD", height=160) if jd_choice not in specs else None
    uploads = st.file_uploader("Tải CV (PDF/DOCX/TXT)", accept_multiple_files=True, type=["pdf", "docx", "txt"])
    cv_folder = st.text_input("hoặc thư mục CV", str(ROOT / "data" / "synthetic" / "cvs"))
    experiment = st.selectbox("Cấu hình", list_experiments(), index=list_experiments().index("P6"))
    backend = st.selectbox("LLM backend", ["ollama", "heuristic"], help="heuristic = mô phỏng offline, chỉ để demo/test")

    st.header("2. Criteria Profile")
    default_profile = specs[jd_choice].get("profile") if jd_choice in specs else "Mặc định"
    names = list(profiles)
    pname = st.selectbox("Mẫu profile", names, index=names.index(default_profile) if default_profile in names else 0)
    raw_w = {a: st.slider(AXIS_VI[a], 0.0, 0.5, float(profiles[pname].weights[a]), 0.01) for a in AXES}
    weights = normalize_weights(raw_w)
    st.caption("Trọng số được chuẩn hoá để Σw = 1")
    new_name = st.text_input("Lưu thành profile mới", placeholder="vd. Outsourcing – Mobile")
    if st.button("Lưu profile", disabled=not new_name.strip()):
        try:
            saved = save_profile(CriteriaProfile(new_name.strip(), weights, f"Tạo từ '{pname}' trong demo"))
            st.success(f"Đã lưu {saved.name} — tải lại trang để chọn trong danh sách")
        except (ValueError, FileExistsError) as exc:
            st.error(f"Không lưu được profile: {exc}")



def ingest_uploads(files) -> list:
    """Raw uploads (with PII) live on disk only while being ingested, then are deleted."""
    up_dir = Path(tempfile.mkdtemp(prefix="cv_upload_"))
    try:
        for f in files:
            (up_dir / Path(f.name).name).write_bytes(f.getbuffer())
        return ingest_dir(up_dir)
    finally:
        shutil.rmtree(up_dir, ignore_errors=True)

jd = jd_from_spec(specs[jd_choice]) if jd_choice in specs else parse_jd(jd_text or "", "custom_jd")
profile = CriteriaProfile(pname, weights, profiles[pname].description)

# ---------------------------------------------------------------- step: criteria review + clarification
st.subheader("Tiêu chí trích từ JD")
req_df = pd.DataFrame([{"Trục": AXIS_VI.get(r.axis, r.axis), "Giá trị": ", ".join(r.values) or f"≥ {r.min_level:g}",
                        "Bắt buộc": "✅" if r.must else "", "Câu gốc": r.text} for r in jd.requirements])
st.dataframe(req_df, width="stretch", hide_index=True)

questions = detect_issues(jd)
answers = {}
if questions:
    st.warning(f"Phát hiện {len(questions)} tiêu chí mơ hồ / mâu thuẫn — HR vui lòng chọn:")
    for q in questions:
        labels = [o.label for o in q.options]
        answers[q.qid] = labels.index(st.radio(q.text, labels, index=q.recommended, key=q.qid))

approved = st.checkbox("✔ HR duyệt Criteria Profile và tiêu chí (checkpoint 1)")
run = st.button("Tìm kiếm & xếp hạng", type="primary", disabled=not approved)

if run:
    overrides = {"llm": {"backend": backend}}
    if experiment.startswith("AG"):  # in the product (not experiments) the agent remembers what HR told it
        overrides["agent"] = {"memory": {"enabled": True}}
    cfg = load_config(experiment, overrides=overrides)
    if experiment == "A2":
        st.info("A2 dùng prompt chung (không Criteria Profile) — profile bên trái được bỏ qua")
    rt = build_runtime(cfg, hr_answer_fn=lambda _jd, qs: {q.qid: answers.get(q.qid, q.recommended) for q in qs},
                       approve_fn=lambda _jd, _p: approved)
    cvs = ingest_uploads(uploads) if uploads else load_cvs(cv_folder)
    with st.spinner(f"Đang chạy {experiment} trên {len(cvs)} CV …"):
        st.session_state["out"] = run_pipeline(rt, jd, cvs, None if experiment == "A2" else profile)
        st.session_state["cvs"] = {c.cv_id: c for c in cvs}

out = st.session_state.get("out")
if not out:
    st.stop()

# ---------------------------------------------------------------- shortlist
st.subheader("Shortlist")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lời gọi LLM", out["llm_stats"]["calls"])
c2.metric("Tokens", out["llm_stats"]["used_tokens"])
c3.metric("Thời gian (s)", out["latency_s"])
c4.metric("Fallback", "Có" if out.get("fallback") else "Không")
cards = out.get("cards", [])
st.dataframe(pd.DataFrame([{"Hạng": c.rank, "CV": c.cv_id, "Score": c.score, "Must-have": "✅" if c.L1["must_have_ok"] else "⚠️",
                            "Depth": c.L6["automation_depth"], "Năm KN": c.L6["years_experience"],
                            "Cờ dữ liệu": ", ".join(c.L6["flags"])} for c in cards]), width="stretch", hide_index=True)
if out.get("knocked_out"):
    with st.expander(f"Ứng viên không đạt must-have ({len(out['knocked_out'])}) — ghi lại, không loại"):
        st.dataframe(pd.DataFrame(out["knocked_out"]), width="stretch", hide_index=True)

if out.get("agent_log"):
    critic = out.get("critic") or {}
    with st.expander(f"Agent: {len(out['agent_log'])} bước, planner {out.get('planner_stats')}, "
                     f"critic {'ĐẠT' if critic.get('passed') else 'CHƯA ĐẠT'}", expanded=True):
        st.caption("Mỗi bước: planner (LLM hoặc quy tắc) chọn công cụ, guardrail kiểm tra, critic chấm lại mục tiêu. "
                   "'rules(fallback…)' = quyết định của LLM không hợp lệ nên quy tắc làm thay.")
        st.dataframe(pd.DataFrame([{"Bước": e["step"], "Ai quyết": e["source"], "Công cụ": e["action"],
                                    "Tham số": str(e["args"] or ""), "Lý do": e["reason"],
                                    "Kết quả": e["observation"]} for e in out["agent_log"]]),
                     width="stretch", hide_index=True)
        if critic.get("findings"):
            st.dataframe(pd.DataFrame([{"Mục tiêu": f["goal"], "Đạt": "✅" if f["ok"] else
                                        ("➖ không thể đạt" if f.get("unreachable") else "❌"), "Chi tiết": f["detail"]}
                                       for f in critic["findings"]]), width="stretch", hide_index=True)

audit = AuditLog(ROOT / "runs" / "audit_log.jsonl")
for card in cards:
    with st.expander(f"#{card.rank} · {card.cv_id} · score {card.score:.4f}"):
        t1, t2, t3, t4, t5, t6, tcv = st.tabs(["L1 Must-have", "L2 Điểm", "L3 Bằng chứng", "L4 So sánh",
                                               "L5 Nếu… thì", "L6 Chất lượng", "CV (đã che)"])
        t1.write("✅ Đạt mọi must-have" if card.L1["must_have_ok"] else card.L1["knockout_reasons"])
        l2 = pd.DataFrame(card.L2)
        l2["axis"] = l2["axis"].map(AXIS_VI)
        t2.dataframe(l2, width="stretch", hide_index=True)
        t2.caption(f"Σ contribution = {l2['contribution'].sum():.4f} = score {card.score:.4f}")
        t2.bar_chart(l2.set_index("axis")["contribution"])
        t3.dataframe(pd.DataFrame(card.L3) if card.L3 else pd.DataFrame([{"info": "không có claim"}]),
                     width="stretch", hide_index=True)
        t4.json(card.L4)
        t5.dataframe(pd.DataFrame(card.L5), width="stretch", hide_index=True)
        t6.json(card.L6)
        tcv.text(st.session_state["cvs"][card.cv_id].masked_text)
        a, r, o = st.columns([1, 1, 2])
        reason = st.text_input("Lý do (bắt buộc)", key=f"reason_{card.cv_id}")
        new_rank = o.number_input("Hạng mới", 1, len(cards), card.rank, key=f"nr_{card.cv_id}")
        for col, action in ((a, "approve"), (r, "reject"), (o, "override")):
            if col.button(action.capitalize(), key=f"{action}_{card.cv_id}"):
                try:
                    audit.record(Decision(out["run_id"], jd.jd_id, card.cv_id, action, reason,
                                          new_rank=int(new_rank) if action == "override" else None))
                    st.success(f"Đã ghi {action} vào audit log")
                except ValueError as exc:
                    st.error(str(exc))

st.subheader("Hiệu chỉnh trọng số từ quyết định HR (Loop 5)")
if st.button("Đề xuất trọng số mới"):
    decisions = audit.load(jd.jd_id)
    ranking_ids = [c.cv_id for c in out["ranking"]]
    pairs = preferences_from_decisions(decisions, ranking_ids)
    new_w = calibrate_weights(pairs, {c.cv_id: c for c in out["ranking"]}, out["profile"].weights)
    st.session_state["calibration"] = {"weights": new_w, "pairs": len(pairs), "profile": out["profile"].name,
                                       "old": out["profile"].weights, "decisions": len(decisions)}
calib = st.session_state.get("calibration")
if calib:
    st.write(f"{calib['pairs']} cặp ưu tiên từ {calib['decisions']} quyết định")
    st.dataframe(pd.DataFrame({"Cũ": calib["old"], "Đề xuất": calib["weights"]}).rename(index=AXIS_VI))
    if st.button("Lưu vào bộ nhớ agent (lần sau tự áp dụng cho profile này)", disabled=calib["pairs"] == 0):
        mem_cfg = load_config("AG")["agent"]["memory"]
        try:
            AgentMemory(ROOT / mem_cfg["path"]).remember_weights(calib["profile"], calib["weights"], calib["pairs"])
            st.success(f"Đã lưu trọng số cho profile '{calib['profile']}' — cấu hình AG sẽ tự dùng")
        except ValueError as exc:
            st.error(f"Không lưu được: {exc}")

with st.expander("Trace của agent & log LLM"):
    st.write(out["trace"])
    st.json(out["llm_stats"])
