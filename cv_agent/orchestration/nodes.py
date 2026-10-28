"""Graph nodes — one per stage. Each takes the shared state and returns a partial update."""
from __future__ import annotations

import hashlib
import logging
from dataclasses import replace
from typing import Any, Dict, List

from cv_agent.arbiter.arbiter import llm_rerank_all, selective_arbiter
from cv_agent.criteria.clarify import apply_answers, detect_issues, simulated_hr
from cv_agent.criteria.profiles import list_profiles, uniform_profile
from cv_agent.llm.backends import LLMError
from cv_agent.llm.gateway import BudgetExceeded
from cv_agent.orchestration.state import PipelineState, Runtime
from cv_agent.ranking.reranker import rerank
from cv_agent.ranking.scoring import must_have_ok, rank, score_candidate
from cv_agent.report.explain import build_cards
from cv_agent.retrieval.hybrid import HybridRetriever, build_query, expansion_terms
from cv_agent.schema import ScoredCandidate
from cv_agent.verify.claims import generate_claims
from cv_agent.verify.evidence import verify_claims

log = logging.getLogger(__name__)
# LLM failures that trigger the fixed-pipeline fallback instead of crashing the run (FR-26)
LLM_FAILURES = (BudgetExceeded, LLMError)


def _trace(state: PipelineState, msg: str) -> List[str]:
    log.info(msg)
    return list(state.get("trace", [])) + [msg]


def _resolve_profile(name: str):
    if name == "Uniform":
        return uniform_profile()
    profiles = list_profiles()
    if name not in profiles:
        raise KeyError(f"Profile {name!r} not found; available: {sorted(profiles)}")
    return profiles[name]


# ---------------------------------------------------------------- stage 2: criteria + clarification (Loop 1)

def criteria_node(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    ccfg = rt.cfg["criteria"]
    profile = state.get("profile") or _resolve_profile(ccfg.get("profile", "Mặc định"))
    questions = detect_issues(state["jd"]) if ccfg["clarification"]["enabled"] else []
    return {"profile": profile, "questions": questions,
            "trace": _trace(state, f"criteria: {len(state['jd'].must_haves)} must-have, "
                                   f"{len(state['jd'].nice_to_haves)} nice-to-have, {len(questions)} câu hỏi cho HR")}


def need_clarification(rt: Runtime, state: PipelineState) -> str:
    max_rounds = rt.cfg["criteria"]["clarification"]["max_rounds"]
    return "clarify" if state.get("questions") and state.get("clarify_rounds", 0) < max_rounds else "approve"


def clarify_node(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    questions = state["questions"]
    answers = rt.hr_answer_fn(state["jd"], questions) if rt.hr_answer_fn else simulated_hr(questions)
    jd = apply_answers(state["jd"], questions, answers)
    # a question answered once must not be asked again in the next round
    asked = set(state.get("hr_answers", {})) | set(answers)
    remaining = [q for q in detect_issues(jd) if q.qid not in asked]
    return {"jd": jd, "hr_answers": {**state.get("hr_answers", {}), **answers}, "questions": remaining,
            "clarify_rounds": state.get("clarify_rounds", 0) + 1,
            "trace": _trace(state, f"clarify: HR trả lời {answers}; còn {len(remaining)} câu hỏi")}


def approve_node(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    """HR checkpoint 1 (FR-10): profile must be approved before searching."""
    ok = rt.approve_fn(state["jd"], state["profile"]) if rt.approve_fn else True
    if not ok:
        raise PermissionError("HR did not approve the Criteria Profile; search aborted")
    return {"profile": replace(state["profile"], approved=True),
            "trace": _trace(state, f"approve: HR duyệt profile '{state['profile'].name}'")}


# ---------------------------------------------------------------- stage 3: hybrid retrieval (Loop 2)

def _retriever(rt: Runtime, state: PipelineState) -> HybridRetriever:
    rcfg = rt.cfg["retrieval"]
    digest = hashlib.sha1("|".join(f"{cid}:{cv.masked_text}" for cid, cv in sorted(state["cvs"].items()))
                          .encode("utf-8")).hexdigest()
    key = (rcfg["mode"], digest)
    if key not in rt.retriever_cache:
        rt.retriever_cache[key] = HybridRetriever(list(state["cvs"].values()), rt.encoder, rcfg["mode"], rcfg["rrf_k"])
    return rt.retriever_cache[key]


def retrieve_node(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    rcfg = rt.cfg["retrieval"]
    jd, expansion = state["jd"], list(state.get("expansion", []))
    pool = _retriever(rt, state).search(build_query(jd, expansion), rcfg["top_k"])
    must_pass = sum(1 for cid, _ in pool if must_have_ok(state["cvs"][cid], jd)[0])
    return {"candidate_pool": pool, "must_pass_count": must_pass,
            "trace": _trace(state, f"retrieve[{rcfg['mode']}]: pool={len(pool)}, đạt must-have={must_pass}, "
                                   f"expansion={expansion or '-'}")}


def need_expansion(rt: Runtime, state: PipelineState) -> str:
    qe = rt.cfg["retrieval"]["query_expansion"]
    if not qe["enabled"] or state.get("qe_rounds", 0) >= qe["max_rounds"]:
        return "rank"
    return "expand" if state.get("must_pass_count", 0) < qe["min_must_pass"] else "rank"


def expand_node(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    current = list(state.get("expansion", []))
    new_terms = [t for t in expansion_terms(state["jd"]) if t not in current]
    round_no = state.get("qe_rounds", 0) + 1
    # nothing new to add -> jump to max rounds so the loop terminates
    qe_rounds = round_no if new_terms else rt.cfg["retrieval"]["query_expansion"]["max_rounds"]
    return {"expansion": current + new_terms[:5], "qe_rounds": qe_rounds,
            "trace": _trace(state, f"query_expansion vòng {round_no}: +{new_terms[:5]}")}


# ---------------------------------------------------------------- stage 4: rerank + criteria scoring

def _rerank_within_criteria(rt: Runtime, state: PipelineState, pool) -> List[ScoredCandidate]:
    """rerank.mode = within_criteria: rank the whole pool by criteria first, then score only the top_n with the
    reranker and use it to order candidates the criteria score cannot separate (it never drops anyone)."""
    cvs, jd = state["cvs"], state["jd"]
    ranking = rank([score_candidate(cvs[cid], jd, state["profile"], s) for cid, s in pool])
    n = rt.cfg["rerank"]["top_n"]
    head, tail = ranking[:n], ranking[n:]
    scores = dict(rerank(rt.reranker, jd, [cvs[c.cv_id] for c in head], n))
    return rank([replace(c, rerank_score=scores.get(c.cv_id, 0.0)) for c in head]) + tail


def rank_node(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    cvs, jd, pool = state["cvs"], state["jd"], state["candidate_pool"]
    if not rt.cfg["criteria"]["enabled"]:  # B0-B2: retrieval order only
        ranking = [ScoredCandidate(cv_id=cid, retrieval_score=s, score=s) for cid, s in pool]
        return {"ranking": ranking, "knocked_out": [], "trace": _trace(state, "rank: retrieval order (no criteria)")}
    retr = dict(pool)
    rcfg = rt.cfg["rerank"]
    if rcfg["enabled"] and rcfg.get("mode", "filter") == "within_criteria":
        ranking = _rerank_within_criteria(rt, state, pool)
        note = f"criteria -> rerank[{rt.reranker.name}] trong top {rcfg['top_n']} (chỉ phá hoà điểm)"
    elif rcfg["enabled"]:
        top = rerank(rt.reranker, jd, [cvs[cid] for cid, _ in pool], rt.cfg["rerank"]["top_n"])
        head = rank([score_candidate(cvs[cid], jd, state["profile"], retr[cid], rs) for cid, rs in top])
        # the tail keeps retrieval order so Recall@k (k > top_n) stays measurable
        in_head = {c.cv_id for c in head}
        tail = [score_candidate(cvs[cid], jd, state["profile"], s) for cid, s in pool if cid not in in_head]
        ranking = head + tail
        note = f"rerank[{rt.reranker.name}] {len(pool)} -> {len(top)}"
    else:
        ranking = rank([score_candidate(cvs[cid], jd, state["profile"], s) for cid, s in pool])
        note = "no rerank"
    knocked = [{"cv_id": c.cv_id, "reasons": c.knockout_reasons, "potential_score": round(c.potential_score, 4)}
               for c in ranking if not c.must_have_ok]
    return {"ranking": ranking, "knocked_out": knocked,
            "trace": _trace(state, f"rank: {note}; knock-out={len(knocked)} (ghi lại, không loại)")}


# ---------------------------------------------------------------- stage 5: Selective Arbiter / LLM-all

def judge_node(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    mode, jcfg = rt.cfg["judge"]["mode"], rt.cfg["judge"]
    if mode in {"none", "pointwise"} or state.get("fallback"):
        return {"trace": _trace(state, f"judge: {mode} (không đổi thứ tự)")}
    profile = None if rt.cfg["criteria"].get("generic_prompt") else state["profile"]
    n_head = rt.cfg["rerank"]["top_n"] if rt.cfg["rerank"]["enabled"] else rt.cfg["report"]["top_n"] * 2
    head, tail = state["ranking"][:n_head], state["ranking"][n_head:]
    try:
        if mode == "listwise":
            res = llm_rerank_all(rt.gateway, state["jd"], profile, head, state["cvs"])
            msg = f"judge[listwise]: top {len(res.ranking)}"
        else:
            # all_pairs (LLM-all, H3 baseline) = same pairwise protocol on every adjacent pair
            threshold = float("inf") if mode == "all_pairs" else jcfg["near_tie_threshold"]
            # all_pairs judges the whole head; selective only the shortlist zone (judge.max_rank)
            max_rank = None if mode == "all_pairs" else jcfg.get("max_rank")
            res = selective_arbiter(rt.gateway, state["jd"], profile, head, state["cvs"], threshold, jcfg["passes"],
                                    max_rank, rt.cfg["llm"].get("cv_chars", 700))
            msg = (f"judge[{mode}]: {len(res.pairs_checked)} cặp được LLM xét, {len(res.swaps)} lần đổi chỗ, "
                   f"{res.consistent_pairs} cặp nhất quán khi đảo thứ tự")
    except LLM_FAILURES as exc:  # FR-26: fall back to the fixed pipeline ranking
        return {"fallback": True, "budget": rt.gateway.budget.snapshot(),
                "trace": _trace(state, f"judge: {type(exc).__name__} ({exc}) -> fallback pipeline cố định")}
    update = {"ranking": res.ranking + tail, "budget": rt.gateway.budget.snapshot(),
              "judge_stats": {"pairs": len(res.pairs_checked), "swaps": len(res.swaps),
                              "consistent_pairs": res.consistent_pairs, "interrupted": res.interrupted}}
    if res.interrupted:  # partial arbiter decisions are kept; later LLM stages are skipped
        update["fallback"] = True
        msg += f"; dừng giữa chừng ({res.interrupted}) -> giữ kết quả đã có, fallback"
    return {**update, "trace": _trace(state, msg)}


# ---------------------------------------------------------------- stage 6: claims + verify_evidence (Loop 4)

def evidence_node(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    if rt.cfg["judge"]["mode"] == "none" or not rt.cfg["criteria"]["enabled"] or not rt.cfg["report"].get("claims", True):
        return {"claims": {}, "trace": _trace(state, "evidence: bỏ qua (không sinh claim)")}
    cv_chars = rt.cfg["llm"].get("claims_cv_chars", 1000)
    cvs, jd = state["cvs"], state["jd"]
    top = state["ranking"][: rt.cfg["report"]["top_n"]]
    by_id = {c.cv_id: c for c in top}
    claims: Dict[str, list] = {}
    fallback = state.get("fallback", False)
    for c in top:
        if fallback:
            break
        try:
            claims[c.cv_id] = generate_claims(rt.gateway, jd, cvs[c.cv_id], c, cv_chars=cv_chars)
        except LLM_FAILURES as exc:
            fallback = True
            log.warning("claims stopped (%s): %s", type(exc).__name__, exc)
    vcfg = rt.cfg["verify"]
    if not vcfg["enabled"]:
        claims = {k: [replace(cl, status="unverified") for cl in v] for k, v in claims.items()}
        return {"claims": claims, "fallback": fallback, "budget": rt.gateway.budget.snapshot(),
                "trace": _trace(state, f"evidence: {sum(map(len, claims.values()))} claims (không verify)")}

    def regen(cv_id, failed, feedback, attempt):
        return generate_claims(rt.gateway, jd, cvs[cv_id], by_id[cv_id], failed, attempt, cv_chars)

    final, stats = verify_claims(claims, {k: cvs[k].masked_text for k in claims}, rt.nli,
                                 None if fallback else regen, vcfg["max_rounds"], vcfg["on_fail"],
                                 vcfg["fuzzy_threshold"])
    return {"claims": final, "verify_stats": stats, "fallback": fallback, "budget": rt.gateway.budget.snapshot(),
            "trace": _trace(state, f"verify_evidence: {stats}")}


# ---------------------------------------------------------------- stage 7: report

def report_node(rt: Runtime, state: PipelineState) -> Dict[str, Any]:
    if not rt.cfg["criteria"]["enabled"]:
        return {"cards": [], "trace": _trace(state, "report: baseline không có explanation card")}
    cards = build_cards(state.get("ranking", []), state["cvs"], state["jd"], state["profile"], state.get("claims", {}),
                        rt.gateway.records, rt.cfg["report"]["top_n"])
    return {"cards": cards, "budget": rt.gateway.budget.snapshot(),
            "trace": _trace(state, f"report: {len(cards)} explanation cards")}
