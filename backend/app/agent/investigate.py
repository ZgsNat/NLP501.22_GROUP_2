"""N12 investigate (loop L6): a bounded, tool-using agent for CVs the parser could not read reliably.

Rules decide most of the pipeline because, on clean text, they are near-perfect (tool F1 0.994,
docs/EXTRACTION_RESULTS.md). They fail on noisy text: OCR garble ("S%lenium W#bDriver") and typos ("Selenuim") drop
tool recall to 0.111, and the CV is knocked out on a must-have it actually has. That is where judgement helps, so the
LLM gets this one open-ended job, with the controls of docs/AI_OUTPUT_CRITERIA.md:
  - it acts only through typed tools (Q1, Q8); the tools are deterministic and read only the redacted CV;
  - nothing it says is believed: every proposed skill is verified by code (quote grounded, the quote names the skill
    by fuzzy match, not hedged) before it becomes a fact (Q2, Q4);
  - it runs only when a deterministic trigger fires, on at most Settings.max_investigations CVs, for at most
    Settings.max_investigation_steps steps each, and stops as soon as every missing must-have is decided (Q12).
Verified facts enter the CV profile as ordinary skill mentions, so the rules, the knock-out and the cards use them
like any other evidence, and HR sees where each one came from."""

import dataclasses
import json
import re
from difflib import SequenceMatcher

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent import prompts
from app.agent.schemas import InvestigatorAction
from app.agent.scoring import locate
from app.nlp import extract, quality
from app.nlp.pipeline import CVProfile
from app.nlp.segment import ENTRY_SECTIONS
from app.nlp.taxonomy import aliases, skills
from app.nlp.taxonomy import key as tkey

TOKEN = re.compile(r"[^\s/,;()]+")  # "Postman/Newman" is two tools
NEAR = 0.8  # a token this similar to an alias, but not equal, is a near-miss spelling (typo or OCR noise)
MATCH = 0.75  # a proposed quote must contain the skill at least this similar


def _clean(s: str) -> str:
    return s.lower().strip(".,;:()[]\"'")


def similarity(term: str, line: str) -> tuple[float, int, int]:
    """Best SequenceMatcher ratio between `term` and any run of as many tokens in `line` → (ratio, start, end)."""
    toks = list(TOKEN.finditer(line))
    n, best = max(len(term.split()), 1), (0.0, 0, 0)
    for i in range(len(toks) - n + 1):
        span = line[toks[i].start():toks[i + n - 1].end()]
        r = SequenceMatcher(None, term.lower(), _clean(span), autojunk=False).ratio()
        if r > best[0]:
            best = (r, toks[i].start(), toks[i + n - 1].end())
    return best


def near_misses(p: CVProfile, terms: list[str]) -> dict[str, str]:
    """term → a CV line that almost, but not exactly, spells one of its aliases."""
    out = {}
    for t in terms:
        for line in p.text.split("\n"):
            low = line.lower()
            if any(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", low) for a in aliases(t)):
                continue
            if max(similarity(a, line)[0] for a in aliases(t)) >= NEAR:
                out[t] = line.strip()
                break
    return out


def triggers(p: CVProfile, missing: list[str], subs: dict[str, list[str]]) -> str | None:
    """Why a knocked-out CV deserves an investigation, or None: the parse is unreliable, or a missing must-have (or an
    accepted substitute) appears with a near-miss spelling. A plain absence is not investigated."""
    flags = set(quality.flags(p)) & {"garbled_text", "no_experience_section", "few_skills"}
    if "garbled_text" in flags:
        return "the text is garbled (OCR noise)"
    near = near_misses(p, [t for m in missing for t in [m, *subs.get(m, [])]])
    if near:
        k, line = next(iter(near.items()))
        return f"'{k}' looks misspelt in \"{line[:60]}\""
    return None


def with_facts(p: CVProfile, facts: list[dict]) -> CVProfile:
    """The profile plus verified facts, as ordinary skill mentions (chunk_id 'investigator' marks their origin)."""
    if not facts:
        return p
    sk = {k: list(v) for k, v in p.skills.items()}
    for f in facts:
        sk.setdefault(f["skill"], []).append(
            extract.Mention(f["skill"], "investigator", f["section"], f["strength"], f["quote"]))
    return dataclasses.replace(p, skills=sk)


# ── tools: deterministic, read-only ───────────────────────────────────────────────────────────────────────
def read_section(p: CVProfile, kind: str) -> str:
    secs = [s for s in p.sections if s.kind == kind.strip().lower()]
    if not secs:
        return f"No section '{kind}'. Sections: {sorted({s.kind for s in p.sections})}"
    return "\n".join(p.text[s.start:s.end] for s in secs)[:1500]


def search(p: CVProfile, term: str, limit: int = 5) -> str:
    scored = sorted(((similarity(term, ln)[0], ln.strip()) for ln in p.text.split("\n") if ln.strip()), reverse=True)
    hits = [f"[{r:.2f}] {ln}" for r, ln in scored[:limit] if r >= 0.6]
    return "\n".join(hits) or f"No line resembles '{term}'."


def verify(p: CVProfile, skill: str, quote: str, allowed: set[str]) -> tuple[dict | None, str]:
    """Code, not the model, decides whether a proposal becomes a fact."""
    k = tkey(skill)
    if k not in skills():
        return None, f"rejected: '{skill}' is not a skill in the taxonomy"
    if k not in allowed:
        return None, f"rejected: '{k}' is not one of the skills under investigation {sorted(allowed)}"
    grounded, section = locate(quote, p)
    if not grounded:
        return None, "rejected: the quote is not in the CV (copy the line exactly as shown)"
    r, a, b = max(similarity(al, quote) for al in aliases(k))
    if r < MATCH:
        return None, f"rejected: the quote does not name {k} (best match {r:.2f})"
    if extract.is_hedged(quote.lower(), a, b):
        return None, f"rejected: {k} is only hedged here (familiar with / learning), which does not count"
    section = section or "other"
    fact = {"skill": k, "quote": quote.strip(), "section": section, "similarity": round(r, 2),
            "strength": "used" if section in ENTRY_SECTIONS else "listed"}
    return fact, f"accepted: {k} ({fact['strength']} in {section}, match {r:.2f})"


async def investigate_cv(model, p: CVProfile, title: str, missing: list[str], subs: dict[str, list[str]],
                         why: str, max_steps: int) -> dict:
    """ReAct loop over typed actions → {"facts", "trace", "calls", "invalid"}. Stops at `finish`, at the step budget,
    or as soon as every missing must-have has a verified fact (the deterministic stopping rule)."""
    allowed = {tkey(t) for m in missing for t in [m, *subs.get(m, [])]}
    msgs = [SystemMessage(prompts.INVESTIGATE),
            HumanMessage(f"Role: {title}. Knocked out for missing {missing}"
                         + (f"; accepted alternatives {({m: subs[m] for m in missing if subs.get(m)})}" if any(subs.get(m) for m in missing) else "")
                         + f". Why investigate: {why}. Skills you may propose: {sorted(allowed)}. "
                         f"CV sections: {sorted({s.kind for s in p.sections})}. Budget: {max_steps} actions.")]
    facts, trace, seen, calls, invalid = [], [], set(), 0, False
    for step in range(1, max_steps + 1):
        try:
            calls += 1
            act: InvestigatorAction = await model.with_structured_output(InvestigatorAction).ainvoke(msgs)
        except Exception:  # Q1: an unusable answer ends the investigation; the knock-out stands (safe default)
            invalid = True
            break
        sig = (act.tool, act.arg.strip().lower(), act.quote.strip())
        if act.tool == "finish":
            trace.append({"step": step, "tool": "finish", "arg": act.arg, "thought": act.thought, "observation": ""})
            break
        if sig in seen:
            obs = "You already did exactly this. Choose a different action or finish."
        elif act.tool == "read_section":
            obs = read_section(p, act.arg)
        elif act.tool == "search":
            obs = search(p, act.arg)
        else:
            fact, obs = verify(p, act.arg, act.quote, allowed)
            if fact and fact["skill"] not in {f["skill"] for f in facts}:
                facts.append(fact)
        seen.add(sig)
        trace.append({"step": step, "tool": act.tool, "arg": act.arg, "quote": act.quote, "thought": act.thought,
                      "observation": obs[:300]})
        decided = {f["skill"] for f in facts}
        if all(m in decided or any(tkey(s) in decided for s in subs.get(m, [])) for m in missing):
            break  # every missing must-have now has a verified fact
        msgs += [AIMessage(json.dumps(act.model_dump(), ensure_ascii=False)), HumanMessage(f"Observation:\n{obs}")]
    return {"facts": facts, "trace": trace, "calls": calls, "invalid": invalid}
