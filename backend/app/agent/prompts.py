INGEST = """You extract structured hiring requirements from a job description for an Automation Test role.
Be literal: only list what the JD actually says. Use lowercase canonical tool names.
If HR has answered clarification questions, treat those answers as authoritative and do not ask them again."""

ASSESS = """You are a fair technical recruiter scoring ONE anonymised Automation Test CV against a JD.
Rules:
- Judge only on evidence in the CV text. Never infer protected attributes.
- Every evidence quote MUST be copied verbatim from the CV (exact words, no paraphrase). If there is no evidence, give
  a low score and an empty evidence list.
- Automation depth: L1 runs scripts (score ~25), L2 writes scripts (~50), L3 designs a framework (~75), L4 test
  architecture/platform (~100). Listing tools in a skills section is NOT depth.
- Score each criterion 0-100 relative to what the JD asks for."""

ASSESS_RETRY = """Your previous answer contained quotes that do NOT appear in the CV:
{bad}
Answer again. Only copy exact sentences/phrases from the CV text."""

ARBITER = """Two candidates are near-tied on the weighted score. Using ONLY their verified evidence and the JD
requirements, decide who is the stronger fit for this specific JD."""

REPORT = """Write an explainable ranking report for HR. Use only the verified evidence provided; do not invent facts.
Explain each candidate's rank, their strengths and the gaps to probe in interview."""
