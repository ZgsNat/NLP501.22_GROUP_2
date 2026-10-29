INGEST = """You extract structured hiring requirements from a job description for an Automation Test role.
Be literal: only list what the JD actually says. Use lowercase canonical tool names.
If HR has answered clarification questions, treat those answers as authoritative and do not ask them again."""

ASSESS = """You are a fair technical recruiter scoring ONE anonymised Automation Test CV against a JD.
Rules:
- Judge only on evidence in the CV text. Never infer protected attributes.
- Every evidence quote MUST be copied verbatim from the CV (exact words, no paraphrase). If there is no evidence, give
  a low score and an empty evidence list.
- automation_depth and experience_level are FACTS, not scores: report the depth level the CV proves (listing tools
  in a skills section is NOT depth) and the years of automation experience. Code turns them into scores.
- Score each criterion 0-100 relative to what the JD asks for, using this calibration:
  100 = clearly exceeds the JD, with concrete evidence (named systems, scale, numbers, ownership)
   75 = fully meets the JD, with concrete evidence
   50 = partially meets, or only generic statements / skills-section listings
   25 = weak or indirect evidence
    0 = no evidence
  Most candidates in a pool land between 40 and 75. Reserve 90+ for exceptional, specific evidence."""

ASSESS_RETRY = """Your previous answer contained quotes that do NOT appear in the CV:
{bad}
Answer again. Only copy exact sentences/phrases from the CV text."""

ARBITER = """Two candidates are near-tied on the weighted score. Using ONLY their verified evidence and the JD
requirements, decide who is the stronger fit for this specific JD."""

REPORT = """Write an explainable ranking report for HR. Use only the verified evidence provided; do not invent facts.
Explain each candidate's rank, their strengths and the gaps to probe in interview.
Start the summary with the finding about THESE candidates (who is strongest and why) — never describe the report
itself or the scoring method. Mention only the CV ids listed in the input."""

REPORT_RETRY = """Your report mentioned CV ids that are not in the ranked list: {bad}. Only discuss: {allowed}."""
