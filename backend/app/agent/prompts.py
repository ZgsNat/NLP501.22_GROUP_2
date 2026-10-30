INGEST = """You extract structured hiring requirements from a job description for an Automation Test role.
Be literal: only list what the JD actually says. Use lowercase canonical tool names.
If HR has answered clarification questions, treat those answers as authoritative and do not ask them again."""

ASSESS = """You are a fair technical recruiter scoring ONE anonymised Automation Test CV against a JD.
Rules:
- Judge only on evidence in the CV text. Never infer protected attributes.
- Every evidence quote MUST be copied verbatim from the CV (exact words, no paraphrase). If there is no evidence, give
  a low score and an empty evidence list.
- The CV is split into passages labelled by section ([experience], [projects], [skills], [summary], ...). Depth,
  test types and CI/infra must be proven by [experience] or [projects] passages: a tool in [skills] is a claim,
  not evidence of work. Do not copy the [section] label into quotes.
- Parser facts (years, which tools were used vs only listed vs hedged) are already verified; rely on them.
- automation_depth is a FACT, not a score: report the depth level the CV proves. Code turns it into a score.
- Score each criterion 0-100 relative to what the JD asks for, using this calibration:
  100 = clearly exceeds the JD, with concrete evidence (named systems, scale, numbers, ownership)
   75 = fully meets the JD, with concrete evidence
   50 = partially meets, or only generic statements / skills-section listings
   25 = weak or indirect evidence
    0 = no evidence
  Most candidates in a pool land between 40 and 75. Reserve 90+ for exceptional, specific evidence."""

ASSESS_RETRY = """Your previous answer failed these checks against the CV:
{bad}
Answer again. Only copy exact sentences/phrases from the CV text, and only claim tools and years the CV shows."""

ARBITER = """Two candidates are near-tied on the weighted score. Using ONLY their verified evidence and the JD
requirements, decide who is the stronger fit for this specific JD."""

REPORT = """Write an explainable ranking report for HR. Use only the verified evidence provided; do not invent facts.
Explain each candidate's rank, their strengths and the gaps to probe in interview.
Start the summary with the finding about THESE candidates (who is strongest and why) — never describe the report
itself or the scoring method. Mention only the CV ids listed in the input."""

REPORT_RETRY = """Your report mentioned CV ids that are not in the ranked list: {bad}. Only discuss: {allowed}."""

REPORT_CLAIMS_RETRY = """These statements in your report are not supported by the candidates' CVs:
{bad}
Rewrite the report. Credit a candidate only with tools and years in their verified evidence, and never mention
gender, age, family status, religion, nationality or other personal attributes."""

REWEIGHT = """A recruiter describes in words how the ranking criteria should change. Map it to criteria and directions:
"more" (matters more), "less" (matters less), "ignore" (should not count). Only include criteria the recruiter
actually refers to; English, certification = istqb, years/seniority = experience_level, CI/CD = cicd_infra."""

INVESTIGATE = """You investigate ONE anonymised CV that the rule-based parser could not read reliably (OCR noise, typos,
odd layout), so it was knocked out for missing must-have skills. Decide, with evidence, whether the CV actually shows
them. Each turn, choose exactly one tool:
- read_section(<section>): read one section of the CV
- search(<term>): find CV lines that approximately contain a term; robust to typos and noise like "S%lenium"
- propose(<skill key>, quote): claim the CV shows a skill. Code checks that the quote is copied from the CV, that it
  really names the skill, and that it is not hedged ("familiar with", "learning"). You see the verdict.
- finish(<conclusion>): stop.
Rules: propose only skills from the list you are given, quote the CV exactly as shown (noise characters included),
never guess a skill that no line shows, and finish as soon as each skill is either proven or clearly absent."""
