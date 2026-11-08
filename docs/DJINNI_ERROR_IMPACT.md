# Extraction-Induced Error Impact

Status: **HUMAN_GOLD_PENDING**.

The primary downstream analysis is intentionally blocked until the 50-profile Djinni extraction set has frozen human-adjudicated Gold. The historical 400-pair calculation used a heuristic-derived skill reference and cannot establish a human-grounded false-exclusion rate or rank correlation.

When Gold is available, the registered analysis will report:

- Oracle Skill Profile versus E0, E1, and E2;
- skill-score mean absolute error;
- Kendall's τ for score-induced ordering;
- top-k turnover;
- extraction-induced false exclusions with the eligible-pair denominator;
- false acceptances with the ineligible-pair denominator.

This analysis does not report nDCG because the extraction annotation is a skill profile, not a complete job-candidate relevance judgement. Investigator-agent rescue remains qualitative unless a quantitative run stores its actual traces and decisions.

`python -m app.eval.djinni_error_impact --weak-diagnostic` may still produce a local debugging report under `.runtime/`. That output is not committed and cannot be cited as Human Gold evidence.
