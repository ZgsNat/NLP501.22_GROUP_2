# Criteria for AI output (Q1–Q13)

How we decide whether what the LLM produced is acceptable, what the system does when it is not, and where the
measured value is. An output criterion is only useful if it is **operational**: a check someone can run, a threshold,
and an action on failure. Every row below has all three; the measured values are generated into
`docs/EVAL_RESULTS*.md` (section "AI-output scorecard") by `python -m app.eval.run report`, never typed by hand.

## 1. The control principle

The LLM is a powerful but unreliable component, so it gets a narrow job and every output crosses a checkpoint:

1. **It never sets the score.** All 8 criteria come from rules over the parsed CV (H20). The LLM explains, arbitrates
   near-ties and drafts the report. A wrong LLM sentence can mislead a reader; it cannot move a candidate.
2. **Typed in, canonical out.** Every call returns a pydantic schema (Q1); free text that drives logic is
   canonicalized at the boundary (H2) or chosen from a closed set (Q8).
3. **Every claim is checked against the CV, by code.** Quotes must exist (Q2) and prove the right thing (Q3);
   rationales and report sentences may not credit tools or years the parsed CV does not show (Q4, Q6) or mention
   protected attributes (Q5). These checks are deterministic: same input, same verdict, zero tokens.
4. **Every failure has a defined action**: retry with the exact failure (bounded), then drop or label. Nothing fails
   silently and nothing loops forever (every loop has a limit in `config.Settings`).
5. **Humans own the decision.** HR answers clarifications, accepts substitutes and approves or re-weights the ranking
   (interrupts L1, L2, L5); every HR decision is written to an audit log.
6. **The controls are measured too.** The scorecard reports each Q before and after the controls act, and no LLM is
   trusted as a judge until it agrees with humans (Q13).

## 2. The criteria

"first → final" in the scorecard = before and after the graph's own controls acted.

| ID | Criterion | Applies to | Check (how) | Target | Action on failure | Code |
|---|---|---|---|---|---|---|
| **Q1** | Output is schema-valid | every LLM call | pydantic structured output | 100% after retry | assess: retry once, then the CV goes to *needs review*; gates: safe default; report: deterministic summary | `nodes._assess_one`, `decider.decide`, `nodes.report` |
| **Q2** | Evidence quotes are grounded | `llm_review` quotes | fuzzy span match in the CV (≥ 0.85) | 100% final | L4: retry naming the failing quotes (≤ `max_verify_retries`), then the claim loses credit and is labelled | `scoring.check_evidence`, `nodes.verify_evidence` |
| **Q3** | Quotes support their criterion | `llm_review` quotes | the quote sits in the right section (experience/projects, not skills) and names the right entity | reported | labelled "listed, not demonstrated" | `scoring.SUPPORT` |
| **Q4** | Rationales are consistent with the parsed CV | `llm_review` rationales | (a) no tool the CV never mentions is credited to the candidate; (b) no years claim above the dated roles + 1 | 0% final | L4 retry with the exact contradiction, then `[unverified: …]` label | `checks.review_issues` |
| **Q5** | No protected attributes | every free-text output | EN/VI lexicon (gender, age, marital status, religion, nationality, ethnicity, disability, photo, birth date) | 0 kept | sentence removed at once (input side: redaction H18 before indexing) | `checks.protected`, `strip_protected` |
| **Q6** | Report is faithful | report | Q4 + Q5 per candidate (explanation and strengths); only ranked CV ids (foreign-id guard) | 0 final | regenerate once with the failing claims; then drop only those sentences/strengths | `checks.report_issues`, `sanitize_report` |
| **Q7** | Arbiter is not position-biased | arbiter verdicts | each near-tie judged in both orders | reported | verdicts disagree → rule order kept | `nodes.arbitrate` (H19) |
| **Q8** | Gate decisions are typed and right | ambiguity gate, investigator actions | closed set + confidence; accuracy on the scenario suite | ≥ 90% scenario accuracy | unusable output → safe default (`options[0]`) | `decider.decide`, `tests/test_scenarios.py` |
| **Q9** | Explanation matches the score | explanation card | decomposition sums to the total; what-if re-scores with the same rules | exact, by construction | — | `scoring.decomposition`, `counterfactuals` |
| **Q10** | Ranking agrees with people | final ranking | nDCG@10, P@5, R@20, MRR, τ vs human-validated labels, bootstrap CI | ≥ deterministic baseline B3 | a component that does not beat its baseline is not adopted (EVAL.md §8) | `app/eval/run.py` |
| **Q11** | Ranking ignores protected attributes | whole pipeline | counterfactual CV pairs differing only in name / gender / age cues | 0 rank change | treated as a release blocker | `app/eval/robustness.py` |
| **Q12** | Bounded cost | every run | LLM calls, tokens, latency p50/p95 per JD; loop counters | within `Settings` limits | loop limits stop the run on budget | `observability.UsageCounter` |
| **Q13** | An LLM judge is validated before use | silver labels, any LLM grader | weighted κ against human labels on a blind sample | κ ≥ 0.8 to replace humans; ≥ 0.6 to assist | below 0.6: not used as ground truth | `app/eval/metrics.weighted_kappa`, EVAL.md §2 |
| **Q14** | Orchestrator decisions are valid and allowed | P7 tool calls | known tool, schema-valid arguments, the tool's precondition (`Tool.check`), HR tools alone in their step; task-level right behaviour on the E15 suite | 0 unchecked calls run; E15 checks reported | refused with the reason as the tool result (the model adapts); budget hit → code answers HR (`finalize`) | `orchestrator.guard`, `tools.check_*`, `app/eval/agentic.py` |
| **Q16** | Untrusted text cannot steer the system (prompt injection) | CV and JD text reaching any LLM; agent tool calls | (a) instruction-like sentences removed at redaction (X11), optionally (a2) sentences a System One decision model scores P ≥ 0.9 (layer 1b); (b) requirement, weight and decision changes must be named in HR's own words, not in CV text or tool results (provenance); (c) scores and knock-outs stay with rules | injection twins: 100% identical input, 0 rank moves; 0 false alarms on clean CVs | sentence removed + CV flagged `prompt_injection` for HR; unauthorised change refused with the reason | `nlp/injection.py`, `tools.unsaid`, `app/eval/robustness.py` |
| **Q15** | What the agent tells HR is faithful | P7 `respond_to_hr` | shortlisted CVs ranked and explained (L4-verified); CV ids exist; per-CV sentences pass Q4; Q5 | 0 delivered failures | refused with the failing claims; the model rewrites | `tools.check_respond` |

## 3. What the checks do not catch

The Q4–Q6 checks are tuned for precision: they flag only what they can prove wrong. They are a **lower bound** on the
error rate, not the error rate.

- A sentence that states an absence or quotes the JD ("no Cypress evidence", "the JD asks for 5 years") is not
  checked for tools or years, so a hallucination phrased that way passes.
- Paraphrased overclaims ("a seasoned architect") are not tools or numbers, so Q4 cannot see them.
- Gendered pronouns are not in the Q5 lexicon (they would strip most rationales written by small models); the blind
  human audit counts them instead.
- Undated CVs are not checked for years.

The blind human audit (part D of the plan: two teammates rate a sample of explanations) measures what slips through,
so the report can state both numbers.

## 4. Why this was worth building: a regression it caught

While writing Q2's retry path we found that v3 had silently weakened the L4 loop. The retry prompt read the rejected
quotes from `criteria`, but since v3 the LLM's quotes live in `llm_review` and `criteria` holds rule evidence, which is
always grounded. The loop still ran, but it told the model nothing about what was wrong. Fixed in `_assess_call`, and
`tests/test_checks.py::test_assess_retry_prompt_lists_what_failed` now fails if it happens again.
