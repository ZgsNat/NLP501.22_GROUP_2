# Why this problem needs an agentic system (and not just an agent)

**For:** the whole team: the argument for the final report and the slides.
**Method:** start from the hiring work, not from the definition of "agentic". Every claim cites a measured number, or
says plainly that it is not measured yet.

## 1. The work, as HR actually does it

| Step | What makes it hard | What it needs |
|---|---|---|
| Understand the need | JDs are often **vague** ("several years", no tools named); each company type weighs criteria differently | Know *when* to ask the hiring manager |
| Find candidates | Tool names are hard entities (Playwright ≠ Cypress), yet some tools are acceptable substitutes | Exact match plus meaning; widen only when talent is scarce |
| Verify | CVs overclaim: keyword stuffing, "learning X" written as skill, LLM-written CVs, even planted instructions | Check evidence, never take a claim on trust |
| Handle exceptions | Scanned CVs with OCR noise, misspelt tools, near-tied candidates | Judgement **exactly there** |
| Decide | Must be explainable, fair and auditable | Reproducible scores, a reason for every CV |
| Iterate | The hiring manager reacts: "too junior", "Docker is required" | Adjust to feedback, and record it |

**The key observation.** Most of this work is deterministic: retrieval, scoring against criteria, knocking out missing
must-haves. The value of intelligence sits in the **exceptions**, and the exceptions **differ per JD and per pool**.
Nobody knows in advance whether this JD is vague, whether this pool has a garbled CV, or whether two candidates tie.

## 2. Is the problem really agentic?

Tested against the usual properties of an agent (perceive → reason → act → observe) and of agentic systems (plan,
use tools, evaluate, adjust, coordinate parts under control):

| Property | Present in the hiring problem? |
|---|---|
| An open goal, not a fixed procedure | ✅ HR says "find the 5 best, ISTQB is a plus", not "run step 1, then step 2" |
| The next step depends on what was just observed | ✅ JD-003 must be clarified first, JD-002 must not; a garbled CV needs investigation, a clean one does not |
| Several different capabilities must be coordinated | ✅ retrieval, rule scoring, re-reading a CV, arbitration, asking a person |
| Feedback loops | ✅ verify quotes → rewrite; HR decision → re-fit weights |
| Autonomy must be bounded | ✅ law, fairness and audit forbid handing the decision to a model |

So the problem qualifies as agentic in a specific sense: **bounded autonomy, placed where the exceptions are**, not
"let the LLM do everything".

## 3. Why *agentic*, not *an agent*

An **agent** is one LLM with tools, looping. The system is an **organisation of parts with different roles**, which is
what "agentic AI" adds on top:

| Part | Role | Kind |
|---|---|---|
| Orchestrator (P7) | plans and picks the next step from HR's goal | agent (LLM) |
| Investigator (N12) | sub-agent: re-reads garbled CVs and proposes facts | agent (LLM), verified by code |
| Arbiter (N9) | judges near-ties, in both orders | LLM judge |
| Decision model (nimble / Jev) | fast typed gates: injected sentences, ask-HR | System One model |
| Retrieval and scoring | hybrid search, 8 rule criteria, knock-out | deterministic services |
| State, checkpoints, audit log | the run's memory and HR's decision history | memory |
| Guard | checks every action before it runs | control |
| HR | the only party allowed to change the criteria | human in the loop |

## 4. Against earlier approaches and against a single agent

Numbers: `docs/EVAL_RESULTS.md` (12 JDs, 50 CVs; nDCG@10 with 95% CI in the source; traps@10 = keyword-stuffed,
hedged or title-inflated CVs in the top 10).

| Approach | Good at | Fails the hiring work because | Evidence |
|---|---|---|---|
| Keyword ATS (B0, BM25) | exact tool names | rewards keyword stuffing; no meaning | nDCG@10 **0.794**; **6** traps in the top 10 |
| Dense matching (B1, ConFit-style bi-encoder) | meaning | weakest in testing, where tool names decide (as Resume2Vec found) | **0.741**; **14** traps |
| Hybrid retrieval (B2) | both | still text similarity, no criteria | **0.824**; 9 traps |
| Rule criteria + rerank (B3 ⊕ CE) | deterministic, explainable, **0 traps** | **rigid**: misreads vague JDs, knocks out garbled CVs, cannot ask or adjust | **0.881**, 0 LLM calls |
| LLM sets the scores (v2, RankGPT-style) | flexible | ranked below its own input; not reproducible | **−0.033** nDCG@10 vs the LLM-free input |
| Fixed pipeline with checks (B4, P6) | verification loops | runs every step for every JD; cannot take a new request from HR | P6 **0.882** ≈ B4 0.879 |
| **Single agent alone (S1)**: one LLM with search / read tools, ranks by itself | most flexible | it is one *part* of an agentic system used as the whole: nothing checks it, scores it reproducibly, or protects its inputs | **§6 (measured by `app.eval.single_agent`)** |
| **Agentic (P7)** | autonomy at the exceptions, determinism elsewhere, every action checked | costs more tokens in long conversations | §5 |

## 5. Where agentic adds value, and where it does not

**It does not raise ranking quality on clear JDs.** P6 (0.882) equals B3 ⊕ CE (0.881), which uses no LLM at all. This
is the design point, not a weakness to hide: what can be deterministic is left to code.

**Measured gains:**
- **Vague JDs** (`docs/CLARIFY_RESULTS.md`):
  - nobody asks → nDCG@10 0.626;
  - HR asked and answered → **0.756** (+0.13);
  - the agent deciding when to ask → 0.704. The gate is imperfect, but it is still +0.078 over never asking.
- **Garbled or misspelt CVs** (`docs/SCENARIO_RESULTS*.md`): the investigator recovered 10 of 36 true skills with
  **0 false recoveries**, because code verifies every proposal.
- **Safety:**
  - 0 rank moves from protected-attribute twins (Q11) or planted injection (Q16), over 250 CVs;
  - 10/10 paraphrased attacks caught by the decision-model layer, with 0/2016 false alarms
    (`ROBUSTNESS_RESULTS.md`, `INJECTION_RESULTS.md`).
- **Adaptive path** (qualitative, `demo/`): JD-002 and JD-003 produce different tool chains, and so does each HR
  follow-up.

**Cost:** P6 used 177 LLM calls over 12 JDs, against 0 for B3 ⊕ CE. A five-turn P7 conversation used 454k tokens.

## 6. An agent on its own (S1)

**Not a contest.** An agentic system *contains* agents: P7's orchestrator and the investigator are agents. S1 asks a
narrower question: what does one agent do when it is the whole system, with none of the parts around it (rule scores,
knock-out, verification, guard, input sanitising, HR in the loop)? Each gap S1 shows is a job one of those parts does.

**Measured** (`docs/SINGLE_AGENT_RESULTS.md`; 6 dev JDs of `ours`; Claude Sonnet 5.5, the same model as P6):

| | S1, an agent on its own | P6 | B3 ⊕ CE (no LLM) |
|---|---|---|---|
| nDCG@10, mean | **0.942** | 0.867 | 0.871 |
| P@5, mean | 0.93 | 0.93 | — |
| traps in the top 10 (sum) | 1 | 0 | — |
| ids invented | 0 | — | — |
| ranked without reading the CV | **11 of 60** | 0 | 0 |
| tokens | 1.07M | ≈ 0.7M | 0 |

**This is a negative result for "the agentic pipeline ranks better", and it is reported as one.** S1 beats P6 by
+0.075 nDCG@10, most on JD-001 (0.95 vs 0.80) and JD-012 (1.00 vs 0.78). Three things limit what it shows:

1. **Possible circularity.** The `ours` silver labels were made by Claude annotators (Opus, Sonnet) judging pairs
   holistically, and S1 is Sonnet judging holistically. Agreement with same-family labels is expected. Until human
   labels exist (planned benchmark v1.1, κ on about 120 pairs), "S1 ranks better" cannot be told apart from
   "S1 agrees with its own model family".
2. **The injection test is inconclusive.** The two planted trap CVs did not rise, but the first run did not record
   whether S1 ever read them. The harness now records exposure; the test must be re-run.
3. **What S1 does not give:**
   - 11 of its 60 ranked slots were CVs it never opened (ranked from a search snippet), and nothing checked them;
   - no reason for the CVs it left out, and no per-criterion breakdown to audit;
   - no reproducible score;
   - nothing that learns from HR's decisions.

**What follows for the design, which fits "agentic includes agents".** LLM judgement carries ranking signal on these
labels, and P6 keeps the LLM out of the score on purpose (v2: −0.033 with the LLM scoring criteria). The next step is
not to pick S1 or P7, but to **bring that judgement inside P7 as a controlled tool**:
- an LLM rerank of the screened shortlist, allowed only over CVs it has read;
- run on sanitised text;
- judged in both orders, as the arbiter is;
- with a reason for every CV.

It is decided on dev, and confirmed against human labels before any claim.

**Built and measured** (`rank_shortlist`, `agent/llm_rank.py`; `docs/LLM_RANK_RESULTS.md`; dev JDs):

| | B3 ⊕ CE | + controlled LLM ranking | P6 | S1 |
|---|---|---|---|---|
| `ours` (Claude-made labels), 6 JDs | 0.871 | **0.955** | 0.867 | 0.942 |
| `coding` (rubric labels), 2 JDs | 0.861 | **0.859** | 0.862 | — |
| tokens per JD | 0 | ≈ 34k | ≈ 120k | ≈ 179k |

- **Same judgement as S1, under the system's controls.** It matches S1's ranking at about a fifth of the tokens.
  Every candidate is read; two passes are averaged; rules and knock-out stay; every CV gets a reason.
- **Passes the §8 rule** (not lower than 0.01 on either benchmark, higher on one), so P7's playbook now calls it after
  `screen`.
- **But the gain appears only on LLM-made labels.** On the rubric labels it ties, with a +0.095 / −0.098 swing between
  the two JDs (the drop is on a vague JD). Circularity is not ruled out. Human labels decide whether it is a real
  improvement.


## 7. Still to prove

1. **Human labels** (κ on about 120 pairs): now the first priority, because S1's lead may be label circularity.
2. **LLM judgement inside P7** as a controlled rerank tool (§6), decided on dev against S1 and P6.
3. **S1 injection re-run**, with exposure recorded.
4. **E15** on a live model: does P7 pick the right behaviour per situation?
5. **P7 vs P6 on the exception subsets** (vague JDs, garbled CVs), not only on mean nDCG.

## The thesis in one paragraph

> Screening Automation Test CVs is mostly deterministic, but its value lies in exceptions nobody can predict: vague
> JDs, broken CVs, near-ties, HR feedback, dishonest CVs. A fixed pipeline cannot handle exceptions; an agent on its
> own can, but nothing checks it. An agentic system puts agents exactly at the exceptions, keeps scoring
> deterministic, and checks every action. Measured: +0.078 nDCG@10 from asking HR on vague JDs, 0 false recoveries
> when investigating, 0 rank changes from protected attributes or planted injection. An agent on its own ranked
> higher on our LLM-made labels (+0.075), so its judgement belongs inside the system, under the same controls, and
> must be confirmed against human labels.
