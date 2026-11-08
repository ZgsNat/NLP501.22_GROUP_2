# Annotation Guidelines for Real-World CV Information Extraction Benchmark

Version: 1.1 (Registered 2026-11-08)
Scope: Djinni Unseen Candidate Profiles Benchmark (50 Held-Out Profiles)

---

## 1. Objectives & Granularity

Information extraction on candidate CVs must decouple syntactic recognition from semantic evaluation. Each extraction annotation operates on four distinct levels:

1. **Mention Span (`mention_span`)**: The exact surface text substring appearing in the CV text, recorded with character offsets `[start, end)` and `raw_text`.
2. **Canonical Skill (`canonical_skill`)**: The normalized skill key defined in `data/skills_taxonomy.yaml`. If the skill is an authentic technical tool/concept written by the candidate but is absent from our taxonomy, it is recorded as `null` and flagged as **Out-Of-Vocabulary (OOV)**.
3. **Evidence Status (`evidence_status`)**:
   - `used`: The tool was actively applied in a past project or professional role (e.g., *"Built UI test automation suites using Selenium and Java"*).
   - `listed`: The tool is merely stated in a skills inventory, profile summary, or keyword list without contextual job narrative (e.g., *"Skills: Selenium, Postman, Git"*).
   - `hedged`: The tool mention is qualified by uncertainty, beginner status, self-study, or attribution to others (e.g., *"familiar with Cypress"*, *"currently learning Playwright"*, *"worked with backend developers writing Go"*).
4. **Candidate-Level Qualification (`candidate_confirmed`)**: A boolean judgement attached to each mention. It records whether that evidence supports crediting the candidate with the skill. `used` normally supports `true`, `hedged` normally supports `false`, and `listed` requires the annotator to judge whether the list represents the candidate's own stack. The export keeps this decision explicit rather than deriving it after the fact.

---

## 2. Span Boundary Rules

- **Minimal Informative Span**: Capture the complete tool/technology name without leading/trailing punctuation or conjunctions.
  - Correct: `Selenium WebDriver`, `Cypress.io`, `Postman`, `REST Assured`.
  - Incorrect: `using Selenium WebDriver, and` (contains extraneous tokens).
- **Sub-skill & Compound Terms**:
  - For compound toolchains like `Selenium with Java`, extract two distinct spans: `Selenium` (tool) and `Java` (language).
  - Version numbers attached to tools should be included in the raw text span (e.g., `JUnit 5`, `Python 3.10`), but mapped to the canonical base tool (`junit`, `python`).

---

## 3. Evidence Classification Criteria

### 3.1 `used`
- Preceded or followed by direct action verbs: *developed, implemented, executed, configured, automated, designed, maintained, integrated*.
- Embedded within a dated job role, project description, or task bullet.

### 3.2 `listed`
- Appears under sections titled *Skills, Technical Stack, Tools, Competencies, Highlights*.
- Appears in comma-separated or bulleted keyword inventories without narrative context.

### 3.3 `hedged`
- **NegEx-style Hedge Cues**:
  - Prior cues: *familiar with, basic knowledge of, exposure to, beginner, learning, tutorial, attended course, assisted with*.
  - Posterior cues: *(basic)*, *(novice)*, *(theoretical knowledge only)*.
- **Third-Party Attribution**: Mentions describing the work of colleagues, clients, or the wider engineering team rather than the candidate's own duties (*"collaborated with developers who maintained the Docker containers"* $\rightarrow$ Docker is hedged/third-party).

---

## 4. Inter-Annotator Agreement (IAA) Protocol

Two distinct people (Annotator A and Annotator B) annotate all 50 profiles independently. Each receives a differently ordered offline kit. They must not see system predictions, weak labels, the other person's labels, or benchmark scores before export. The export records `human=true`, `independent=true`, `saw_system_output=false`, annotator identity, document hashes, and protocol hashes.

The following metrics are registered before labels are opened:

1. **Span agreement:** exact-boundary precision/recall/$F_1$ and token-overlap precision/recall/$F_1$. Alignment is one-to-one and uses token Jaccard $\ge 0.5$.
2. **Canonical agreement:** exact canonical-skill agreement on aligned spans, with OOV names normalized only for whitespace and case.
3. **Status agreement:** **unweighted Cohen's $\kappa$** on aligned spans. `used`, `listed`, and `hedged` are nominal labels; no ordinal distance is assumed.
4. **Candidate agreement:** micro precision/recall/$F_1$, macro-$F_1$, and mean Jaccard over confirmed skill sets.
5. **OOV audit:** each annotator's OOV rate and OOV-versus-in-taxonomy agreement on aligned spans.

### Adjudication
Only after both files pass validation does the tool reveal a conflict-only kit. A third person, distinct from A and B, resolves every boundary, canonical, status, candidate-confirmation, and unmatched-span conflict. No heuristic or model may adjudicate. The builder refuses incomplete decisions and refuses to overwrite an existing frozen Gold file.

The official **Human-Adjudicated Reference** exists only after this gate succeeds. Before then, the benchmark status is `HUMAN_ANNOTATION_PENDING`; generated weak labels may be used only for debugging and cannot support IAA, model selection, or headline results.
