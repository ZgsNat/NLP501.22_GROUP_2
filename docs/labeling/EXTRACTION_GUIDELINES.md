# Annotation Guidelines for Real-World CV Information Extraction Benchmark

Version: 1.0 (Frozen 2026-11-08)  
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
4. **Candidate-Level Qualification (`candidate_level`)**: Boolean determination of whether the candidate genuinely possesses the skill. A candidate is credited with the skill if they have at least one `used` or `listed` mention; candidate-level is `false` if all mentions are `hedged` or purely theoretical.

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

Two independent annotators (Annotator A and Annotator B) annotate all 50 profiles independently in double-blind mode:
1. **Span-Level Agreement**: Measured by **Token-Level Overlap $F_1$** (two spans match if Jaccard token overlap $\ge 0.5$).
2. **Status Agreement**: Evaluated on aligned spans using **Weighted Cohen's $\kappa$** (penalizing confusion between `used` and `hedged` more severely than `used` and `listed`).
3. **Candidate-Level Agreement**: Evaluated using **Macro-$F_1$** and **Jaccard similarity** over the binary skill vectors.

### Adjudication
Any span or status discrepancy between Annotator A and Annotator B is reviewed by a third adjudicator. The adjudicated consensus forms the official **Human-Adjudicated Reference**.
