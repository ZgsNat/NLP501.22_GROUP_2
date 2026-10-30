# How we process CV text

A CV is a **semi-structured document**, not a bag of words. The same token means different things in different places:
- "Selenium" under *Experience* is evidence of work.
- Under *Skills* it is a claim.
- After "familiar with" it is a hedge.
- In "our developers run Selenium tests" it describes *someone else's* work.

v1 embedded each CV as one vector and gave the whole text to the LLM. v2 recovers the structure first, and every later step reads it.

```
raw file ─► X1 read_file ─► X2 normalize ─► X4 redact ─► X3 detect_language ─► X5 sections · entries · dates
         ─► X6 chunks ─► X7 skill mentions (+hedges) · years · depth cue ─► X8 CVProfile
                                                   │
       ┌───────────────────────────────────────────┼─────────────────────────────────────────────┐
  D9 retrieval over chunks       N3 knock-out on usable mentions     N6 LLM sees labelled passages     N7 evidence must be
  (one query per requirement)    N6 rule scores (tools, years)        + verified parser facts           grounded AND supported
```

Every stage is deterministic, costs zero tokens and is unit-tested (`backend/tests/test_nlp.py`, 51 cases in English and Vietnamese). Code: `backend/app/nlp/`. IDs refer to the architecture map, `docs/architecture.html`.

## Stages

| ID | Stage | Technique | Why |
|---|---|---|---|
| X1 | `read_file` | PDF text layer (pypdf); DOCX paragraphs and tables (python-docx); MD and TXT | HR has PDFs and Word files, not markdown. Scanned PDFs are rejected with an "OCR needed" error rather than ranked as empty. |
| X2 | `normalize` | Unicode NFC; strip zero-width characters and soft hyphens; expand ligatures; normalise bullets (`• ● ▪ + –` → `- `); de-hyphenate PDF line breaks; re-join hard-wrapped lines | PDFs and Word store Vietnamese diacritics *decomposed* (e + ◌̂ + ◌́). Without NFC, "kiểm thử" from a PDF and from a web form are different strings, so BM25, regexes and quote grounding fail silently. |
| X4 | `redact` | Remove "label: value" lines (EN and VN: name, birth date, age, gender, marital status, nationality, religion, health, address, photo); inline email, URL and phone patterns. Phones must start with `+cc` or a trunk `0`, so year ranges like `2019.05-2021` are never removed. v3.1: also the **unlabelled name heading the CV** (one of the first 3 lines with 2–5 capitalised tokens, no digits or contact punctuation, not a section heading or job title; masked wherever it recurs) and a **gender value under a garbled label** ("%%nd~r: Female"). | The fairness policy forbids protected attributes. *Asking* an LLM to ignore them is a hope; deleting them before indexing and prompting is a guarantee. Uploads persist only the redacted text. |
| X3 | `detect_language` | Share of words with Vietnamese-only letters or Vietnamese function words: `vi` ≥ 35%, `mixed` ≥ 6% | Routing and reporting. The corpus has 44 en, 5 vi and 1 mixed CV. |
| X5 | `sections`, `entries`, `date_range`, `merged_months` | A heading lexicon (≈90 EN and VN aliases, including bilingual headings like "Summary / Tóm tắt") accepted only on heading-like lines: markdown `#`, ALL CAPS, trailing colon, bold or short. Role and project entries are anchored on `###`, bold or dated lines. A date line attaches to the title line above it. The date grammar covers `Jan 2020 – Present`, `03/2021 - nay`, `2019/05–2021/08`, `Tháng 6/2019 – Tháng 2/2021`, `2018 to 2020` and `Since 2022`. Intervals are merged, and entries marked as a career break are excluded. | Years of experience are **computed from parsed dates**, not estimated by the LLM. A freelance job held alongside a full-time one is not double-counted. A career gap is neither counted nor penalised. All 50 CVs had parseable dated roles, giving 151 entries. |
| X6 | `chunking.structural` (also `window` and `whole` for the ablation) | One chunk per role or project entry and one per other section, split at line boundaries above 150 words. Each chunk carries a **contextual header** ("Senior QA \| experience: Automation Engineer — Bank (2018–2021)"). The header is embedded with the chunk but kept out of its text, so offsets stay exact. | See the next section. |
| X7 | `skill_mentions`, `is_hedged` | **Gazetteer NER** over a curated skills taxonomy (37 skills with aliases), with alphanumeric boundaries so "java" never matches inside "javascript". Each mention gets an **evidence strength**: `used` (in a role or project), `listed` (skills, summary) or `hedged`. | Tool names are a closed vocabulary, so a gazetteer beats a statistical NER here. Of 1,116 mentions, 592 are used, 455 listed and 69 hedged. |
| X7 | hedge scope | **NegEx-style scoping**, in EN and VN. A cue *before* the term ("familiar with", "currently learning", "followed a tutorial", "làm quen", "đang tự học", "khóa học") reaches to the end of the sentence or to "but / however / nhưng". A cue *after* the term applies to its own comma segment only, so "Java (basic), Selenium" hedges Java and not Selenium. "Our developers / a colleague" cues mark **someone else's work**. | This is what defeats keyword stuffing: a must-have that is only hedged does **not** pass the knock-out (H14). |
| X7 | `depth_cue` | Automation depth L1–L4 from **clause-initial action verbs**, i.e. the candidate's own predicate after an optional "I / we / tôi" subject. L1 and L2 cues count only in clauses about tools or scripts, never in "manual" clauses. | "Executed the existing *automated* suite" is L1: there "automated" is an adjective, not the verb. "Our developers run Cypress tests" is not the candidate. Inflated titles ("Automation Architect") are ignored, because only experience and project lines are read. |
| X8 | `CVProfile` | One object holding the processed text plus everything above | Single source of truth for offsets, grounding, rule scores, the LLM prompt and the UI's parse preview. |

## Chunking strategy (X6): what we tried and what we measured

| Strategy | Chunks per CV | Median tokens | Max tokens | Chunks over the 512-token encoder limit |
|---|---|---|---|---|
| `whole` (v1) | 1 | 343 | 1,417 | **11 of 50 CVs**: the long CVs were embedded truncated |
| `window` (120 words, 30 overlap) | 3.9 | 165 | 247 | 0 |
| `structural` (+ contextual header) | 6.8 | 69 | 285 | 0 |

Retrieval does **one query per requirement**: each must-have expanded with its aliases and weighted ×2, plus the role, the nice-to-haves and the test types. It keeps each CV's best chunk per query (max-pooling: a requirement is met if one passage meets it) and takes a weighted mean of the per-query normalised scores (`backend/app/rag/retrieve.py`, D9). The cross-encoder then scores (requirement, evidence passage) pairs, which are short, instead of a truncated whole CV.

What the evaluation found ([EVAL.md](EVAL.md), [EVAL_RESULTS.md](EVAL_RESULTS.md)):
- **The chunking strategy alone does not change top-10 quality on this corpus.** Structural vs whole, both with the cross-encoder, gives ΔnDCG@10 ≈ 0.00, with p ≈ 0.5.
- **The cross-encoder is the largest retrieval gain** (+0.05 to +0.08 nDCG@10).
- **Structure pays off downstream, not in retrieval.** It enables hedge-aware knock-out, rule scores from parsed facts, section-aware evidence checks and a passage per requirement to show HR. Removing keyword-stuffed or hedged CVs from the top 10 and improving the full order (Kendall's τ) comes from there.
- **A negative result we keep:** weighting skills and summary passages lower ("claims count less than evidence") halved trap CVs at retrieval but cost 0.06 nDCG@10 (significant). Strong candidates also name their tools mostly in those sections. The switch stays in the code, off, with the reason next to it.

## The JD side: requirements (N1, H2, X7)

The LLM reads the JD into `JDRequirements`, then deterministic code fixes what an 8B model gets wrong. We found each of these by inspecting the cached extraction for all 12 JDs:

| Problem | Example | Fix |
|---|---|---|
| Nice-to-have tools silently dropped: only phrases worded "or similar" survived canonicalization (a bug in our code, not the LLM) | JD-011 lost Kubernetes, Docker, Pact, k6, JMeter and ISTQB | Every taxonomy key in a nice-to-have phrase is kept |
| One phrase with two "one of" groups collapsed into one | "(Java, TypeScript or Python) and CI/CD (GitLab CI, GitHub Actions or Jenkins)" made CI tools count as a language | `canonicalize` reads each and-clause (`and`, `và`, `;`) on its own |
| Vietnamese "or" and "or similar" not recognised | "RestAssured **hoặc** Postman … Jenkins hoặc công cụ CI **tương tự**" became three hard must-haves, knocking out 7 CVs graded ≥ 2 | `hoặc` = or; `tương tự` and `tương đương` = soft |
| JD alternatives credited as substitutes | A TypeScript architect got 0.6 credit on "Java, TypeScript or Python" | Alternatives the JD names count in full; only L2 fallbacks get 0.6 (C1) |
| Required depth under-read | `depth_level = 1` for 8 of 12 JDs, including "ownership of a framework" | `jd_depth`: the CV depth lexicon over the JD's duty and must-have bullets (not nice-to-haves) sets a floor |

Measured effect on the deterministic systems (paired over 12 JDs): requirement-wise retrieval + CE improves (+0.021 nDCG@10, p = 0.003); B3 ⊕ CE is flat (−0.005, τ +0.012). That flat result is the net of JD-011 (+0.12, the bug) and JD-001 (−0.12).
- **JD-001:** our depth policy lists "CI quality gates that block releases" as L4, so "Define and enforce quality gates" makes the job L4, and L3 framework builders that the annotators graded 3 drop. We kept the a-priori rule (the highest level among duty bullets) rather than tune it to the labels. "Dominant level instead of max" is a hypothesis for new JDs.

### v3: a deterministic JD parser (X9), measured against the LLM

`app/nlp/jd.py` ports the line logic of the `coding` branch's parser (Đán):
- section headers and cue words (EN + VN) mark each line must / nice / other;
- benefits are dropped;
- each must line goes through `canonicalize`.

On our 12 JDs it reproduces the LLM's must-haves and "one of" groups exactly, at zero tokens. It is also more literal: JD-003's "several years" stays unknown, where the LLM invented 3.

It still lost the pre-registered comparison (EVAL.md §8) on both benchmarks. With the 8 rule criteria and the cross-encoder, `rules` scored −0.006 (ours) and −0.005 (coding) nDCG@10 against `llm`. So `JD_PARSER=llm` stays the default. The parser supplies only:
- the depth floor;
- foreign languages;
- domains.

`rules` is the option for running with no LLM at all.

## Data-quality flags (X10)

`app/nlp/quality.py`, ported from the `coding` branch, flags CVs whose score deserves a second look:
- short or garbled text;
- fewer than 3 skills;
- no experience section;
- no dated roles;
- near-duplicates (Jaccard over 5-word shingles ≥ 0.85).

Flags never remove a CV. They appear on its row in the ranking.

v3.1 thresholds, measured on the coding set (`docs/EXTRACTION_RESULTS.md`): *garbled* = more than 15% noisy tokens
(symbols inside words, "Aut%%ati#n"); garbled CVs score ≥ 55%, clean ones ≤ 3%. The letter-ratio test alone caught
none of the 7 garbled CVs. *Short* = under 50 words: at 120 every template CV (median 68 words) was flagged, so the
flag carried no information. Now 7/7 garbled CVs and 17/193 clean CVs are flagged, the latter all for having no dated
roles. A garbled CV that is knocked out is handed to the investigator (N12), which reads it with fuzzy, verified tools.

## Measured against external gold (v3.1)

`python -m app.eval.extraction` scores the pipeline against the generator truth of Đán's 200 CVs, which our code was
never developed on. Results are in `docs/EXTRACTION_RESULTS.md`:
- tool F1 0.978 overall, 0.994 on clean CVs, 0.192 on garbled ones;
- years MAE 0.79;
- depth 89% exact;
- 0% leak for every PII field.

`python -m app.eval.robustness` builds counterfactual twins of every CV. On both benchmarks, name, gender, age and
marital-status twins give identical masked input and 0 score changes (`docs/ROBUSTNESS_RESULTS.md`).

The first run of these evaluations found three bugs, now fixed and tested:
- **the bare-name leak** (100% of named CVs);
- **the garbled-text flag** that never fired;
- **double-spaced CVs losing role titles.** A blank line between "Principal QA Engineer, Shop" and its dates made
  the date line the title, which inflated automation years from 8.0 to 14.8 (`segment.entries` now looks back to the
  previous non-empty line).

## Faithfulness: grounded vs supported (N7, H15)

v1 counted a quote as faithful if it **existed** in the CV (difflib ≥ 0.85). v2 adds a second level:
- **Grounded:** the quote exists in a chunk, so we know *which section* it came from.
- **Supported:** it is grounded, it comes from a section that can prove the criterion (depth, test types and CI/infra need experience or project passages), and it names the right kind of entity (a language for `programming_language`, a CI or infra tool or pipeline term for `cicd_infra`).

An ungrounded quote sends the assessment back to the LLM (L4). A grounded but unsupported one caps the criterion at 50, the rubric's "only skills-section listings" level. The UI shows such a quote as *listed, not demonstrated*.

## Known limitations

- **Lexicon recall.** The depth cue misses unusual phrasing, so two plain-language L3 CVs read as L2. That is why depth is still judged by the LLM, with the cue as a cross-check and as the B3 baseline. Hedge cues are a closed list, so new phrasings need new cues and a test.
- **Vietnamese BM25.** `Qdrant/bm25` tokenises on whitespace with English stemming. Tool names are English in VN CVs, so must-have matching works, but Vietnamese prose gets no word segmentation. A VN word segmenter or a multilingual dense model is the next step.
- **"Present"** resolves to today's date, so synthetic CVs written in 2025 gain about a year.
- **No OCR.** Scanned PDFs are rejected rather than guessed.
- **"X with A, B or C".** `canonicalize` makes A, B and C alternatives to X. The right reading here is X plus one of A, B or C. Grouping alternatives by taxonomy category would fix this phrasing, but it would make JD-008's "RestAssured with Java, **or** Postman/Newman" stricter than the JD means. Both phrasings are common, so a fix needs a test for each.
- **Names elsewhere.** A bare name heading the CV is masked (v3.1); a name only inside prose (a signature line at the end, "Contact Nguyen …") is not. A NER-based name redactor would close that gap. Heavily garbled names survive in garbled form; those CVs are flagged.
