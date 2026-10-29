// Canned run for `?mock=1`: exercises clarification (L1), query expansion (L2), hallucination
// check (L4), arbiter (L3), review + HR re-score (L5) and done, with delays between events.
import type { Api } from './api'
import {
  CRITERIA, type Assessment, type Config, type Criterion, type NodeName, type RankRow,
  type ResumeBody, type Report, type StreamEvent, type Values,
} from './types'

const config: Config = {
  providers: [{ name: 'openai', configured: false }, { name: 'ollama', configured: true }],
  defaults: { reasoning: 'ollama:qwen3:8b', fast: 'ollama:qwen3:8b' },
  company_profiles: [{ key: 'default', label: 'Default (balanced)' }, { key: 'bank', label: 'Bank & Fintech' }],
  criteria: [
    { key: 'tools', label: 'Tools / frameworks (must-have)' }, { key: 'automation_depth', label: 'Automation depth (1–4)' },
    { key: 'programming_language', label: 'Programming language' }, { key: 'test_types', label: 'Test types (Web / Mobile / API)' },
    { key: 'cicd_infra', label: 'CI/CD & test infrastructure' }, { key: 'experience_level', label: 'Years of experience & seniority' },
    { key: 'istqb', label: 'ISTQB foundation' }, { key: 'domain_language', label: 'Domain knowledge & English' },
  ],
  sample_jds: [
    { id: 'JD-001', title: 'Senior Automation Tester (Bank)', company_profile: 'bank',
      text: 'We are looking for a Senior Automation Test Engineer for our internet banking platform.\nMust have Selenium, Java, REST API testing and CI/CD experience. Good automation skills required, 5+ years.' },
    { id: 'JD-002', title: 'Mobile QA Automation (Startup)', company_profile: 'default',
      text: 'Fast-moving mobile startup seeking a QA automation engineer. Appium, Python, Android/iOS.' },
  ],
  langfuse_url: 'http://localhost:3000',
}

const CVS: Record<string, { headline: string; text: string }> = {
  'CV-014': { headline: 'Senior Automation Test Engineer, 6 yrs, Selenium/Java', text:
`Senior Automation Test Engineer · 6 years

SUMMARY
Designed and maintained a Selenium WebDriver + Java framework with Page Object pattern for a banking web app.

EXPERIENCE
- Built REST Assured API suites (320 tests) running in Jenkins pipelines on every merge.
- Introduced Docker-based Selenium Grid, cutting regression from 6h to 55min.

CERTIFICATIONS
ISTQB Foundation Level (2021)

LANGUAGES
English: IELTS 6.5` },
  'CV-027': { headline: 'SDET, 5 yrs, Selenium/Python, API + Web', text:
`SDET · 5 years

SUMMARY
Built a Selenium + Python (pytest) framework covering web and API layers for a payments product.

EXPERIENCE
- Wrote 400+ pytest API tests with requests, executed by GitHub Actions on each pull request.
- Mentored 3 junior testers on automation best practice.

CERTIFICATIONS
ISTQB Foundation Level (2020)

LANGUAGES
English: fluent, daily stand-ups with EU team` },
  'CV-033': { headline: 'QA Engineer, 3 yrs, Cypress/JavaScript', text:
`QA Engineer · 3 years

SUMMARY
Front-end focused QA using Cypress and JavaScript.

EXPERIENCE
- Implemented Cypress end-to-end suite covering 40 flows for an e-commerce site.
- Manual regression and exploratory testing for mobile releases.

LANGUAGES
English: intermediate` },
  'CV-041': { headline: 'Manual Tester, 4 yrs, moving to automation', text:
`Manual Tester · 4 years

SUMMARY
Manual functional testing of banking back-office tools; currently learning automation.

EXPERIENCE
- Wrote 500 manual test cases in TestRail.
- Completed an online course on test automation basics.

LANGUAGES
English: good reading, limited speaking` },
}

// per CV, per criterion: [score, rationale, evidence quote?]
type Row = Record<Criterion, [number, string, string?]>
const ROWS: Record<string, Row> = {
  'CV-014': {
    tools: [94, 'Selenium WebDriver is the JD must-have and is used in production.', 'Selenium WebDriver + Java framework with Page Object pattern'],
    automation_depth: [88, 'Framework design plus grid infrastructure: level 3-4.', 'Docker-based Selenium Grid, cutting regression from 6h to 55min'],
    programming_language: [90, 'Java, matching the JD.'], test_types: [70, 'Web and API, no mobile.', 'REST Assured API suites (320 tests)'],
    cicd_infra: [88, 'Jenkins pipelines on every merge.', 'running in Jenkins pipelines on every merge'], experience_level: [90, '6 years vs 5+ required.', 'Senior Automation Test Engineer · 6 years'],
    istqb: [100, 'ISTQB Foundation held.', 'ISTQB Foundation Level (2021)'], domain_language: [65, 'Banking web app; English IELTS 6.5.', 'English: IELTS 6.5'],
  },
  'CV-027': {
    tools: [94, 'Selenium + pytest framework.', 'Built a Selenium + Python (pytest) framework'],
    automation_depth: [88, 'Framework ownership and mentoring.', 'Mentored 3 junior testers on automation best practice'],
    programming_language: [92, 'Python is a strong language fit.'], test_types: [80, 'Web + API.', 'covering web and API layers'],
    cicd_infra: [80, 'GitHub Actions on each PR.', 'executed by GitHub Actions on each pull request'], experience_level: [78, '5 years.', 'SDET · 5 years'],
    istqb: [100, 'ISTQB Foundation held.', 'ISTQB Foundation Level (2020)'], domain_language: [70, 'Payments domain; fluent English.', 'English: fluent'],
  },
  'CV-033': {
    tools: [70, 'Cypress, not Selenium; partial match.', 'Cypress and JavaScript'],
    automation_depth: [70, 'Claims team leadership of automation engineers.', 'Led a team of 5 automation engineers'],
    programming_language: [60, 'JavaScript only.'], test_types: [50, 'Web and manual mobile.', 'Manual regression and exploratory testing for mobile releases'],
    cicd_infra: [40, 'No CI/CD evidence.'], experience_level: [40, '3 years, below 5+.', 'QA Engineer · 3 years'],
    istqb: [0, 'No certification.'], domain_language: [55, 'E-commerce; intermediate English.', 'English: intermediate'],
  },
  'CV-041': {
    tools: [10, 'No automation tool used in production.', 'currently learning automation'],
    automation_depth: [10, 'Course only.', 'Completed an online course on test automation basics'],
    programming_language: [20, 'None stated.'], test_types: [30, 'Manual functional.'], cicd_infra: [0, 'None.'],
    experience_level: [40, '4 years, all manual.', 'Manual Tester · 4 years'], istqb: [0, 'None.'], domain_language: [60, 'Banking back-office.'],
  },
}
const WEIGHTS: Record<Criterion, number> = {
  tools: 0.2, automation_depth: 0.2, programming_language: 0.1, test_types: 0.1,
  cicd_infra: 0.1, experience_level: 0.1, istqb: 0.1, domain_language: 0.1,
}

const FIX_QUOTE = 'Implemented Cypress end-to-end suite covering 40 flows for an e-commerce site'

function assessments(pass: 1 | 2, verified: boolean): Record<string, Assessment> {
  const out: Record<string, Assessment> = {}
  for (const [id, row] of Object.entries(ROWS)) {
    const text = CVS[id].text.toLowerCase()
    const criteria = {} as Assessment['criteria']
    for (const c of CRITERIA) {
      let [score, rationale, quote] = row[c]
      if (id === 'CV-033' && c === 'automation_depth' && pass === 2) {
        score = 55; rationale = 'Single Cypress suite; no framework or CI evidence: level 2.'; quote = FIX_QUOTE
      }
      criteria[c] = { score, rationale, evidence: quote ? [{ quote, grounded: verified ? text.includes(quote.toLowerCase()) : null }] : [] }
    }
    out[id] = { criteria, knocked_out: id === 'CV-041', knockout_reason: id === 'CV-041' ? 'Missing must-have: Selenium (no automation tool in production use)' : null, attempts: pass }
  }
  return out
}

function ranking(w: Record<Criterion, number>): RankRow[] {
  const a = assessments(2, true)
  const rows = Object.keys(a).map((id) => ({
    cv_id: id, headline: CVS[id].headline, knocked_out: a[id].knocked_out,
    total: a[id].knocked_out ? 0 : CRITERIA.reduce((s, c) => s + w[c] * a[id].criteria[c].score / 100, 0),
  })).sort((x, y) => y.total - x.total)
  return rows.map((r, i) => ({
    ...r, rank: i + 1,
    arbiter_note: !r.knocked_out && i < 2 && Math.abs(rows[0].total - rows[1].total) < 0.05
      ? `Near-tie (Δ ${Math.abs(rows[0].total - rows[1].total).toFixed(3)}); arbiter compared ${rows[0].cv_id} vs ${rows[1].cv_id} on evidence depth.` : null,
  }))
}

const report: Report = {
  summary: 'Two strong Selenium candidates lead; one is knocked out for missing the must-have tool.',
  candidates: [
    { cv_id: 'CV-014', explanation: 'Production Selenium + Java framework with grid infrastructure and Jenkins CI.', strengths: ['Selenium WebDriver + Java', 'CI/CD in Jenkins', 'ISTQB'], gaps: ['No mobile testing'] },
    { cv_id: 'CV-027', explanation: 'Python SDET covering web and API with GitHub Actions; slightly less infrastructure depth.', strengths: ['Selenium + pytest', 'Mentoring', 'ISTQB'], gaps: ['Less grid/infra experience'] },
    { cv_id: 'CV-033', explanation: 'Cypress-focused QA; different toolchain and below the experience bar.', strengths: ['Cypress E2E'], gaps: ['No Selenium', 'No CI/CD evidence', '3 yrs < 5 yrs'] },
    { cv_id: 'CV-041', explanation: 'Manual tester without production automation.', strengths: ['Banking domain'], gaps: ['Selenium missing (knock-out)'] },
  ],
}

const requirements = (clear: boolean): Values['requirements'] => ({
  title: 'Senior Automation Test Engineer', must_have: ['Selenium', 'Java', 'REST API testing'],
  nice_to_have: ['Docker', 'ISTQB'], min_years: 5, depth_level: clear ? 3 : null,
  languages: ['Java'], test_types: ['Web', 'API'],
  ambiguities: clear ? [] : ['"Good automation skills": which depth level (1-4)?', 'Is mobile testing in scope?'],
})
const terms = ['Selenium', 'Java', 'REST API', 'CI/CD']
const termsExpanded = [...terms, 'WebDriver', 'RestAssured', 'Jenkins']
const policy = [{ source: 'bank_policy.md#screening', text: 'Banking roles require demonstrable production automation experience; certification is a tiebreaker.' }]
const cand = (n: number, rerank: boolean): Values['candidates'] =>
  Object.keys(CVS).slice(0, n).map((id, i) => ({ cv_id: id, headline: CVS[id].headline, retrieval_score: +(0.8 - i * 0.1).toFixed(2), rerank_score: rerank ? +(0.9 - i * 0.12).toFixed(2) : null, matched_must_have: i < 3 ? ['Selenium'] : [] }))

// --- event scripting ---------------------------------------------------------------------
type Step = StreamEvent
const node = (n: NodeName, ms: number, ...body: StreamEvent[]): StreamEvent[] =>
  [{ type: 'node_start', node: n }, ...body, { type: 'node_end', node: n, duration_ms: ms }]
const log = (n: NodeName, level: 'info' | 'loop' | 'warn', message: string): StreamEvent => ({ type: 'log', node: n, level, message })
const upd = (n: NodeName, update: Partial<Values>): StreamEvent => ({ type: 'update', node: n, update })
const loops = (clarify: number, expand: number, verify: number, arbitrate: number, review: number) => ({ clarify, expand, verify, arbitrate, review })

function scoring(w: Record<Criterion, number>, lp: Values['loops']): StreamEvent[] {
  const rk = ranking(w)
  return [
    ...node('score', 90, log('score', 'info', 'Score = knock-out × Σ w·S; CV-041 knocked out (missing Selenium)'), upd('score', { ranking: rk })),
    ...node('arbitrate', 1400,
      rk[0].arbiter_note
        ? log('arbitrate', 'loop', `L3 near-tie between ${rk[0].cv_id} and ${rk[1].cv_id}: calling arbiter`)
        : log('arbitrate', 'info', 'No near-ties: arbiter skipped'),
      upd('arbitrate', { ranking: rk, loops: { ...lp, arbitrate: lp.arbitrate + (rk[0].arbiter_note ? 1 : 0) } })),
    ...node('report', 2100, log('report', 'info', 'Generated explanations for top candidates'), upd('report', { report })),
    ...node('hr_review', 5, upd('hr_review', { ranking: rk })),
    { type: 'interrupt', kind: 'review', payload: { ranking: rk, weights: w } },
  ]
}

const start: Step[] = [
  { type: 'run_started', run_id: 'mock-run' },
  ...node('ingest', 640, log('ingest', 'info', 'Parsed JD; loaded bank profile weights'),
    upd('ingest', { requirements: requirements(false), weights: WEIGHTS, policy_context: policy })),
  ...node('clarify', 880,
    log('clarify', 'warn', '2 ambiguities found in the JD (depth level, mobile scope)'),
    upd('clarify', { loops: loops(0, 0, 0, 0, 0) })),
  { type: 'interrupt', kind: 'clarification', payload: { questions: requirements(false).ambiguities } },
]

const afterClarify: Step[] = [
  ...node('clarify', 20, log('clarify', 'loop', 'L1 HR answered 2 questions: re-ingesting JD with clarifications'), upd('clarify', { loops: loops(1, 0, 0, 0, 0) })),
  ...node('ingest', 510, log('ingest', 'info', 'Merged HR answers: depth level 3, mobile out of scope'), upd('ingest', { requirements: requirements(true) })),
  ...node('clarify', 700, log('clarify', 'info', 'No ambiguity left')),
  ...node('search', 430, log('search', 'info', 'BM25 + dense retrieval'), upd('search', { query_terms: terms, candidates: cand(2, false) }),
    log('search', 'warn', 'Coverage too low: 2 candidates match all must-haves (< 4)')),
  ...node('expand_query', 380, log('expand_query', 'loop', 'L2 expanding query with ESCO synonyms: WebDriver, RestAssured, Jenkins'),
    upd('expand_query', { query_terms: termsExpanded, loops: loops(1, 1, 0, 0, 0) })),
  ...node('search', 460, log('search', 'info', 'Re-ran retrieval with expanded terms: 4 candidates'), upd('search', { candidates: cand(4, false) })),
  ...node('rerank', 1200, log('rerank', 'info', 'Cross-encoder rerank of 4 candidates'), upd('rerank', { candidates: cand(4, true) })),
  ...node('assess', 5200, log('assess', 'info', 'Assessed 4 CVs on 8 criteria'), upd('assess', { assessments: assessments(1, false) })),
  ...node('verify_evidence', 160,
    log('verify_evidence', 'warn', 'CV-033: quote "Led a team of 5 automation engineers" not found in CV'),
    log('verify_evidence', 'loop', 'L4 ungrounded evidence: sending CV-033 back to assess'),
    upd('verify_evidence', { assessments: assessments(1, true), loops: loops(1, 1, 1, 0, 0) })),
  ...node('assess', 1700, log('assess', 'info', 'Re-assessed CV-033 with the grounding feedback (attempt 2)'), upd('assess', { assessments: { 'CV-033': assessments(2, false)['CV-033'] } })),
  ...node('verify_evidence', 150, log('verify_evidence', 'info', 'All evidence quotes grounded'), upd('verify_evidence', { assessments: assessments(2, true) })),
  ...scoring(WEIGHTS, loops(1, 1, 1, 0, 0)),
]

const afterAdjust = (w: Record<Criterion, number>): Step[] => [
  ...node('hr_review', 10, log('hr_review', 'loop', 'L5 HR adjusted weights: re-scoring with new weights'), upd('hr_review', { weights: w, loops: loops(1, 1, 1, 1, 1) })),
  ...scoring(w, loops(1, 1, 1, 1, 1)),
]

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
async function* play(steps: Step[]): AsyncGenerator<StreamEvent> {
  for (const s of steps) {
    await sleep(s.type === 'log' ? 450 : 650)
    yield s
  }
}

let weights = WEIGHTS
export const mockApi: Api = {
  config: async () => config,
  start: () => { weights = WEIGHTS; return play(start) },
  resume: (_id, body: ResumeBody) => {
    if ('answers' in body) return play(afterClarify)
    if (body.action === 'adjust') { weights = body.weights as Record<Criterion, number>; return play(afterAdjust(weights)) }
    return play([{ type: 'done', report, usage: { calls: 14, input_tokens: 41200, output_tokens: 9800, total_tokens: 51000, by_node: { ingest: { calls: 2, input_tokens: 1800, output_tokens: 400 }, assess: { calls: 9, input_tokens: 33000, output_tokens: 8200 }, arbitrate: { calls: 1, input_tokens: 1400, output_tokens: 100 }, report: { calls: 2, input_tokens: 5000, output_tokens: 1100 } } } }])
  },
  getRun: async (id) => ({
    run_id: id,
    values: {
      requirements: requirements(true), weights, policy_context: policy, query_terms: termsExpanded, candidates: cand(4, true),
      assessments: assessments(2, true), ranking: ranking(weights), report, loops: loops(1, 1, 1, 1, 0),
    },
    next: ['hr_review'],
    interrupt: { kind: 'review', payload: { ranking: ranking(weights), weights } },
  }),
  getCv: async (id) => {
    await sleep(200)
    const cv = CVS[id]
    if (!cv) throw new Error(`404 CV ${id} not found`)
    return { id, ...cv }
  },
}
