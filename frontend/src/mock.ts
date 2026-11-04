// Canned run for `?mock=1`: exercises clarification (L1), query expansion (L2), hallucination
// check (L4), arbiter (L3), review + HR re-score (L5) and done, with delays between events.
import type { Api } from './api'
import {
  CRITERIA, type Assessment, type Config, type Criterion, type CVProfileView, type Facts, type NodeName, type RankRow,
  type ResumeBody, type Report, type SkillStrength, type StreamEvent, type Values,
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
Contact: [EMAIL] · [PHONE] · [BIRTH_DATE]

SUMMARY
Designed and maintained a Selenium WebDriver + Java framework with Page Object pattern for a banking web app.

EXPERIENCE
- Built REST Assured API suites (320 tests) running in Jenkins pipelines on every merge.
- Introduced Docker-based Selenium Grid, cutting regression from 6h to 55min.

SKILLS
Selenium WebDriver, Java, TestNG, REST Assured, Jenkins, Docker

CERTIFICATIONS
ISTQB Foundation Level (2021)

LANGUAGES
English: IELTS 6.5` },
  'CV-027': { headline: 'SDET, 5 yrs, Selenium/Python, API + Web', text:
`SDET · 5 years
Contact: [EMAIL] · [PHONE]

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

SKILLS
Cypress, JavaScript, Jenkins, Git

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
  // Pool CVs that are not assessed in this run (shortlisting / knock-out / review demos).
  'CV-005': { headline: 'Automation Engineer, 4 yrs, Selenium/Java', text:
`Automation Engineer · 4 years
Contact: [EMAIL]

SUMMARY
Selenium and Java automation for an insurance portal.

EXPERIENCE
- Maintained 150 Selenium tests in Java; ran them nightly from Jenkins.

SKILLS
Selenium, Java, Maven` },
  'CV-009': { headline: 'QA Automation, 3 yrs, Selenium/Python', text:
`QA Automation · 3 years

SUMMARY
Selenium and Python scripts for regression of an HR portal.

EXPERIENCE
- Wrote 80 Selenium scripts in Python for the regression suite.

SKILLS
Selenium, Python, Git` },
  'CV-048': { headline: 'QA Engineer, 4 yrs, mobile testing', text:
`QA Engineer · 4 years
Contact: [EMAIL] · [PHONE]

SUMMARY
Mobile QA on Android and iOS releases.

EXPERIENCE
- Executed manual test plans on 12 mobile releases.

SKILLS
Postman, familiar with Appium (not used in production)` },
  'CV-052': { headline: 'Kiểm thử tự động, 5 năm, Selenium + Python', text:
`Kiểm thử tự động · 5 năm
Contact: [EMAIL] · [PHONE]

SUMMARY
Kỹ sư kiểm thử tự động with 5 years on banking apps. Selenium, Python.

EXPERIENCE
- Xây dựng framework Selenium + Python cho ứng dụng ngân hàng.

SKILLS
Selenium, Python, pytest, Jenkins` },
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

// Section a quote was found in = the last ALL-CAPS heading line before it ("header" before the first heading).
function sectionOf(text: string, quote: string): string | null {
  const i = text.toLowerCase().indexOf(quote.toLowerCase())
  if (i < 0) return null
  const heads = [...text.slice(0, i).matchAll(/^[A-Z][A-Z &/]+$/gm)]
  return heads.length ? heads[heads.length - 1][0].toLowerCase() : 'header'
}

const FACTS: Record<string, Facts> = {
  'CV-014': { depth_level: 4, years: 6, depth_cue: 4, years_automation: 6 },
  'CV-027': { depth_level: 3, years: 5, depth_cue: 3, years_automation: 5 },
  'CV-033': { depth_level: 2, years: 3, depth_cue: 2, years_automation: 2 },
  'CV-041': { depth_level: 1, years: 4, depth_cue: 1, years_automation: null },
}

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
      if (id === 'CV-033' && c === 'cicd_infra') { quote = 'Jenkins'; rationale = 'Jenkins is only named in a skills list; no pipeline is described.' }
      const section = quote ? sectionOf(CVS[id].text, quote) : null
      const grounded = verified ? text.includes((quote ?? '').toLowerCase()) : null
      criteria[c] = { score, rationale, evidence: quote ? [{ quote, grounded, section, supported: verified && grounded ? section !== 'skills' : null }] : [] }
    }
    out[id] = { criteria, knocked_out: id === 'CV-041', knockout_reason: id === 'CV-041' ? 'Missing must-have: Selenium (no automation tool in production use)' : null, attempts: pass, facts: FACTS[id] }
  }
  return out
}

// Every CV in the pool appears: ranked, then not_shortlisted, needs_review, knocked_out.
function ranking(w: Record<Criterion, number>): RankRow[] {
  const a = assessments(2, true)
  const scored = Object.keys(a).filter((id) => !a[id].knocked_out).map((id) => ({
    cv_id: id, headline: CVS[id].headline,
    total: CRITERIA.reduce((s, c) => s + w[c] * a[id].criteria[c].score / 100, 0),
  })).sort((x, y) => y.total - x.total)
  const tier = (t: number) => (t >= 0.75 ? 'strong' : t >= 0.6 ? 'possible' : 'weak') as 'strong' | 'possible' | 'weak'
  const ranked: RankRow[] = scored.map((r, i) => ({
    ...r, rank: i + 1, knocked_out: false, status: 'ranked', tier: tier(r.total), reason: null,
    arbiter_note: i < 2 && Math.abs(scored[0].total - scored[1].total) < 0.05
      ? `Near-tie (Δ ${Math.abs(scored[0].total - scored[1].total).toFixed(3)}); arbiter compared ${scored[0].cv_id} vs ${scored[1].cv_id} on evidence depth.` : null,
  }))
  const other = (cv_id: string, status: RankRow['status'], reason: string): RankRow => ({
    rank: ranked.length + 1, cv_id, headline: CVS[cv_id].headline, total: 0, knocked_out: status === 'knocked_out',
    arbiter_note: null, status, tier: null, reason,
  })
  return [
    ...ranked,
    other('CV-005', 'not_shortlisted', `Rank ${ranked.length + 1} after rerank; best evidence matched: selenium, java; missing: rest assured`),
    other('CV-009', 'not_shortlisted', `Rank ${ranked.length + 2} after rerank; best evidence matched: selenium; missing: java, jenkins`),
    other('CV-052', 'needs_review', 'Mixed Vietnamese/English CV: the assessor could not ground its evidence after 2 attempts'),
    other('CV-041', 'knocked_out', 'Missing must-have: selenium (no automation tool in production use)'),
    other('CV-048', 'knocked_out', `Only hedged mention of appium: 'familiar with Appium (not used in production)'`),
  ]
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
const loops = (clarify: number, expand: number, verify: number, arbitrate: number, review: number, investigate = 0) => ({ clarify, expand, verify, arbitrate, review, investigate })

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
  ...node('investigate', 15, log('investigate', 'info', 'No knock-out looks like a parsing failure (no garbled text, no misspelt must-have) → investigator not needed (0 LLM calls)')),
  // L2 now asks HR before widening the query: the node starts, then waits on the substitutes interrupt.
  { type: 'node_start', node: 'expand_query' },
  log('expand_query', 'warn', 'Only 2 CVs show selenium: asking HR which related tools may substitute'),
  { type: 'interrupt', kind: 'substitutes', payload: { proposals: [{ must_have: 'selenium', covered: 2, related: ['cypress', 'playwright', 'webdriverio'] }] } },
]

const afterSubstitutes = (accept: Record<string, string[]>): Step[] => {
  const tools = Object.values(accept).flat()
  const q = [...termsExpanded, ...tools]
  return [
    ...node('expand_query', 380,
      log('expand_query', 'loop', tools.length
        ? `L2 HR accepted substitutes (${tools.join(', ')}): expanding query with them and ESCO synonyms`
        : 'L2 HR rejected all substitutes: expanding with ESCO synonyms only'),
      upd('expand_query', { query_terms: q, loops: loops(1, 1, 0, 0, 0) })),
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
}

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

// --- mock NLP pipeline: a tiny stand-in for the backend's parser, so any CV (also uploaded ones) has a preview ---
const VOCAB = ['selenium', 'playwright', 'cypress', 'appium', 'webdriver', 'rest assured', 'postman', 'testng', 'pytest', 'java', 'python',
  'javascript', 'jenkins', 'github actions', 'docker', 'git', 'testrail', 'maven']
const AUTOMATION = ['selenium', 'playwright', 'cypress', 'appium']
const HEDGE = /familiar|learning|course|basic|exposure|not used/i

function profileOf(id: string, headline: string, text: string): CVProfileView {
  const lines = text.split('\n')
  const sections: CVProfileView['sections'] = []
  const chunks: CVProfileView['chunks'] = []
  const skills: CVProfileView['skills'] = []
  let kind = 'header', heading = ''
  const push = (t: string) => {
    if (t.trim()) chunks.push({ id: `${id}#${chunks.length}`, section: kind, header: `${id} | ${headline} | ${heading || 'HEADER'}`, text: t.trim() })
  }
  let para: string[] = []
  const flush = () => { push(para.join('\n')); para = [] }
  for (const l of lines) {
    if (/^[A-ZĐ][A-ZĐ &/]+$/.test(l.trim())) { flush(); heading = l.trim(); kind = heading.toLowerCase(); sections.push({ kind, heading }); continue }
    if (l.startsWith('- ')) { flush(); push(l.slice(2)); continue }
    if (!l.trim()) { flush(); continue }
    para.push(l)
  }
  flush()
  if (chunks.some((c) => c.section === 'header')) sections.unshift({ kind: 'header', heading: '(top of CV)' })

  const secOf = (ln: string) => chunks.find((c) => c.text.includes(ln.replace(/^- /, '').trim()))?.section ?? 'header'
  for (const sk of VOCAB) {
    const hits = lines.filter((l) => l.toLowerCase().includes(sk) && !l.startsWith('Contact:'))
    if (!hits.length) continue
    const body = hits.filter((l) => secOf(l) !== 'skills')
    const hedged = hits.find((l) => HEDGE.test(l) && new RegExp(`(${sk}).*(${HEDGE.source})|(${HEDGE.source}).*(${sk})`, 'i').test(l))
    const best = hedged ?? body[0] ?? hits[0]
    const strength: SkillStrength = hedged ? 'hedged' : body.length ? 'used' : 'listed'
    skills.push({ skill: sk, strength, section: secOf(best), line: best.replace(/^- /, '').trim() })
  }
  const years = /(\d+)\s*(?:years|yrs|năm)/i.exec(text)
  const total = years ? +years[1] : null
  const usesAuto = skills.some((k) => AUTOMATION.includes(k.skill) && k.strength === 'used')
  const cueLine = lines.find((l) => /framework|architecture|grid/i.test(l) && !HEDGE.test(l)) ?? null
  const cue = cueLine ? 3 : usesAuto ? 2 : skills.length ? 1 : 0
  const tag = (t: string) => (text.match(new RegExp(`\\[${t}\\]`, 'g')) ?? []).length
  const redactions = Object.fromEntries(Object.entries({ email: tag('EMAIL'), phone: tag('PHONE'), birth_date: tag('BIRTH_DATE') }).filter(([, n]) => n))
  const vi = /[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]/i.test(text)
  const en = /\b(the|and|with|for|built|years)\b/i.test(text)
  return {
    id, headline, lang: vi && en ? 'mixed' : vi ? 'vi' : 'en', redactions, sections, chunks, skills,
    years_total: total, years_automation: usesAuto && total ? Math.max(1, total - 1) : null, depth_cue: cue,
    depth_cue_line: (cueLine ?? (usesAuto ? skills.find((k) => AUTOMATION.includes(k.skill))!.line : null))?.replace(/^- /, '') ?? null,
  }
}

const uploaded: Record<string, { headline: string; text: string }> = {}
const lookup = (id: string) => CVS[id] ?? uploaded[id]
const ALLOWED = ['.pdf', '.docx', '.md', '.txt']
const SAMPLE_UPLOAD = `Test Automation Engineer · 4 years
Contact: [EMAIL] · [PHONE]

SUMMARY
Built a Playwright + TypeScript framework for a logistics web app.

EXPERIENCE
- Wrote 120 Playwright tests and ran them in GitHub Actions on every pull request.
- Currently learning Appium for mobile.

SKILLS
Playwright, TypeScript, GitHub Actions, Docker`

let weights = WEIGHTS
export const mockApi: Api = {
  config: async () => config,
  start: () => { weights = WEIGHTS; return play(start) },
  resume: (_id, body: ResumeBody) => {
    if ('answers' in body) return play(afterClarify)
    if ('accept' in body) return play(afterSubstitutes(body.accept))
    if ('action' in body && body.action === 'adjust') { weights = body.weights as Record<Criterion, number>; return play(afterAdjust(weights)) }
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
    const cv = lookup(id)
    if (!cv) throw new Error(`404 CV ${id} not found`)
    return { id, ...cv, profile: profileOf(id, cv.headline, cv.text) }
  },
  listCvs: async () => [
    ...Object.entries(CVS).map(([id, c]) => ({ id, headline: c.headline, lang: profileOf(id, c.headline, c.text).lang, uploaded: false })),
    ...Object.entries(uploaded).map(([id, c]) => ({ id, headline: c.headline, lang: profileOf(id, c.headline, c.text).lang, uploaded: true })),
  ],
  uploadCv: async (file) => {
    await sleep(600)
    const ext = file.name.slice(file.name.lastIndexOf('.')).toLowerCase()
    if (!ALLOWED.includes(ext)) throw new Error(`Unsupported format "${ext}". Use .pdf, .docx, .md or .txt`)
    // The mock cannot parse binaries, so .pdf/.docx get a sample CV text instead.
    const text = ext === '.md' || ext === '.txt' ? await file.text() : SAMPLE_UPLOAD
    if (!text.trim()) throw new Error('No text could be extracted from the file')
    const id = `CV-U${String(Object.keys(uploaded).length + 1).padStart(3, '0')}`
    const headline = text.split('\n').find((l) => l.trim())!.trim().slice(0, 90)
    uploaded[id] = { headline, text }
    return profileOf(id, headline, text)
  },
  batchUploadCvs: async (files) => {
    await sleep(800)
    const items = []
    for (const f of files) {
      const id = `CV-U${String(Object.keys(uploaded).length + 1).padStart(3, '0')}`
      const headline = f.name.replace(/\.[^/.]+$/, '')
      uploaded[id] = { headline, text: SAMPLE_UPLOAD }
      items.push({ id, filename: f.name, headline, skills: ['selenium', 'java'], years_automation: 3.0, lang: 'vi' as const })
    }
    return { total: files.length, succeeded: files.length, failed: 0, items, errors: [] }
  },
  deleteCv: async (id) => {
    delete uploaded[id]
    return { status: 'deleted', id }
  },
  resetCvs: async () => {
    const count = Object.keys(uploaded).length
    for (const k of Object.keys(uploaded)) delete uploaded[k]
    return { status: 'reset', deleted_count: count }
  },
  getExportCsvUrl: (runId) => `#export-${runId}`,
  parseJd: async (file) => {
    await sleep(400)
    const raw_text = await file.text()
    return {
      title: file.name.replace(/\.[^/.]+$/, ''),
      must_have: ['playwright', 'typescript', 'api_testing'],
      nice_to_have: ['istqb', 'docker'],
      substitutes: { playwright: ['selenium'] },
      min_years: 3,
      depth_level: 3,
      languages: ['typescript', 'python'],
      test_types: ['web', 'api'],
      foreign_languages: ['english'],
      domains: ['banking'],
      filename: file.name,
      raw_text: raw_text || 'Senior Automation QA Engineer with Playwright and CI/CD experience.',
    }
  },
}
