// Types mirror docs/API.md (the contract).

export const CRITERIA = [
  'tools', 'automation_depth', 'programming_language', 'test_types',
  'cicd_infra', 'experience_level', 'istqb', 'domain_language',
] as const
export type Criterion = (typeof CRITERIA)[number]

export const NODES = [
  'ingest', 'clarify', 'search', 'expand_query', 'rerank', 'assess',
  'verify_evidence', 'score', 'arbitrate', 'report', 'hr_review',
] as const
export type NodeName = (typeof NODES)[number]

export type LoopKey = 'clarify' | 'expand' | 'verify' | 'arbitrate' | 'review'
export type Loops = Record<LoopKey, number>

export interface Config {
  providers: { name: string; configured: boolean }[]
  defaults: { reasoning: string; fast: string }
  company_profiles: { key: string; label: string }[]
  criteria: { key: string; label: string }[]
  sample_jds: { id: string; title: string; company_profile: string; text: string }[]
  langfuse_url: string | null
}

export interface RunRequest {
  jd_text: string
  company_profile: string
  models?: { reasoning?: string; fast?: string }
}

export type ResumeBody =
  | { answers: Record<string, string> }
  | { action: 'approve' }
  | { action: 'adjust'; weights?: Record<string, number>; instruction?: string }
  | { action: 'decide'; decisions: Decision[] } // HR decisions (with reasons) calibrate the weights
  | { accept: Record<string, string[]> } // substitutes: must-have → accepted related tools ([] = reject)

export interface Requirements {
  title: string
  must_have: string[]
  nice_to_have: string[]
  min_years: number | null
  depth_level: number | null
  languages: string[]
  test_types: string[]
  ambiguities: string[]
}

// grounded: quote exists in the CV; supported: the section it came from actually demonstrates the claim
// (grounded but unsupported = "listed, not demonstrated", e.g. depth evidence quoted from a skills list).
export interface Evidence { quote: string; grounded: boolean | null; supported: boolean | null; section: string | null }
export interface CriterionScore { score: number; rationale: string; evidence: Evidence[] }
export interface Decision { cv_id: string; action: 'approve' | 'reject' | 'override'; reason: string; new_rank?: number }
export interface Assessment {
  criteria: Record<Criterion, CriterionScore> // v3: all 8 by rule from the parsed CV (they set the score)
  llm_review?: Partial<Record<Criterion, CriterionScore>> // the LLM's judgement, quotes verified (explains, does not score)
  knocked_out: boolean
  knockout_reason: string | null
  attempts: number
  facts?: Facts
}
export interface Facts { depth_level: number | null; years: number | null; depth_cue: number | null; years_automation: number | null }
export interface RankRow {
  rank: number
  cv_id: string
  headline: string
  total: number
  knocked_out: boolean
  arbiter_note: string | null
  status: RankStatus
  tier: Tier | null // only set for status "ranked"
  reason: string | null
  card?: Card | null // deterministic explanation: ranked rows get all parts, knocked-out rows only what_if
  flags?: string[] // data quality: short_text, garbled_text, few_skills, no_dated_roles, duplicate_of:<id>, …
}
export interface Card {
  decomposition?: { criterion: Criterion; weight: number; score: number; contribution: number }[]
  versus_next?: { vs: string; advantages: { criterion: Criterion; delta: number }[]; disadvantages: { criterion: Criterion; delta: number }[] } | null
  what_if?: { if_used: string[]; must_have: boolean; passes_knockout: boolean; total: number; delta: number }[]
}
export type RankStatus = 'ranked' | 'needs_review' | 'not_shortlisted' | 'knocked_out'
export type Tier = 'strong' | 'possible' | 'weak'
export interface Report {
  summary: string
  candidates: { cv_id: string; explanation: string; strengths: string[]; gaps: string[] }[]
}

export interface Values {
  requirements: Requirements
  weights: Record<Criterion, number>
  policy_context: { source: string; text: string }[]
  query_terms: string[]
  candidates: {
    cv_id: string; headline: string; retrieval_score: number
    rerank_score: number | null; matched_must_have: string[]
  }[]
  assessments: Record<string, Assessment>
  ranking: RankRow[]
  report: Report
  loops: Loops
}

export interface SubstituteProposal { must_have: string; covered: number; related: string[] }

export type Interrupt =
  | { kind: 'clarification'; payload: { questions: string[] } }
  | { kind: 'substitutes'; payload: { proposals: SubstituteProposal[] } }
  | { kind: 'review'; payload: { ranking: RankRow[]; weights: Record<string, number>; adjustments_left?: number } }

// Graph node that owns each interrupt (it shows "waiting for HR" in the graph).
export const INTERRUPT_NODE: Record<Interrupt['kind'], NodeName> = {
  clarification: 'clarify', substitutes: 'expand_query', review: 'hr_review',
}

export type LogLevel = 'info' | 'loop' | 'warn'
export type StreamEvent =
  | { type: 'run_started'; run_id: string; trace_url?: string | null }
  | { type: 'node_start'; node: NodeName }
  | { type: 'log'; node: NodeName; level: LogLevel; message: string }
  | { type: 'update'; node: NodeName; update: Partial<Values> }
  | { type: 'node_end'; node: NodeName; duration_ms: number }
  | ({ type: 'interrupt' } & Interrupt)
  | { type: 'done'; report: Report; usage?: Usage }
  | { type: 'error'; message: string }

export interface RunSnapshot {
  run_id: string
  values: Partial<Values>
  next: string[]
  interrupt: Interrupt | null
  usage?: Usage
  logs?: LogLine[]
}

// LLM calls + tokens for a run (all resumes), per graph node — see backend observability.UsageCounter
export interface Usage {
  calls: number; input_tokens: number; output_tokens: number; total_tokens: number
  by_node: Record<string, { calls: number; input_tokens: number; output_tokens: number }>
}

export interface CVListItem { id: string; headline: string; lang: Lang; uploaded: boolean }
export interface CV { id: string; headline: string; text: string; profile: CVProfileView }

// Output of the CV NLP pipeline (language → redaction → sections → chunks → skills/facts).
export type Lang = 'en' | 'vi' | 'mixed'
export type SkillStrength = 'used' | 'listed' | 'hedged'
export interface CVProfileView {
  id: string
  headline: string
  lang: Lang
  redactions: Record<string, number>
  sections: { kind: string; heading: string }[]
  chunks: { id: string; section: string; header: string; text: string }[]
  skills: { skill: string; strength: SkillStrength; section: string; line: string }[]
  years_total: number | null
  years_automation: number | null
  depth_cue: number
  depth_cue_line: string | null
}

// Client-side live state, built by folding stream events (see applyEvent in App.tsx).
export type NodeStatus = 'idle' | 'running' | 'done' | 'waiting'
export interface LogLine { node: NodeName; level: LogLevel; message: string }
export interface RunState {
  runId: string | null
  values: Partial<Values>
  nodes: Partial<Record<NodeName, { status: NodeStatus; ms?: number }>>
  logs: LogLine[]
  interrupt: Interrupt | null
  expanded: string[] // query terms added after the first query_terms update (L2)
  flash: Partial<Record<LoopKey, number>> // bumped each time a loop fires
  running: boolean
  done: boolean
  usage: Usage | null
  traceUrl: string | null
}
export const emptyRun: RunState = {
  runId: null, values: {}, nodes: {}, logs: [], interrupt: null,
  expanded: [], flash: {}, running: false, done: false, usage: null, traceUrl: null,
}
