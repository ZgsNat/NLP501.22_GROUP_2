// Types mirror docs/API.md (the contract).

export const CRITERIA = [
  'tools', 'automation_depth', 'programming_language', 'test_types',
  'cicd_infra', 'experience_level', 'istqb', 'domain_language',
] as const
export type Criterion = (typeof CRITERIA)[number]

export const NODES = [
  'ingest', 'clarify', 'search', 'investigate', 'expand_query', 'rerank', 'assess',
  'verify_evidence', 'score', 'arbitrate', 'report', 'hr_review',
] as const
// P7: the orchestrator graph (backend/app/agent/orchestrator.py); the model picks tools, the guard checks them.
export const AGENT_NODES = ['start', 'agent', 'guard', 'act', 'ask_hr', 'finalize', 'hr_turn'] as const
export type NodeName = (typeof NODES)[number] | (typeof AGENT_NODES)[number] | 'update_requirements' | 'set_weights'
export type Mode = 'agent' | 'pipeline'

export type LoopKey = 'clarify' | 'expand' | 'investigate' | 'verify' | 'arbitrate' | 'review'
// P7 adds agent, agent_tokens, guard_rejections, cache_hits, budget_stops, hr_turn, version
export type Loops = Record<LoopKey, number> & Partial<Record<string, number>>

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
  mode?: Mode // default agent (P7)
  goal?: string // agent mode: what HR wants, in their words
}

export type ResumeBody =
  | { answers: Record<string, string> }
  | { action: 'approve' }
  | { action: 'adjust'; weights?: Record<string, number>; instruction?: string }
  | { action: 'decide'; decisions: Decision[] } // HR decisions (with reasons) calibrate the weights
  | { accept: Record<string, string[]> } // substitutes: must-have → accepted related tools ([] = reject)
  | { message: string } // agent mode: HR's next message to the orchestrator

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
  goal?: string // agent mode
  shortlist?: string[]
  reply?: string
}

export interface SubstituteProposal { must_have: string; covered: number; related: string[] }

export type Interrupt =
  | { kind: 'clarification'; payload: { questions: string[] } }
  | { kind: 'substitutes'; payload: { proposals: SubstituteProposal[] } }
  | { kind: 'review'; payload: { ranking: RankRow[]; weights: Record<string, number>; adjustments_left?: number } }
  | { kind: 'chat'; payload: { message: string; shortlist: RankRow[]; turns_left: number } }

// Graph node that owns each interrupt (it shows "waiting for HR" in the graph).
export const INTERRUPT_NODE: Record<Interrupt['kind'], NodeName> = {
  clarification: 'clarify', substitutes: 'expand_query', review: 'hr_review', chat: 'hr_turn',
}

export type LogLevel = 'info' | 'loop' | 'warn'
export type StreamEvent =
  | { type: 'run_started'; run_id: string; mode?: Mode; trace_url?: string | null }
  | { type: 'tool'; name: string; args: Record<string, unknown>; result: string } // P7: a tool the guard let run
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

export interface CVListItem {
  id: string
  headline: string
  filename?: string
  uploaded_at?: string
  lang: Lang
  skills?: string[]
  years_automation?: number | null
  uploaded: boolean
}
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
  mode: Mode
  tools: { name: string; args: Record<string, unknown>; result: string }[] // P7 tool calls that ran, in order
  chat: { who: 'hr' | 'agent'; text: string }[] // P7 conversation
}
export const emptyRun: RunState = {
  runId: null, values: {}, nodes: {}, logs: [], interrupt: null,
  expanded: [], flash: {}, running: false, done: false, usage: null, traceUrl: null, mode: 'agent', tools: [], chat: [],
}
