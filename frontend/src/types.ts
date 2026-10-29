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
  | { action: 'adjust'; weights: Record<string, number> }

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

export interface Evidence { quote: string; grounded: boolean | null }
export interface CriterionScore { score: number; rationale: string; evidence: Evidence[] }
export interface Assessment {
  criteria: Record<Criterion, CriterionScore>
  knocked_out: boolean
  knockout_reason: string | null
  attempts: number
}
export interface RankRow {
  rank: number
  cv_id: string
  headline: string
  total: number
  knocked_out: boolean
  arbiter_note: string | null
}
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

export type Interrupt =
  | { kind: 'clarification'; payload: { questions: string[] } }
  | { kind: 'review'; payload: { ranking: RankRow[]; weights: Record<string, number> } }

export type LogLevel = 'info' | 'loop' | 'warn'
export type StreamEvent =
  | { type: 'run_started'; run_id: string }
  | { type: 'node_start'; node: NodeName }
  | { type: 'log'; node: NodeName; level: LogLevel; message: string }
  | { type: 'update'; node: NodeName; update: Partial<Values> }
  | { type: 'node_end'; node: NodeName; duration_ms: number }
  | ({ type: 'interrupt' } & Interrupt)
  | { type: 'done'; report: Report }
  | { type: 'error'; message: string }

export interface RunSnapshot {
  run_id: string
  values: Partial<Values>
  next: string[]
  interrupt: Interrupt | null
}

export interface CV { id: string; headline: string; text: string }

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
}
export const emptyRun: RunState = {
  runId: null, values: {}, nodes: {}, logs: [], interrupt: null,
  expanded: [], flash: {}, running: false, done: false,
}
