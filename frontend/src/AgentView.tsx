import type { CSSProperties } from 'react'
import type { NodeName, RunState } from './types'

const W = 120, H = 44
// The P7 orchestrator (backend/app/agent/orchestrator.py): the model decides, the guard checks, code acts.
const POS: Partial<Record<NodeName, [number, number, string, string]>> = {
  start: [10, 90, 'HR goal + JD', 'slate'],
  agent: [170, 90, 'LLM picks a tool', 'amber'],
  guard: [340, 90, 'harness checks', 'rose'],
  act: [520, 30, 'run tools (code)', 'sky'],
  ask_hr: [520, 150, 'question to HR', 'emerald'],
  finalize: [700, 150, 'budget → code answers', 'rose'],
  hr_turn: [700, 30, 'HR replies / approves', 'emerald'],
}
const EDGES = [
  'M130 112 L170 112', 'M290 112 L340 112', 'M460 104 L520 56', 'M460 120 L520 168', 'M640 172 L700 172',
  'M760 150 L760 74', 'M640 52 L700 52',
]
const LOOPS = [
  { d: 'M580 30 L580 8 L230 8 L230 90', label: 'result → agent', x: 400, y: 8, tone: 'sky' },
  { d: 'M580 194 L580 224 L230 224 L230 134', label: 'HR answer → agent', x: 400, y: 224, tone: 'emerald' },
  { d: 'M400 134 C400 176 290 176 260 134', label: 'refused → agent', x: 330, y: 172, tone: 'rose' },
  { d: 'M820 52 L860 52 L860 250 L200 250 L200 134', label: 'HR message → agent', x: 600, y: 250, tone: 'emerald' },
]
const TONES = ['slate', 'amber', 'sky', 'rose', 'emerald']

export default function AgentView({ run }: { run: RunState }) {
  const l = run.values.loops ?? ({} as NonNullable<RunState['values']['loops']>)
  const stats: [string, number | undefined][] = [
    ['decisions', l.agent], ['tools run', run.tools.length], ['refused by guard', l.guard_rejections],
    ['cache hits', l.cache_hits], ['budget stops', l.budget_stops], ['orchestrator tokens', l.agent_tokens],
  ]
  const refusals = run.logs.filter((x) => x.node === 'guard' && x.level === 'warn')
  return (
    <section className="card">
      <h2>Agent (P7) <span className="muted small">the model chooses each step; the harness checks it</span></h2>
      <svg viewBox="0 0 880 262" className="graph" role="img" aria-label="Orchestrator graph">
        <defs>
          {TONES.map((t) => (
            <marker key={t} id={`ag-${t}`} viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0 1 L8 5 L0 9z" style={{ fill: `var(--${t})` }} />
            </marker>
          ))}
        </defs>
        {EDGES.map((d) => <path key={d} d={d} className="edge" markerEnd="url(#ag-slate)" />)}
        {LOOPS.map((x) => (
          <g key={x.label} className="loop used" style={{ '--tone': `var(--${x.tone})` } as CSSProperties}>
            <path d={x.d} className="loop-path" markerEnd={`url(#ag-${x.tone})`} />
            <rect x={x.x - x.label.length * 3.3 - 8} y={x.y - 10} width={x.label.length * 6.6 + 16} height={20} rx={5} className="loop-badge" />
            <text x={x.x} y={x.y + 4} textAnchor="middle" className="loop-label">{x.label}</text>
          </g>
        ))}
        {(Object.keys(POS) as NodeName[]).map((name) => {
          const [x, y, sub, tone] = POS[name]!
          const status = run.nodes[name]?.status ?? 'idle'
          return (
            <g key={name} className={`node ${status}`} style={{ '--tone': `var(--${tone})` } as CSSProperties}>
              <rect x={x} y={y} width={W} height={H} rx={7} />
              <text x={x + 10} y={y + 18} className="n-title">{name}</text>
              <text x={x + 10} y={y + 34} className="n-sub">{status === 'waiting' ? 'waiting for HR' : sub}</text>
              <circle cx={x + W - 10} cy={y + 33} r={4} className="dot" />
            </g>
          )
        })}
      </svg>
      <div className="chips">
        {stats.map(([k, v]) => <span key={k} className="badge gray">{k}: <b>{(v ?? 0).toLocaleString()}</b></span>)}
      </div>
      <h3>Tool calls</h3>
      {run.tools.length === 0 && <p className="muted">None yet.</p>}
      <ol className="tool-calls">
        {run.tools.map((t, i) => (
          <li key={i}>
            <span className="mono"><b>{t.name}</b>({Object.keys(t.args).length ? JSON.stringify(t.args) : ''})</span>
            <div className="muted small mono">{t.result.slice(0, 180)}</div>
          </li>
        ))}
      </ol>
      {refusals.length > 0 && (
        <>
          <h3>Refused by the harness</h3>
          <ul>{refusals.map((r, i) => <li key={i} className="log warn">{r.message}</li>)}</ul>
        </>
      )}
      {run.chat.length > 0 && (
        <>
          <h3>Conversation</h3>
          <div className="chat">
            {run.chat.map((c, i) => <p key={i} className={`bubble ${c.who}`}><b>{c.who === 'hr' ? 'HR' : 'Agent'}:</b> {c.text}</p>)}
          </div>
        </>
      )}
    </section>
  )
}
