import type { CSSProperties } from 'react'
import type { LoopKey, NodeName, RunState } from './types'

const W = 130, H = 44
// [x, y, subtitle, tone]; layout follows the proposed-methods slide, flow wraps right-to-left on row 2.
const POS: Record<NodeName, [number, number, string, string]> = {
  ingest: [10, 70, 'parse JD + profile', 'slate'],
  clarify: [175, 70, 'ambiguous criteria?', 'amber'],
  search: [340, 70, 'chunks · per requirement', 'sky'],
  expand_query: [340, 160, 'related tools, ask HR', 'sky'],
  investigate: [505, 160, 'tool-using agent', 'amber'],
  rerank: [505, 70, 'CE on passages', 'purple'],
  assess: [670, 70, '4 LLM + depth + 3 rules', 'slate'],
  verify_evidence: [835, 70, 'grounded + supported?', 'rose'],
  score: [835, 250, 'Σ w·S, tier, reasons', 'purple'],
  arbitrate: [670, 250, 'near-tie judge', 'purple'],
  report: [505, 250, 'explanations', 'emerald'],
  hr_review: [340, 250, 'human approval', 'emerald'],
}

const EDGES = [
  'M140 92 L175 92', 'M305 92 L340 92', 'M470 92 L505 92', 'M635 92 L670 92', 'M800 92 L835 92',
  'M900 114 L900 250', 'M835 272 L800 272', 'M670 272 L635 272', 'M505 272 L470 272',
  'M340 272 L254 272', 'M375 114 L375 160', 'M470 108 L505 168', 'M570 160 L570 114',
]

// Loop edges: path, label anchor, tone.
const LOOPS: { key: LoopKey; id: string; name: string; d: string; x: number; y: number; tone: string }[] = [
  { key: 'clarify', id: 'L1', name: 'Ask HR', d: 'M240 70 L240 30 L75 30 L75 70', x: 157, y: 30, tone: 'amber' },
  { key: 'expand', id: 'L2', name: 'Expand + ask HR', d: 'M440 160 L440 114', x: 505, y: 138, tone: 'sky' },
  { key: 'investigate', id: 'L6', name: 'Investigate knock-outs', d: 'M635 172 C675 172 675 198 635 198', x: 700, y: 186, tone: 'amber' },
  { key: 'arbitrate', id: 'L3', name: 'Arbiter', d: 'M705 250 C705 212 765 212 765 250', x: 735, y: 200, tone: 'purple' },
  { key: 'verify', id: 'L4', name: 'Block hallucination', d: 'M900 70 L900 30 L735 30 L735 70', x: 817, y: 30, tone: 'rose' },
  { key: 'review', id: 'L5', name: 'HR re-score', d: 'M405 294 L405 345 L900 345 L900 294', x: 652, y: 345, tone: 'emerald' },
]
const TONES = ['slate', 'amber', 'sky', 'purple', 'rose', 'emerald']

const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`)

export default function GraphView({ run }: { run: RunState }) {
  const loops = run.values.loops
  return (
    <section className="card">
      <h2>Agent graph</h2>
      <svg viewBox="0 0 1000 380" className="graph" role="img" aria-label="Agent state graph">
        <defs>
          {TONES.map((t) => (
            <marker key={t} id={`ar-${t}`} viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0 1 L8 5 L0 9z" style={{ fill: `var(--${t})` }} />
            </marker>
          ))}
        </defs>

        {EDGES.map((d) => <path key={d} d={d} className="edge" markerEnd="url(#ar-slate)" />)}

        {LOOPS.map((l) => {
          const n = run.flash[l.key]
          const count = loops?.[l.key] ?? 0
          const label = `${l.id} ${l.name} ×${count}`
          const w = label.length * 6.4 + 16
          return (
            <g key={`${l.key}-${n ?? 0}`} className={`loop${n ? ' flash' : ''}${count ? ' used' : ''}`} style={{ '--tone': `var(--${l.tone})` } as CSSProperties}>
              <path d={l.d} className="loop-path" markerEnd={`url(#ar-${l.tone})`} />
              <rect x={l.x - w / 2} y={l.y - 10} width={w} height={20} rx={5} className="loop-badge" />
              <text x={l.x} y={l.y + 4} textAnchor="middle" className="loop-label">{label}</text>
            </g>
          )
        })}

        <g className="node end">
          <rect x={170} y={250} width={84} height={H} rx={22} />
          <text x={212} y={277} textAnchor="middle" className="n-title">END</text>
        </g>

        {(Object.keys(POS) as NodeName[]).map((name) => {
          const [x, y, sub, tone] = POS[name]
          const st = run.nodes[name]
          const status = st?.status ?? 'idle'
          return (
            <g key={name} className={`node ${status}`} style={{ '--tone': `var(--${tone})` } as CSSProperties}>
              <rect x={x} y={y} width={W} height={H} rx={7} />
              <text x={x + 10} y={y + 18} className="n-title">{name}</text>
              <text x={x + 10} y={y + 34} className="n-sub">{status === 'waiting' ? 'waiting for HR' : sub}</text>
              {st?.ms != null && <text x={x + W - 8} y={y + 18} textAnchor="end" className="n-ms">{fmtMs(st.ms)}</text>}
              <circle cx={x + W - 10} cy={y + 33} r={4} className="dot" />
            </g>
          )
        })}
      </svg>
      <div className="legend">
        <span><i className="dot-idle" /> idle</span>
        <span><i className="dot-running" /> running</span>
        <span><i className="dot-done" /> done</span>
        <span><i className="dot-waiting" /> waiting for HR</span>
        <span className="muted">dashed = reflection loop; flashes when it fires</span>
      </div>
    </section>
  )
}
