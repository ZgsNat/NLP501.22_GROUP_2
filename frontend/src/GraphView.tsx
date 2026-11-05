import type { CSSProperties } from 'react'
import { useT, type Translate } from './i18n'
import type { LoopKey, NODES, RunState } from './types'

type NodeName = (typeof NODES)[number]

const W = 130, H = 44
// [x, y, subtitle, tone]; layout follows the proposed-methods slide, flow wraps right-to-left on row 2.
// Node names are the graph's identifiers and stay as they are; only the subtitles are translated.
function positions(t: Translate): Record<NodeName, [number, number, string, string]> {
  return {
    ingest: [10, 70, t('parse JD + profile', 'đọc JD + hồ sơ cty'), 'slate'],
    clarify: [175, 70, t('ambiguous criteria?', 'tiêu chí mơ hồ?'), 'amber'],
    search: [340, 70, t('chunks · per requirement', 'đoạn CV · theo yêu cầu'), 'sky'],
    expand_query: [340, 160, t('related tools, ask HR', 'công cụ liên quan, hỏi HR'), 'sky'],
    investigate: [505, 160, t('tool-using agent', 'agent dùng công cụ'), 'amber'],
    rerank: [505, 70, t('CE on passages', 'cross-encoder chấm đoạn'), 'purple'],
    assess: [670, 70, t('4 LLM + depth + 3 rules', '4 LLM + mức độ + 3 luật'), 'slate'],
    verify_evidence: [835, 70, t('grounded + supported?', 'có thật + chứng minh?'), 'rose'],
    score: [835, 250, t('Σ w·S, tier, reasons', 'Σ w·S, nhóm, lý do'), 'purple'],
    arbitrate: [670, 250, t('near-tie judge', 'phân xử gần hoà'), 'purple'],
    report: [505, 250, t('explanations', 'giải thích'), 'emerald'],
    hr_review: [340, 250, t('human approval', 'HR duyệt'), 'emerald'],
  }
}

const EDGES = [
  'M140 92 L175 92', 'M305 92 L340 92', 'M470 92 L505 92', 'M635 92 L670 92', 'M800 92 L835 92',
  'M900 114 L900 250', 'M835 272 L800 272', 'M670 272 L635 272', 'M505 272 L470 272',
  'M340 272 L254 272', 'M375 114 L375 160', 'M470 108 L505 168', 'M570 160 L570 114',
]

// Loop edges: path, label anchor, tone.
function loopDefs(t: Translate): { key: LoopKey; id: string; name: string; d: string; x: number; y: number; tone: string }[] {
  return [
    { key: 'clarify', id: 'L1', name: t('Ask HR', 'Hỏi HR'), d: 'M240 70 L240 30 L75 30 L75 70', x: 157, y: 30, tone: 'amber' },
    { key: 'expand', id: 'L2', name: t('Expand + ask HR', 'Mở rộng + hỏi HR'), d: 'M440 160 L440 114', x: 505, y: 138, tone: 'sky' },
    { key: 'investigate', id: 'L6', name: t('Investigate knock-outs', 'Điều tra CV bị loại'), d: 'M635 172 C675 172 675 198 635 198', x: 700, y: 186, tone: 'amber' },
    { key: 'arbitrate', id: 'L3', name: 'Arbiter', d: 'M705 250 C705 212 765 212 765 250', x: 735, y: 200, tone: 'purple' },
    { key: 'verify', id: 'L4', name: t('Block hallucination', 'Chặn ảo giác'), d: 'M900 70 L900 30 L735 30 L735 70', x: 817, y: 30, tone: 'rose' },
    { key: 'review', id: 'L5', name: t('HR re-score', 'HR chấm lại'), d: 'M405 294 L405 345 L900 345 L900 294', x: 652, y: 345, tone: 'emerald' },
  ]
}
const TONES = ['slate', 'amber', 'sky', 'purple', 'rose', 'emerald']

const fmtMs = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`)

export default function GraphView({ run }: { run: RunState }) {
  const { t } = useT()
  const pos = positions(t)
  const loops = run.values.loops
  return (
    <section className="card">
      <h2>{t('Agent graph', 'Sơ đồ agent')}</h2>
      <svg viewBox="0 0 1000 380" className="graph" role="img" aria-label={t('Agent state graph', 'Sơ đồ trạng thái của agent')}>
        <defs>
          {TONES.map((tone) => (
            <marker key={tone} id={`ar-${tone}`} viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0 1 L8 5 L0 9z" style={{ fill: `var(--${tone})` }} />
            </marker>
          ))}
        </defs>

        {EDGES.map((d) => <path key={d} d={d} className="edge" markerEnd="url(#ar-slate)" />)}

        {loopDefs(t).map((l) => {
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

        {(Object.keys(pos) as NodeName[]).map((name) => {
          const [x, y, sub, tone] = pos[name]
          const st = run.nodes[name]
          const status = st?.status ?? 'idle'
          return (
            <g key={name} className={`node ${status}`} style={{ '--tone': `var(--${tone})` } as CSSProperties}>
              <rect x={x} y={y} width={W} height={H} rx={7} />
              <text x={x + 10} y={y + 18} className="n-title">{name}</text>
              <text x={x + 10} y={y + 34} className="n-sub">{status === 'waiting' ? t('waiting for HR', 'chờ HR') : sub}</text>
              {st?.ms != null && <text x={x + W - 8} y={y + 18} textAnchor="end" className="n-ms">{fmtMs(st.ms)}</text>}
              <circle cx={x + W - 10} cy={y + 33} r={4} className="dot" />
            </g>
          )
        })}
      </svg>
      <div className="legend">
        <span><i className="dot-idle" /> {t('idle', 'chưa chạy')}</span>
        <span><i className="dot-running" /> {t('running', 'đang chạy')}</span>
        <span><i className="dot-done" /> {t('done', 'xong')}</span>
        <span><i className="dot-waiting" /> {t('waiting for HR', 'chờ HR')}</span>
        <span className="muted">{t('dashed = reflection loop; flashes when it fires', 'nét đứt = vòng lặp phản tư; nhấp nháy khi được kích hoạt')}</span>
      </div>
    </section>
  )
}
