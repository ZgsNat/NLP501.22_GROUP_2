import { useEffect, useRef, useState } from 'react'
import { CRITERIA, type Interrupt, type LogLine, type ResumeBody, type RunState } from './types'

const pretty = (k: string) => k.replace(/_/g, ' ')

export function Timeline({ logs }: { logs: LogLine[] }) {
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => { box.current?.scrollTo({ top: box.current.scrollHeight }) }, [logs.length])

  // Consecutive logs from the same node form a group, so a node that re-runs gets a new group.
  const groups: { node: string; lines: LogLine[] }[] = []
  for (const l of logs) {
    const g = groups[groups.length - 1]
    if (g && g.node === l.node) g.lines.push(l)
    else groups.push({ node: l.node, lines: [l] })
  }
  return (
    <section className="card">
      <h2>Event timeline <span className="muted small">why the agent did what it did</span></h2>
      <div className="timeline" ref={box}>
        {groups.length === 0 && <p className="muted">No events yet.</p>}
        {groups.map((g, i) => (
          <div key={i} className="group">
            <div className="group-node mono">{g.node}</div>
            <ul>
              {g.lines.map((l, j) => (
                <li key={j} className={`log ${l.level}`}>
                  {l.level === 'loop' && <b>↺ loop </b>}
                  {l.level === 'warn' && <b>⚠ </b>}
                  {l.message}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </section>
  )
}

export function Requirements({ run }: { run: RunState }) {
  const { requirements: r, weights, query_terms, policy_context } = run.values
  if (!r && !weights && !query_terms && !policy_context) return null
  const list = (items: string[]) => (items.length ? items.map((x) => <span key={x} className="badge gray">{x}</span>) : <span className="muted">-</span>)
  return (
    <section className="card">
      <h2>Requirements</h2>
      {r && (
        <dl>
          <dt>Title</dt><dd>{r.title}</dd>
          <dt>Must have</dt><dd className="chips">{list(r.must_have)}</dd>
          <dt>Nice to have</dt><dd className="chips">{list(r.nice_to_have)}</dd>
          <dt>Languages</dt><dd className="chips">{list(r.languages)}</dd>
          <dt>Test types</dt><dd className="chips">{list(r.test_types)}</dd>
          <dt>Min years / depth</dt><dd>{r.min_years ?? '-'} yrs / level {r.depth_level ?? '-'}</dd>
          {r.ambiguities.length > 0 && <><dt>Ambiguities</dt><dd><ul>{r.ambiguities.map((a) => <li key={a}>{a}</li>)}</ul></dd></>}
        </dl>
      )}
      {weights && (
        <>
          <h3>Weights</h3>
          {CRITERIA.map((c) => (
            <div key={c} className="wrow">
              <span>{pretty(c)}</span>
              <span className="bar"><i style={{ width: `${(weights[c] ?? 0) * 100}%` }} /></span>
              <span className="mono">{Math.round((weights[c] ?? 0) * 100)}%</span>
            </div>
          ))}
        </>
      )}
      {query_terms && (
        <>
          <h3>Query terms <span className="muted small">highlighted = added by expansion</span></h3>
          <div className="chips">
            {query_terms.map((t) => <span key={t} className={`badge ${run.expanded.includes(t) ? 'sky' : 'gray'}`}>{t}</span>)}
          </div>
        </>
      )}
      {policy_context && policy_context.length > 0 && (
        <>
          <h3>Policy context (RAG)</h3>
          {policy_context.map((p, i) => (
            <blockquote key={i}>{p.text}<cite>{p.source}</cite></blockquote>
          ))}
        </>
      )}
    </section>
  )
}

export function InterruptPanel({ interrupt, criteria, onSubmit }: {
  interrupt: Interrupt
  criteria: { key: string; label: string }[]
  onSubmit: (body: ResumeBody) => void
}) {
  return (
    <section className="card interrupt">
      <h2><span className="badge amber">waiting for HR</span> {interrupt.kind === 'clarification' ? 'Clarification needed' : 'Review ranking'}</h2>
      {interrupt.kind === 'clarification'
        ? <Clarify questions={interrupt.payload.questions} onSubmit={onSubmit} />
        : <Review payload={interrupt.payload} criteria={criteria} onSubmit={onSubmit} />}
    </section>
  )
}

function Clarify({ questions, onSubmit }: { questions: string[]; onSubmit: (b: ResumeBody) => void }) {
  const [answers, setAnswers] = useState<Record<string, string>>({})
  return (
    <form onSubmit={(e) => { e.preventDefault(); onSubmit({ answers }) }}>
      {questions.map((q) => (
        <label key={q}>{q}
          <input value={answers[q] ?? ''} onChange={(e) => setAnswers({ ...answers, [q]: e.target.value })} />
        </label>
      ))}
      <button className="primary" type="submit">Submit answers</button>
    </form>
  )
}

function Review({ payload, criteria, onSubmit }: {
  payload: Extract<Interrupt, { kind: 'review' }>['payload']
  criteria: { key: string; label: string }[]
  onSubmit: (b: ResumeBody) => void
}) {
  const [w, setW] = useState<Record<string, number>>(() => Object.fromEntries(CRITERIA.map((c) => [c, payload.weights[c] ?? 0])))
  const sum = CRITERIA.reduce((a, c) => a + w[c], 0)
  const label = (k: string) => criteria.find((c) => c.key === k)?.label ?? pretty(k)
  const adjust = () => onSubmit({
    action: 'adjust',
    weights: Object.fromEntries(CRITERIA.map((c) => [c, Math.round((w[c] / sum) * 10000) / 10000])),
  })
  return (
    <div className="review">
      <div>
        <h3>Current ranking</h3>
        <ol className="mini-rank">
          {payload.ranking.map((r) => (
            <li key={r.cv_id} className={r.knocked_out ? 'ko' : ''}>
              <span className="mono">{r.cv_id}</span> {r.headline} <b>{Math.round(r.total * 100)}%</b>
            </li>
          ))}
        </ol>
      </div>
      <div>
        <h3>Weights <span className="muted small">normalised to 100% on submit</span></h3>
        {CRITERIA.map((c) => (
          <div key={c} className="wrow slider">
            <span title={label(c)}>{label(c).split(' (')[0]}</span>
            <input type="range" min={0} max={1} step={0.01} value={w[c]} onChange={(e) => setW({ ...w, [c]: +e.target.value })} />
            <span className="mono">{sum ? Math.round((w[c] / sum) * 100) : 0}%</span>
          </div>
        ))}
        <div className="row">
          <button className="primary" onClick={() => onSubmit({ action: 'approve' })}>Approve</button>
          <button className="ghost" disabled={sum <= 0} onClick={adjust}>Re-score with new weights</button>
        </div>
      </div>
    </div>
  )
}
