import { useEffect, useRef, useState } from 'react'
import { CRITERIA, type Interrupt, type SubstituteProposal, type LogLine, type ResumeBody, type RunState } from './types'

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

const TITLE = { clarification: 'Clarification needed', substitutes: 'Accept related tools?', review: 'Review ranking',
  chat: 'The agent answered' }

export function InterruptPanel({ interrupt, criteria, onSubmit }: {
  interrupt: Interrupt
  criteria: { key: string; label: string }[]
  onSubmit: (body: ResumeBody) => void
}) {
  return (
    <section className="card interrupt">
      <h2><span className="badge amber">waiting for HR</span> {TITLE[interrupt.kind]}</h2>
      {interrupt.kind === 'clarification' && <Clarify questions={interrupt.payload.questions} onSubmit={onSubmit} />}
      {interrupt.kind === 'substitutes' && <Substitutes proposals={interrupt.payload.proposals} onSubmit={onSubmit} />}
      {interrupt.kind === 'review' && <Review payload={interrupt.payload} criteria={criteria} onSubmit={onSubmit} />}
      {interrupt.kind === 'chat' && <Chat payload={interrupt.payload} onSubmit={onSubmit} />}
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

// L2: too few CVs show a must-have, so HR decides which related tools may stand in for it.
function Substitutes({ proposals, onSubmit }: { proposals: SubstituteProposal[]; onSubmit: (b: ResumeBody) => void }) {
  const [accept, setAccept] = useState<Record<string, string[]>>(() => Object.fromEntries(proposals.map((p) => [p.must_have, p.related])))
  const flip = (m: string, tool: string) =>
    setAccept((a) => ({ ...a, [m]: a[m].includes(tool) ? a[m].filter((t) => t !== tool) : [...a[m], tool] }))
  return (
    <form onSubmit={(e) => { e.preventDefault(); onSubmit({ accept }) }}>
      {proposals.map((p) => (
        <fieldset key={p.must_have} className="subst">
          <legend>
            Only {p.covered} CV{p.covered === 1 ? '' : 's'} show <b>{p.must_have}</b>. Accept these related tools as substitutes?
          </legend>
          <div className="chips">
            {p.related.map((t) => (
              <label key={t} className="check">
                <input type="checkbox" checked={accept[p.must_have].includes(t)} onChange={() => flip(p.must_have, t)} /> {t}
              </label>
            ))}
          </div>
        </fieldset>
      ))}
      <p className="muted small">Unchecking every tool rejects substitutes for that must-have.</p>
      <button className="primary" type="submit">Continue search</button>
    </form>
  )
}

function Review({ payload, criteria, onSubmit }: {
  payload: Extract<Interrupt, { kind: 'review' }>['payload']
  criteria: { key: string; label: string }[]
  onSubmit: (b: ResumeBody) => void
}) {
  const [w, setW] = useState<Record<string, number>>(() => Object.fromEntries(CRITERIA.map((c) => [c, payload.weights[c] ?? 0])))
  const [say, setSay] = useState('')
  const [dec, setDec] = useState<Record<string, { action: 'approve' | 'reject'; reason: string }>>({})
  const decisions = Object.entries(dec).filter(([, d]) => d.reason.trim()).map(([cv_id, d]) => ({ cv_id, ...d }))
  const mark = (id: string, action: 'approve' | 'reject') =>
    setDec((d) => (d[id]?.action === action ? (({ [id]: _, ...rest }) => rest)(d) : { ...d, [id]: { action, reason: d[id]?.reason ?? '' } }))
  const left = payload.adjustments_left
  const shown = payload.ranking.filter((r) => (r.status ?? (r.knocked_out ? 'knocked_out' : 'ranked')) === 'ranked')
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
          {shown.map((r) => (
            <li key={r.cv_id}>
              <span className="mono">{r.cv_id}</span> {r.headline} <b>{Math.round(r.total * 100)}%</b>{' '}
              <button className={`chip ${dec[r.cv_id]?.action === 'approve' ? 'on' : ''}`} onClick={() => mark(r.cv_id, 'approve')}>✓</button>
              <button className={`chip ${dec[r.cv_id]?.action === 'reject' ? 'on' : ''}`} onClick={() => mark(r.cv_id, 'reject')}>✗</button>
              {dec[r.cv_id] && (
                <input className="reason" placeholder="reason (required)" value={dec[r.cv_id].reason}
                       onChange={(e) => setDec({ ...dec, [r.cv_id]: { ...dec[r.cv_id], reason: e.target.value } })} />
              )}
            </li>
          ))}
        </ol>
        <button className="ghost" disabled={!decisions.length || left === 0}
                onClick={() => onSubmit({ action: 'decide', decisions })}>
          Learn weights from my {decisions.length || ''} decision{decisions.length === 1 ? '' : 's'}
        </button>
        <p className="muted small">Approve or reject candidates with a reason: the weights that best explain your choices
          re-score the pool, and every decision goes to the audit log.</p>
        {payload.ranking.length > shown.length && (
          <p className="muted small">+ {payload.ranking.length - shown.length} more (not shortlisted, needs review or knocked out): see Results.</p>
        )}
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
        <h3>…or say it in words</h3>
        <textarea rows={2} value={say} placeholder="e.g. Certification matters more for this bank client; ignore domain"
                  onChange={(e) => setSay(e.target.value)} />
        <div className="row">
          <button className="primary" onClick={() => onSubmit({ action: 'approve' })}>Approve</button>
          <button className="ghost" disabled={sum <= 0 || left === 0} onClick={adjust}>Re-score with new weights</button>
          <button className="ghost" disabled={!say.trim() || left === 0}
                  onClick={() => onSubmit({ action: 'adjust', instruction: say.trim() })}>Re-score from my words</button>
        </div>
        {left !== undefined && <p className="muted small">{left} re-score round(s) left; each costs 0 new assessments.</p>}
      </div>
    </div>
  )
}

// P7: the agent's answer, and HR's next instruction in plain words (it may ask, re-weight, change requirements, decide).
function Chat({ payload, onSubmit }: { payload: Extract<Interrupt, { kind: 'chat' }>['payload']; onSubmit: (b: ResumeBody) => void }) {
  const [say, setSay] = useState('')
  return (
    <form onSubmit={(e) => { e.preventDefault(); if (say.trim()) onSubmit({ message: say.trim() }) }}>
      <p className="bubble agent">{payload.message}</p>
      {payload.shortlist.length > 0 && (
        <ol className="mini-rank">
          {payload.shortlist.map((r) => <li key={r.cv_id}><span className="mono">{r.cv_id}</span> {r.headline} <b>{Math.round(r.total * 100)}%</b></li>)}
        </ol>
      )}
      <textarea rows={2} value={say} onChange={(e) => setSay(e.target.value)}
                placeholder="e.g. Why is CV-012 below CV-030? · Make Docker a must-have · Reject CV-030, too junior" />
      <div className="row">
        <button className="primary" type="submit" disabled={!say.trim() || payload.turns_left <= 0}>Send to agent</button>
        <button className="ghost" type="button" onClick={() => onSubmit({ action: 'approve' })}>Approve and finish</button>
      </div>
      <p className="muted small">{payload.turns_left} message(s) left in this run.</p>
    </form>
  )
}
