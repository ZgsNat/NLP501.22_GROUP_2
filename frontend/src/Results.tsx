import { Fragment, useState } from 'react'
import { api } from './api'
import { CRITERIA, type Assessment, type Config, type RunState, type Values } from './types'

const pct = (x: number) => `${Math.round(x * 100)}%`

export default function Results({ run, config }: { run: RunState; config: Config | null }) {
  const [sel, setSel] = useState<string | null>(null)
  const { ranking, report, assessments, weights } = run.values
  if (!ranking?.length) return null
  const labels = Object.fromEntries((config?.criteria ?? []).map((c) => [c.key, c.label]))
  const entry = report?.candidates.find((c) => c.cv_id === sel)

  return (
    <section className="card">
      <h2>Results {run.done && <span className="badge green">final</span>}</h2>
      {report && <p>{report.summary}</p>}
      <table className="rank">
        <thead><tr><th>#</th><th>CV</th><th>Headline</th><th>Total</th><th></th><th>Arbiter note</th></tr></thead>
        <tbody>
          {ranking.map((r) => (
            <tr key={r.cv_id} className={`${r.cv_id === sel ? 'sel' : ''} ${r.knocked_out ? 'ko' : ''}`} onClick={() => setSel(r.cv_id === sel ? null : r.cv_id)}>
              <td>{r.rank}</td>
              <td className="mono">{r.cv_id}</td>
              <td>{r.headline}</td>
              <td className="mono"><b>{pct(r.total)}</b></td>
              <td>{r.knocked_out && <span className="badge rose">knocked out</span>}</td>
              <td className="small">{r.arbiter_note}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="muted small">Click a row for candidate detail.</p>
      {sel && (
        <Detail key={sel} id={sel} a={assessments?.[sel]} weights={weights} labels={labels}
          explanation={entry?.explanation} strengths={entry?.strengths} gaps={entry?.gaps} />
      )}
    </section>
  )
}

function Detail({ id, a, weights, labels, explanation, strengths, gaps }: {
  id: string
  a: Assessment | undefined
  weights: Values['weights'] | undefined
  labels: Record<string, string>
  explanation?: string
  strengths?: string[]
  gaps?: string[]
}) {
  const [cv, setCv] = useState<{ text: string } | 'loading' | string | null>(null)
  const label = (k: string) => labels[k] ?? k.replace(/_/g, ' ')
  const showCv = async () => {
    if (cv) return setCv(null)
    setCv('loading')
    try { setCv(await api.getCv(id)) } catch (e) { setCv((e as Error).message) }
  }
  const quotes = a ? CRITERIA.flatMap((c) => a.criteria[c].evidence.filter((e) => e.grounded).map((e) => e.quote)) : []

  return (
    <div className="detail">
      <h3>{id}</h3>
      {a?.knocked_out && <div className="banner rose">Knocked out: {a.knockout_reason ?? 'must-have missing'}</div>}
      {explanation && <p>{explanation}</p>}
      <div className="two">
        {strengths && strengths.length > 0 && <div><h4>Strengths</h4><ul className="good">{strengths.map((s) => <li key={s}>{s}</li>)}</ul></div>}
        {gaps && gaps.length > 0 && <div><h4>Gaps</h4><ul className="bad">{gaps.map((s) => <li key={s}>{s}</li>)}</ul></div>}
      </div>
      {!a ? <p className="muted">No assessment for this candidate (not assessed yet).</p> : (
        <div className="two radar-wrap">
          <Radar a={a} weights={weights} label={label} />
          <div>
            {CRITERIA.map((c) => {
              const s = a.criteria[c]
              return (
                <div key={c} className="crit">
                  <div className="wrow">
                    <span>{label(c).split(' (')[0]}</span>
                    <span className="bar"><i style={{ width: `${s.score}%` }} /></span>
                    <span className="mono">{Math.round(s.score)}</span>
                  </div>
                  <p className="small">{s.rationale}</p>
                  {s.evidence.map((e, i) => (
                    <blockquote key={i} className={`ev ${e.grounded === true ? 'g' : e.grounded === false ? 'u' : 'x'}`}>
                      <b title={e.grounded === true ? 'grounded in CV' : e.grounded === false ? 'ungrounded' : 'unchecked'}>
                        {e.grounded === true ? '✓' : e.grounded === false ? '✗' : '?'}
                      </b> {e.quote}
                    </blockquote>
                  ))}
                </div>
              )
            })}
            <p className="muted small">Attempts: {a.attempts}</p>
          </div>
        </div>
      )}
      <button className="ghost" onClick={showCv}>{cv ? 'Hide CV' : 'View CV'}</button>
      {cv === 'loading' && <p className="muted">Loading…</p>}
      {typeof cv === 'string' && cv !== 'loading' && <p className="err">{cv}</p>}
      {cv && typeof cv === 'object' && <pre className="cv">{highlight(cv.text, quotes)}</pre>}
    </div>
  )
}

// Wrap each quote occurrence in <mark>; overlapping ranges are merged.
function highlight(text: string, quotes: string[]) {
  const low = text.toLowerCase()
  const ranges: [number, number][] = []
  for (const q of quotes) {
    const needle = q.trim().toLowerCase()
    if (!needle) continue
    const i = low.indexOf(needle)
    if (i >= 0) ranges.push([i, i + needle.length])
  }
  ranges.sort((x, y) => x[0] - y[0])
  const merged: [number, number][] = []
  for (const r of ranges) {
    const last = merged[merged.length - 1]
    if (last && r[0] <= last[1]) last[1] = Math.max(last[1], r[1])
    else merged.push([...r])
  }
  const out = []
  let pos = 0
  merged.forEach(([s, e], i) => {
    out.push(<Fragment key={`t${i}`}>{text.slice(pos, s)}</Fragment>, <mark key={`m${i}`}>{text.slice(s, e)}</mark>)
    pos = e
  })
  out.push(<Fragment key="end">{text.slice(pos)}</Fragment>)
  return out
}

// Hand-drawn SVG radar: scores 0-100 on the rim scale; weights scaled so the largest reaches the rim.
function Radar({ a, weights, label }: { a: Assessment; weights: Values['weights'] | undefined; label: (k: string) => string }) {
  const C = 170, R = 105, n = CRITERIA.length
  const pt = (i: number, f: number, r = R) => {
    const t = (-90 + (360 / n) * i) * (Math.PI / 180)
    return [C + Math.cos(t) * r * f, C + Math.sin(t) * r * f] as const
  }
  const poly = (fs: number[]) => fs.map((f, i) => pt(i, f).map((v) => v.toFixed(1)).join(',')).join(' ')
  const maxW = weights ? Math.max(...CRITERIA.map((c) => weights[c] ?? 0)) : 0
  return (
    <svg viewBox="0 0 340 340" className="radar" role="img" aria-label="Criterion scores radar">
      {[0.25, 0.5, 0.75, 1].map((f) => <polygon key={f} points={poly(CRITERIA.map(() => f))} className="ring" />)}
      {CRITERIA.map((c, i) => {
        const [x, y] = pt(i, 1), [lx, ly] = pt(i, 1, R + 18)
        return (
          <g key={c}>
            <line x1={C} y1={C} x2={x} y2={y} className="ring" />
            <text x={lx} y={ly + 3} textAnchor={lx < C - 5 ? 'end' : lx > C + 5 ? 'start' : 'middle'} className="rlabel">{label(c).split(' (')[0].split(' / ')[0]}</text>
          </g>
        )
      })}
      {weights && maxW > 0 && <polygon points={poly(CRITERIA.map((c) => (weights[c] ?? 0) / maxW))} className="poly-w" />}
      <polygon points={poly(CRITERIA.map((c) => a.criteria[c].score / 100))} className="poly-s" />
      {CRITERIA.map((c, i) => {
        const [x, y] = pt(i, a.criteria[c].score / 100)
        return <circle key={c} cx={x} cy={y} r={3} className="pt"><title>{label(c)}: {Math.round(a.criteria[c].score)}</title></circle>
      })}
      <g transform="translate(8 326)" className="rlabel">
        <rect width="10" height="3" y="-3" className="key-s" /><text x="14">score</text>
        <rect x="60" width="10" height="3" y="-3" className="key-w" /><text x="74">JD weight (scaled)</text>
      </g>
    </svg>
  )
}
