import { useEffect, useState, type CSSProperties } from 'react'
import { api } from './api'
import type { CV, SkillStrength } from './types'

// Section kind → tone (a CSS colour variable); unknown sections fall back to slate.
const TONE: Record<string, string> = {
  header: 'slate', summary: 'sky', experience: 'emerald', projects: 'purple', skills: 'slate',
  education: 'amber', certifications: 'amber', languages: 'sky',
}
export const toneOf = (kind: string) => TONE[kind.toLowerCase()] ?? 'slate'
const toneStyle = (kind: string) => ({ '--tone': `var(--${toneOf(kind)})` }) as CSSProperties

export function SecTag({ section }: { section: string | null | undefined }) {
  if (!section) return null
  return <span className="sec" style={toneStyle(section)}>{section}</span>
}

const STRENGTH: Record<SkillStrength, { cls: string; hint: string }> = {
  used: { cls: 'green', hint: 'used in a described project or job' },
  listed: { cls: 'gray', hint: 'only named in a skills list' },
  hedged: { cls: 'amber', hint: 'mentioned with a hedge ("familiar with", "learning")' },
}
const LANG = { en: 'English', vi: 'Vietnamese', mixed: 'Mixed EN/VI' } as const

export default function CvPreview({ id, onClose }: { id: string; onClose: () => void }) {
  const [cv, setCv] = useState<CV | string | null>(null)

  useEffect(() => {
    let live = true
    setCv(null)
    api.getCv(id).then((c) => live && setCv(c)).catch((e) => live && setCv((e as Error).message))
    return () => { live = false }
  }, [id])
  useEffect(() => {
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])

  const p = typeof cv === 'object' && cv ? cv.profile : null
  const removed = p ? Object.entries(p.redactions).filter(([, n]) => n > 0) : []

  return (
    <div className="modal" onClick={onClose}>
      <div className="modal-box" role="dialog" aria-label={`Parse preview ${id}`} onClick={(e) => e.stopPropagation()}>
        <div className="row">
          <h2 className="grow"><span className="mono">{id}</span> Parse preview</h2>
          <button className="ghost" onClick={onClose}>Close</button>
        </div>
        {cv === null && <p className="muted">Loading…</p>}
        {typeof cv === 'string' && <p className="err">{cv}</p>}
        {typeof cv === 'object' && cv && p && (
          <>
            <p className="pv-headline">{cv.headline}</p>

            <h3><span className="step">1</span> Language and redaction</h3>
            <div className="chips">
              <span className="badge sky">{LANG[p.lang] ?? p.lang}</span>
              {removed.length
                ? <span className="badge rose">removed: {removed.map(([k, n]) => `${k} ${n}`).join(', ')}</span>
                : <span className="badge gray">removed: nothing (no PII found)</span>}
            </div>

            <h3><span className="step">2</span> Sections detected</h3>
            <div className="chips">
              {p.sections.map((s, i) => <span key={i} className="sec big" style={toneStyle(s.kind)} title={`heading: ${s.heading}`}>{s.kind}</span>)}
            </div>

            <h3><span className="step">3</span> Chunks <span className="muted small">each carries a contextual header, so retrieval knows where text came from</span></h3>
            <div className="chunks">
              {p.chunks.map((c) => (
                <div key={c.id} className="chunk" style={toneStyle(c.section)}>
                  <div className="chunk-head"><SecTag section={c.section} /><span className="mono muted">{c.header}</span></div>
                  <div className="chunk-text">{c.text}</div>
                </div>
              ))}
            </div>

            <h3><span className="step">4</span> Skills and how strongly they are evidenced</h3>
            {p.skills.length === 0 ? <p className="muted">No known tools or skills detected.</p> : (
              <table className="rank skills">
                <thead><tr><th>Skill</th><th>Strength</th><th>Section</th><th>Line that shows it</th></tr></thead>
                <tbody>
                  {p.skills.map((s, i) => (
                    <tr key={i}>
                      <td><b>{s.skill}</b></td>
                      <td><span className={`badge ${STRENGTH[s.strength].cls}`} title={STRENGTH[s.strength].hint}>{s.strength}</span></td>
                      <td><SecTag section={s.section} /></td>
                      <td className="small">{s.line}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            <h3><span className="step">5</span> Experience facts</h3>
            <dl className="facts">
              <dt>Total years</dt><dd>{p.years_total ?? '-'}</dd>
              <dt>Automation years</dt><dd>{p.years_automation ?? '-'}</dd>
              <dt>Depth cue</dt>
              <dd>{p.depth_cue > 0 ? <b>L{p.depth_cue}</b> : '-'}{p.depth_cue_line && <blockquote>{p.depth_cue_line}</blockquote>}</dd>
            </dl>

            <details className="rawtext">
              <summary>Processed text (redacted)</summary>
              <pre className="cv">{cv.text}</pre>
            </details>
          </>
        )}
      </div>
    </div>
  )
}
