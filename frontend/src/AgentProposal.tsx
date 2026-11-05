import { useState } from 'react'
import { backendText, useT } from './i18n'
import type { RankRow, RunState } from './types'

function Why({ text }: { text: string }) {
  const { t } = useT()
  const [open, setOpen] = useState(false)
  return (
    <p className={`proposal-why ${open ? 'open' : ''}`} onClick={() => setOpen(!open)}
      title={open ? t('Collapse', 'Thu gọn') : t('Show in full', 'Xem đầy đủ')}>
      <b>{t('Why this place:', 'Vì sao ở vị trí này:')}</b> {text}
    </p>
  )
}

// What the assistant recommends for HR's goal, before the full ranking: the goal, the answer, then one card per
// recommended candidate with why it sits there, its strengths, and what to probe in interview.
export default function AgentProposal({ run, labels, onOpenCv, selected, onToggleCompare }: {
  run: RunState
  labels: Record<string, string>
  onOpenCv: (id: string) => void
  selected: string[]
  onToggleCompare: (id: string) => void
}) {
  const { t, lang, flag } = useT()
  const v = run.values
  const ids = v.shortlist ?? []
  if (run.mode !== 'agent' || (!ids.length && !v.reply)) return null
  const rows = new Map((v.ranking ?? []).map((r) => [r.cv_id, r]))
  const notes = new Map((v.report?.candidates ?? []).map((c) => [c.cv_id, c]))
  const label = (k: string) => (labels[k] ?? k.replace(/_/g, ' ')).split(' (')[0]
  // The backend writes gaps as "no <tool> in a role" (scoring.counterfactuals)
  const gapText = (g: string) => (lang === 'vi' ? g.replace(/^no (.+) in a role$/, 'chưa thấy $1 trong kinh nghiệm thực tế') : g)
  const tierOf = {
    strong: ['green', t('Strong fit', 'Rất phù hợp')],
    possible: ['amber', t('Possible fit', 'Có thể phù hợp')],
    weak: ['gray', t('Weak fit', 'Phù hợp thấp')],
  } as const

  return (
    <section className="proposal">
      {v.goal && <p className="proposal-goal"><span className="muted small">{t('Your goal:', 'Yêu cầu của bạn:')}</span> “{v.goal}”</p>}
      {v.reply && <div className="proposal-reply">{v.reply}</div>}
      {ids.length > 0 && (
        <>
          <h3 className="proposal-title">
            {t(`${ids.length} recommended candidates`, `Đề xuất ${ids.length} ứng viên`)}{' '}
            <span className="muted small">{t('in priority order · each one evidence-checked',
              'theo thứ tự ưu tiên · bằng chứng của từng người đã được kiểm chứng')}</span>
          </h3>
          <div className="proposal-grid">
            {ids.map((id, i) => {
              const r = rows.get(id) as RankRow | undefined
              const n = notes.get(id)
              const why = v.llm_reasons?.[id] || n?.explanation
              const tier = r?.tier ? tierOf[r.tier] : null
              return (
                <article key={id} className="proposal-card">
                  <header className="proposal-head">
                    <span className="proposal-rank">{i + 1}</span>
                    <div className="proposal-who">
                      <button className="link mono" onClick={() => onOpenCv(id)} title={t('Open the parsed CV', 'Xem CV đã bóc tách')}>{id}</button>
                      <div className="small">{r?.headline}</div>
                    </div>
                    {r && <b className="mono proposal-score">{Math.round(r.total * 100)}%</b>}
                  </header>
                  {tier && <span className={`badge ${tier[0]}`}>{tier[1]}</span>}
                  {why && <Why text={why} />}
                  {!!n?.strengths?.length && (
                    <p className="small"><b>{t('Strengths:', 'Điểm mạnh:')}</b> {n.strengths.map(label).join(', ')}</p>
                  )}
                  {!!n?.gaps?.length && (
                    <p className="small proposal-gaps"><b>{t('Ask in interview:', 'Nên hỏi khi phỏng vấn:')}</b> {n.gaps.map(gapText).join('; ')}</p>
                  )}
                  {r?.arbiter_note && <p className="small muted">⚖️ {backendText(lang, r.arbiter_note)}</p>}
                  {!!r?.flags?.length && <p className="small muted">⚠ {r.flags.map(flag).join(', ')}</p>}
                  <label className="small proposal-compare">
                    <input type="checkbox" checked={selected.includes(id)} onChange={() => onToggleCompare(id)} /> {t('Compare', 'So sánh')}
                  </label>
                </article>
              )
            })}
          </div>
        </>
      )}
    </section>
  )
}
