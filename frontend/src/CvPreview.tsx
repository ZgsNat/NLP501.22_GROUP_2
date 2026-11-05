import { useEffect, useState, type CSSProperties } from 'react'
import { api } from './api'
import { useT } from './i18n'
import type { CV, SkillStrength } from './types'

// Section kind → tone (a CSS colour variable); unknown sections fall back to slate.
const TONE: Record<string, string> = {
  header: 'slate', summary: 'sky', experience: 'emerald', projects: 'purple', skills: 'slate',
  education: 'amber', certifications: 'amber', languages: 'sky',
}
export const toneOf = (kind: string) => TONE[kind.toLowerCase()] ?? 'slate'
const toneStyle = (kind: string) => ({ '--tone': `var(--${toneOf(kind)})` }) as CSSProperties

// Section kinds are the parser's labels (they also appear in the LLM prompt), so they stay in English.
export function SecTag({ section }: { section: string | null | undefined }) {
  if (!section) return null
  return <span className="sec" style={toneStyle(section)}>{section}</span>
}

export default function CvPreview({ id, onClose }: { id: string; onClose: () => void }) {
  const { t, term } = useT()
  const [cv, setCv] = useState<CV | string | null>(null)
  const strength: Record<SkillStrength, { cls: string; hint: string; name: string }> = {
    used: { cls: 'green', name: t('used', term('Used')), hint: t('used in a described project or job', 'đã dùng trong một dự án hoặc công việc được mô tả') },
    listed: { cls: 'gray', name: t('listed', term('Listed')), hint: t('only named in a skills list', 'chỉ được nêu trong danh sách kỹ năng') },
    hedged: { cls: 'amber', name: t('hedged', term('Hedged')),
      hint: t('mentioned with a hedge ("familiar with", "learning")', 'được nhắc kèm từ rào đón ("làm quen", "đang học")') },
  }
  const langName = { en: t('English', 'Tiếng Anh'), vi: t('Vietnamese', 'Tiếng Việt'), mixed: t('Mixed EN/VI', 'Trộn Anh/Việt') } as const

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
      <div className="modal-box" role="dialog" aria-label={`${t('Parse preview', 'Kết quả bóc tách')} ${id}`} onClick={(e) => e.stopPropagation()}>
        <div className="row">
          <h2 className="grow"><span className="mono">{id}</span> {t('Parse preview', 'Kết quả bóc tách CV')}</h2>
          <button className="ghost" onClick={onClose}>{t('Close', 'Đóng')}</button>
        </div>
        {cv === null && <p className="muted">{t('Loading…', 'Đang tải…')}</p>}
        {typeof cv === 'string' && <p className="err">{cv}</p>}
        {typeof cv === 'object' && cv && p && (
          <>
            <p className="pv-headline">{cv.headline}</p>

            <h3><span className="step">1</span> {t('Language and redaction', `Ngôn ngữ và ${term('Redaction')}`)}</h3>
            <div className="chips">
              <span className="badge sky">{langName[p.lang] ?? p.lang}</span>
              {removed.length
                ? <span className="badge rose">{t('removed', 'đã che')}: {removed.map(([k, n]) => `${k} ${n}`).join(', ')}</span>
                : <span className="badge gray">{t('removed: nothing (no PII found)', 'đã che: không có (không tìm thấy thông tin cá nhân)')}</span>}
            </div>

            <h3><span className="step">2</span> {t('Sections detected', 'Các mục được nhận diện')}</h3>
            <div className="chips">
              {p.sections.map((s, i) => <span key={i} className="sec big" style={toneStyle(s.kind)} title={`${t('heading', 'tiêu đề')}: ${s.heading}`}>{s.kind}</span>)}
            </div>

            <h3><span className="step">3</span> {t('Chunks', term('Chunk'))}{' '}
              <span className="muted small">{t('each carries a contextual header, so retrieval knows where text came from',
                'mỗi đoạn có tiêu đề ngữ cảnh để bước truy hồi biết văn bản đến từ đâu')}</span></h3>
            <div className="chunks">
              {p.chunks.map((c) => (
                <div key={c.id} className="chunk" style={toneStyle(c.section)}>
                  <div className="chunk-head"><SecTag section={c.section} /><span className="mono muted">{c.header}</span></div>
                  <div className="chunk-text">{c.text}</div>
                </div>
              ))}
            </div>

            <h3><span className="step">4</span> {t('Skills and how strongly they are evidenced', 'Kỹ năng và mức độ có bằng chứng')}</h3>
            {p.skills.length === 0 ? <p className="muted">{t('No known tools or skills detected.', 'Không nhận diện được công cụ hay kỹ năng nào.')}</p> : (
              <table className="rank skills">
                <thead><tr>
                  <th>{t('Skill', 'Kỹ năng')}</th><th>{t('Strength', 'Mức bằng chứng')}</th>
                  <th>{t('Section', 'Mục')}</th><th>{t('Line that shows it', 'Dòng thể hiện kỹ năng')}</th>
                </tr></thead>
                <tbody>
                  {p.skills.map((s, i) => (
                    <tr key={i}>
                      <td><b>{s.skill}</b></td>
                      <td><span className={`badge ${strength[s.strength].cls}`} title={strength[s.strength].hint}>{strength[s.strength].name}</span></td>
                      <td><SecTag section={s.section} /></td>
                      <td className="small">{s.line}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}

            <h3><span className="step">5</span> {t('Experience facts', 'Dữ kiện kinh nghiệm')}</h3>
            <dl className="facts">
              <dt>{t('Total years', 'Tổng số năm')}</dt><dd>{p.years_total ?? '-'}</dd>
              <dt>{t('Automation years', 'Số năm làm tự động hoá')}</dt><dd>{p.years_automation ?? '-'}</dd>
              <dt>{t('Depth cue', `${term('Depth')} (depth cue)`)}</dt>
              <dd>{p.depth_cue > 0 ? <b>L{p.depth_cue}</b> : '-'}{p.depth_cue_line && <blockquote>{p.depth_cue_line}</blockquote>}</dd>
            </dl>

            <details className="rawtext">
              <summary>{t('Processed text (redacted)', 'Văn bản đã xử lý (đã che thông tin cá nhân)')}</summary>
              <pre className="cv">{cv.text}</pre>
            </details>
          </>
        )}
      </div>
    </div>
  )
}
