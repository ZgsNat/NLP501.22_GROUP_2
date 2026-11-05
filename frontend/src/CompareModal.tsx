import { backendText, useT } from './i18n'
import { CRITERIA, type Config, type Criterion, type RunState } from './types'

interface Props {
  selectedIds: string[]
  run: RunState
  config?: Config | null
  onClose: () => void
  onOpenCv: (id: string) => void
}

export default function CompareModal({ selectedIds, run, config, onClose, onOpenCv }: Props) {
  const { t, term, lang, crit } = useT()
  if (selectedIds.length === 0) return null

  const tiers: Record<string, { label: string; tone: string }> = {
    strong: { label: t('Strong', 'Xuất sắc'), tone: 'green' },
    possible: { label: t('Possible', 'Tiềm năng'), tone: 'sky' },
    weak: { label: t('Weak', 'Yếu / chưa đạt'), tone: 'amber' },
  }
  const apiLabel = (c: Criterion) => config?.criteria.find((x) => x.key === c)?.label
  const { ranking = [], assessments = {}, report } = run.values
  const weightsMap: Partial<Record<Criterion, number>> = run.values.weights || {}

  const candidatesData = selectedIds.map((id) => {
    const row = ranking.find((r) => r.cv_id === id)
    const a = assessments[id]
    const rep = report?.candidates.find((c) => c.cv_id === id)
    return { id, row, assessment: a, report: rep }
  })

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card compare-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h3>{t(`Side-by-side comparison (${selectedIds.length} candidates)`, `So sánh trực diện ứng viên (${selectedIds.length} ứng viên)`)}</h3>
            <p className="muted small">{t('Criterion scores, experience evidence and the AI assessment, side by side.',
              'Đối chiếu chi tiết điểm theo tiêu chí, bằng chứng kinh nghiệm và đánh giá của AI.')}</p>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label={t('Close', 'Đóng')}>✕</button>
        </div>

        <div className="compare-grid-container">
          <div className="compare-grid" style={{ gridTemplateColumns: `200px repeat(${selectedIds.length}, minmax(260px, 1fr))` }}>
            {/* Header row: Candidate basic info */}
            <div className="compare-cell label-cell sticky-col">
              <b>{t('Candidate', 'Thông tin ứng viên')}</b>
            </div>
            {candidatesData.map(({ id, row }) => (
              <div key={id} className="compare-cell candidate-head">
                <div className="flex-between">
                  <button className="link mono strong" onClick={() => onOpenCv(id)}>{id}</button>
                  {row?.tier && (
                    <span className={`badge ${tiers[row.tier]?.tone || 'gray'}`}>
                      {tiers[row.tier]?.label || row.tier}
                    </span>
                  )}
                </div>
                <div className="candidate-headline">{row?.headline || '-'}</div>
                <div className="candidate-score">
                  <span className="score-num">{Math.round((row?.total || 0) * 100)}%</span>
                  <span className="muted small">{t('Overall fit', 'Độ phù hợp tổng thể')}</span>
                </div>
              </div>
            ))}

            {/* Strengths & Gaps */}
            <div className="compare-cell label-cell sticky-col">
              <b>{t('Strengths & gaps', 'Điểm mạnh & điểm còn thiếu')}</b>
            </div>
            {candidatesData.map(({ id, report: rep }) => (
              <div key={id} className="compare-cell">
                <div className="tag-section">
                  <span className="small muted">{t('Main strengths:', 'Thế mạnh chính:')}</span>
                  <div className="chips">
                    {(rep?.strengths || []).map((s) => (
                      <span key={s} className="badge emerald">{s}</span>
                    ))}
                    {!rep?.strengths?.length && <span className="muted small">-</span>}
                  </div>
                </div>
                <div className="tag-section mt-2">
                  <span className="small muted">{t('Gaps:', 'Điểm còn thiếu:')}</span>
                  <div className="chips">
                    {(rep?.gaps || []).map((g) => (
                      <span key={g} className="badge amber">{g}</span>
                    ))}
                    {!rep?.gaps?.length && <span className="muted small">-</span>}
                  </div>
                </div>
              </div>
            ))}

            {/* Criteria breakdown */}
            {CRITERIA.map((criterion) => {
              const weightPct = Math.round((weightsMap[criterion] || 0) * 100)
              return (
                <div key={criterion} className="contents-row">
                  <div className="compare-cell label-cell sticky-col">
                    <div className="criterion-title">{crit(criterion, apiLabel(criterion))}</div>
                    <div className="muted small">{t('Weight', term('Weight'))}: {weightPct}%</div>
                  </div>
                  {candidatesData.map(({ id, assessment: a }) => {
                    const cScore = a?.criteria[criterion]
                    const score = cScore?.score ?? 0
                    const rationale = cScore?.rationale ? backendText(lang, cScore.rationale) : '-'
                    return (
                      <div key={id} className="compare-cell criterion-data">
                        <div className="score-bar-wrapper">
                          <span className="bar-val">{score}/100</span>
                          <div className="score-bar-track">
                            <div className="score-bar-fill" style={{ width: `${score}%` }} />
                          </div>
                        </div>
                        <p className="criterion-rationale muted small">{rationale}</p>
                      </div>
                    )
                  })}
                </div>
              )
            })}

            {/* Fact details: Years & Depth */}
            <div className="compare-cell label-cell sticky-col">
              <b>{t('Experience facts', 'Kinh nghiệm thực tế')}</b>
            </div>
            {candidatesData.map(({ id, assessment: a }) => (
              <div key={id} className="compare-cell">
                <div className="fact-item">
                  <span className="muted small">{t('Years in automation: ', 'Số năm làm tự động hoá: ')}</span>
                  <b>{a?.facts?.years_automation != null ? `${a.facts.years_automation} ${t('yrs', 'năm')}` : t('unknown', 'không xác định')}</b>
                </div>
                <div className="fact-item">
                  <span className="muted small">{t('Depth level (1–4): ', 'Mức độ tự động hoá (1–4): ')}</span>
                  <b>{a?.facts?.depth_level ? `L${a.facts.depth_level}` : '-'}</b>
                </div>
              </div>
            ))}

            {/* Arbiter note & Summary */}
            <div className="compare-cell label-cell sticky-col">
              <b>{t('Assessment', 'Đánh giá thẩm định')}</b>
            </div>
            {candidatesData.map(({ id, row, report: rep }) => (
              <div key={id} className="compare-cell">
                {row?.arbiter_note && (
                  <div className="arbiter-box">
                    <span className="small strong">{t('Head to head: ', `${term('Arbiter')}: `)}</span>
                    <span className="small">{backendText(lang, row.arbiter_note)}</span>
                  </div>
                )}
                {rep?.explanation && (
                  <p className="small mt-1">{rep.explanation}</p>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="modal-footer">
          <button className="primary" onClick={onClose}>{t('Close the comparison', 'Đóng bảng so sánh')}</button>
        </div>
      </div>
    </div>
  )
}
