import { CRITERIA, type Config, type Criterion, type RunState } from './types'

interface Props {
  selectedIds: string[]
  run: RunState
  config?: Config | null
  onClose: () => void
  onOpenCv: (id: string) => void
}

const CRITERIA_LABELS: Record<Criterion, string> = {
  tools: 'Công cụ tự động hóa (Tools)',
  automation_depth: 'Mức độ chuyên sâu (Depth)',
  programming_language: 'Ngôn ngữ lập trình (Language)',
  test_types: 'Loại kiểm thử (Web/API/Mobile)',
  cicd_infra: 'Hạ tầng CI/CD & Pipeline',
  experience_level: 'Số năm kinh nghiệm (Experience)',
  istqb: 'Chứng chỉ chuyên môn (ISTQB)',
  domain_language: 'Ngoại ngữ & Ngữ cảnh ngành',
}

const TIER_LABELS: Record<string, { label: string; tone: string }> = {
  strong: { label: 'Xuất sắc', tone: 'green' },
  possible: { label: 'Tiềm năng', tone: 'sky' },
  weak: { label: 'Yếu / Chưa đạt', tone: 'amber' },
}

export default function CompareModal({ selectedIds, run, onClose, onOpenCv }: Props) {
  if (selectedIds.length === 0) return null

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
            <h3>So sánh trực diện ứng viên ({selectedIds.length} ứng viên)</h3>
            <p className="muted small">Đối chiếu chi tiết điểm số tiêu chuẩn, bằng chứng kinh nghiệm và đánh giá của hội đồng AI.</p>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <div className="compare-grid-container">
          <div className="compare-grid" style={{ gridTemplateColumns: `200px repeat(${selectedIds.length}, minmax(260px, 1fr))` }}>
            {/* Header row: Candidate basic info */}
            <div className="compare-cell label-cell sticky-col">
              <b>Thông tin ứng viên</b>
            </div>
            {candidatesData.map(({ id, row }) => (
              <div key={id} className="compare-cell candidate-head">
                <div className="flex-between">
                  <button className="link mono strong" onClick={() => onOpenCv(id)}>{id}</button>
                  {row?.tier && (
                    <span className={`badge ${TIER_LABELS[row.tier]?.tone || 'gray'}`}>
                      {TIER_LABELS[row.tier]?.label || row.tier}
                    </span>
                  )}
                </div>
                <div className="candidate-headline">{row?.headline || '-'}</div>
                <div className="candidate-score">
                  <span className="score-num">{Math.round((row?.total || 0) * 100)}%</span>
                  <span className="muted small">Độ phù hợp tổng thể</span>
                </div>
              </div>
            ))}

            {/* Strengths & Gaps */}
            <div className="compare-cell label-cell sticky-col">
              <b>Điểm mạnh & Điểm khuyết</b>
            </div>
            {candidatesData.map(({ id, report: rep }) => (
              <div key={id} className="compare-cell">
                <div className="tag-section">
                  <span className="small muted">Thế mạnh chính:</span>
                  <div className="chips">
                    {(rep?.strengths || []).map((s) => (
                      <span key={s} className="badge emerald">{s}</span>
                    ))}
                    {!rep?.strengths?.length && <span className="muted small">-</span>}
                  </div>
                </div>
                <div className="tag-section mt-2">
                  <span className="small muted">Điểm còn thiếu:</span>
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
                    <div className="criterion-title">{CRITERIA_LABELS[criterion] || criterion}</div>
                    <div className="muted small">Trọng số: {weightPct}%</div>
                  </div>
                  {candidatesData.map(({ id, assessment: a }) => {
                    const cScore = a?.criteria[criterion]
                    const score = cScore?.score ?? 0
                    const rationale = cScore?.rationale || '-'
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
              <b>Kinh nghiệm thực tế</b>
            </div>
            {candidatesData.map(({ id, assessment: a }) => (
              <div key={id} className="compare-cell">
                <div className="fact-item">
                  <span className="muted small">Số năm tự động hóa: </span>
                  <b>{a?.facts?.years_automation != null ? `${a.facts.years_automation} năm` : 'Không xác định'}</b>
                </div>
                <div className="fact-item">
                  <span className="muted small">Cấp độ chuyên sâu (1-4): </span>
                  <b>{a?.facts?.depth_level ? `Cấp độ ${a.facts.depth_level}` : '-'}</b>
                </div>
              </div>
            ))}

            {/* Arbiter note & Summary */}
            <div className="compare-cell label-cell sticky-col">
              <b>Đánh giá thẩm định</b>
            </div>
            {candidatesData.map(({ id, row, report: rep }) => (
              <div key={id} className="compare-cell">
                {row?.arbiter_note && (
                  <div className="arbiter-box">
                    <span className="small strong">So sánh đối đầu: </span>
                    <span className="small">{row.arbiter_note}</span>
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
          <button className="primary" onClick={onClose}>Đóng bảng so sánh</button>
        </div>
      </div>
    </div>
  )
}
