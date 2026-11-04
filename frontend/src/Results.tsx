import { Fragment, useState } from 'react'
import { api } from './api'
import CompareModal from './CompareModal'
import { SecTag } from './CvPreview'
import {
  CRITERIA,
  type Assessment,
  type Card,
  type Config,
  type Evidence,
  type RankRow,
  type RankStatus,
  type RunState,
  type Values,
} from './types'

const pct = (x: number) => `${Math.round(x * 100)}%`
const TIER_BADGE = { strong: 'green', possible: 'sky', weak: 'amber' } as const
const TIER_NAME = { strong: 'Xuất sắc', possible: 'Tiềm năng', weak: 'Cần xem xét' } as const

const GROUPS: { status: Exclude<RankStatus, 'ranked'>; title: string; hint: string; open?: boolean }[] = [
  { status: 'needs_review', title: 'Cần HR rà soát thủ công', hint: 'Agent gặp điểm mơ hồ; khuyến nghị con người đọc lại', open: true },
  { status: 'not_shortlisted', title: 'Không lọt danh sách ngắn', hint: 'Vượt qua vòng loại trừ nhưng điểm tổng chưa đủ cao' },
  { status: 'knocked_out', title: 'Bị loại trừ (Knocked out)', hint: 'Thiếu công cụ bắt buộc hoặc chỉ liệt kê mơ hồ' },
]

export default function Results({
  run,
  config,
  onOpenCv,
}: {
  run: RunState
  config: Config | null
  onOpenCv: (id: string) => void
}) {
  const [sel, setSel] = useState<string | null>(null)
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([])
  const [isCompareOpen, setIsCompareOpen] = useState(false)
  const [activeTab, setActiveTab] = useState<'all' | 'ranked' | 'needs_review' | 'not_shortlisted' | 'knocked_out'>('all')
  const [decisions, setDecisions] = useState<Record<string, 'shortlist' | 'hold' | 'reject'>>({})

  const { ranking, report, assessments, weights } = run.values
  if (!ranking?.length) return null

  const labels = Object.fromEntries((config?.criteria ?? []).map((c) => [c.key, c.label]))
  const entry = report?.candidates.find((c) => c.cv_id === sel)

  const statusOf = (r: RankRow): RankStatus => r.status ?? (r.knocked_out ? 'knocked_out' : 'ranked')
  const ranked = ranking.filter((r) => statusOf(r) === 'ranked')
  const needsReview = ranking.filter((r) => statusOf(r) === 'needs_review')
  const notShortlisted = ranking.filter((r) => statusOf(r) === 'not_shortlisted')
  const knockedOut = ranking.filter((r) => statusOf(r) === 'knocked_out')

  const toggle = (id: string) => setSel(id === sel ? null : id)

  const toggleSelectCompare = (id: string, e: React.MouseEvent) => {
    e.stopPropagation()
    setSelectedForCompare((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id)
      if (prev.length >= 3) {
        alert('Bạn chỉ có thể chọn tối đa 3 ứng viên để so sánh trực diện.')
        return prev
      }
      return [...prev, id]
    })
  }

  const setDecision = (id: string, d: 'shortlist' | 'hold' | 'reject', e: React.MouseEvent) => {
    e.stopPropagation()
    setDecisions((prev) => ({ ...prev, [id]: prev[id] === d ? undefined! : d }))
  }

  const exportCsv = () => {
    if (!run.runId) return
    window.open(api.getExportCsvUrl(run.runId), '_blank')
  }

  const cvLink = (id: string) => (
    <button className="link mono" title="Xem bóc tách trích xuất" onClick={(e) => { e.stopPropagation(); onOpenCv(id) }}>
      {id}
    </button>
  )

  const selRow = ranking.find((r) => r.cv_id === sel)

  return (
    <section className="card results-card">
      <div className="results-top-bar">
        <div>
          <h2>
            Bảng xếp hạng & Đánh giá ứng viên{' '}
            {run.done && <span className="badge green">Đã hoàn tất</span>}
          </h2>
          {report && <p className="results-summary">{report.summary}</p>}
        </div>

        <div className="results-actions">
          {selectedForCompare.length >= 2 && (
            <button className="primary" onClick={() => setIsCompareOpen(true)}>
              ⚖️ So sánh đối đầu ({selectedForCompare.length})
            </button>
          )}
          {run.runId && (
            <button className="ghost" onClick={exportCsv} title="Tải file CSV có tiếng Việt mở bằng Excel">
              📥 Xuất file CSV (Excel)
            </button>
          )}
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="tab-pills">
        <button
          className={`pill-btn ${activeTab === 'all' ? 'active' : ''}`}
          onClick={() => setActiveTab('all')}
        >
          Tất cả ({ranking.length})
        </button>
        <button
          className={`pill-btn ${activeTab === 'ranked' ? 'active' : ''}`}
          onClick={() => setActiveTab('ranked')}
        >
          Đạt tiêu chuẩn ({ranked.length})
        </button>
        {needsReview.length > 0 && (
          <button
            className={`pill-btn ${activeTab === 'needs_review' ? 'active' : ''}`}
            onClick={() => setActiveTab('needs_review')}
          >
            Cần rà soát ({needsReview.length})
          </button>
        )}
        {notShortlisted.length > 0 && (
          <button
            className={`pill-btn ${activeTab === 'not_shortlisted' ? 'active' : ''}`}
            onClick={() => setActiveTab('not_shortlisted')}
          >
            Chưa vào top ({notShortlisted.length})
          </button>
        )}
        {knockedOut.length > 0 && (
          <button
            className={`pill-btn ${activeTab === 'knocked_out' ? 'active' : ''}`}
            onClick={() => setActiveTab('knocked_out')}
          >
            Bị loại ({knockedOut.length})
          </button>
        )}
      </div>

      {/* Main Ranking Table (when activeTab is 'all' or 'ranked') */}
      {(activeTab === 'all' || activeTab === 'ranked') && (
        <div className="table-wrapper">
          <table className="rank modern-rank-table">
            <thead>
              <tr>
                <th style={{ width: '40px' }}>So sánh</th>
                <th style={{ width: '50px' }}>Hạng</th>
                <th style={{ width: '100px' }}>Mã CV</th>
                <th>Tiêu đề / Chức danh</th>
                <th style={{ width: '160px' }}>Điểm phù hợp</th>
                <th>Căn cứ thẩm định (AI Arbiter)</th>
                <th style={{ width: '140px' }} className="text-right">Quyết định HR</th>
              </tr>
            </thead>
            <tbody>
              {ranked.map((r) => {
                const isSelected = selectedForCompare.includes(r.cv_id)
                const dec = decisions[r.cv_id]
                return (
                  <tr
                    key={r.cv_id}
                    className={`${r.cv_id === sel ? 'sel' : ''} ${dec ? `decided-${dec}` : ''}`}
                    onClick={() => toggle(r.cv_id)}
                  >
                    <td onClick={(e) => e.stopPropagation()} className="text-center">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        onChange={(e) => toggleSelectCompare(r.cv_id, e as any)}
                        title="Chọn để so sánh trực diện (tối đa 3 ứng viên)"
                      />
                    </td>
                    <td><span className="rank-badge">#{r.rank}</span></td>
                    <td className="nowrap">{cvLink(r.cv_id)}</td>
                    <td>
                      <div className="headline-text">{r.headline}</div>
                      <Flags f={r.flags} />
                    </td>
                    <td className="nowrap">
                      <div className="score-meter">
                        <div className="score-text">
                          <b className="mono">{pct(r.total)}</b>
                          {r.tier && (
                            <span className={`badge ${TIER_BADGE[r.tier]}`}>
                              {TIER_NAME[r.tier] || r.tier}
                            </span>
                          )}
                        </div>
                        <div className="meter-track">
                          <div className={`meter-fill tone-${r.tier || 'sky'}`} style={{ width: `${r.total * 100}%` }} />
                        </div>
                      </div>
                    </td>
                    <td className="small arbiter-cell">
                      {[r.reason, r.arbiter_note].filter(Boolean).join(' · ')}
                    </td>
                    <td className="text-right nowrap" onClick={(e) => e.stopPropagation()}>
                      <div className="decision-btn-group">
                        <button
                          className={`dec-btn shortlist ${dec === 'shortlist' ? 'active' : ''}`}
                          title="Đưa vào danh sách phỏng vấn"
                          onClick={(e) => setDecision(r.cv_id, 'shortlist', e)}
                        >
                          ✓ Duyệt
                        </button>
                        <button
                          className={`dec-btn hold ${dec === 'hold' ? 'active' : ''}`}
                          title="Cân nhắc thêm"
                          onClick={(e) => setDecision(r.cv_id, 'hold', e)}
                        >
                          ? Giữ
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
              {ranked.length === 0 && (
                <tr>
                  <td colSpan={7} className="muted text-center empty-cell">
                    Không có ứng viên nào vượt qua ngưỡng yêu cầu tối thiểu.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Subgroups (needs_review, not_shortlisted, knocked_out) */}
      {GROUPS.map((g) => {
        if (activeTab !== 'all' && activeTab !== g.status) return null
        const rows = ranking.filter((r) => statusOf(r) === g.status)
        if (!rows.length) return null
        return (
          <details key={g.status} className="group-box" open={activeTab === g.status || g.open}>
            <summary>
              <b>{g.title} ({rows.length})</b> <span className="muted small">{g.hint}</span>
            </summary>
            <table className="rank">
              <tbody>
                {rows.map((r) => (
                  <tr key={r.cv_id} className={r.cv_id === sel ? 'sel' : ''} onClick={() => toggle(r.cv_id)}>
                    <td style={{ width: '40px' }} onClick={(e) => e.stopPropagation()} className="text-center">
                      <input
                        type="checkbox"
                        checked={selectedForCompare.includes(r.cv_id)}
                        onChange={(e) => toggleSelectCompare(r.cv_id, e as any)}
                      />
                    </td>
                    <td className="nowrap" style={{ width: '100px' }}>{cvLink(r.cv_id)}</td>
                    <td>{r.headline} <Flags f={r.flags} /></td>
                    <td className="small why">{r.reason ?? '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        )
      })}

      <p className="muted small note-line">
        💡 <b>Mẹo:</b> Bấm vào một dòng ứng viên để xem phân tích chi tiết, hoặc đánh dấu checkbox để so sánh đối đầu 2-3 hồ sơ.
      </p>

      {/* Expanded Candidate Detail Card */}
      {sel && (
        <Detail
          key={sel}
          id={sel}
          a={assessments?.[sel]}
          weights={weights}
          labels={labels}
          reason={selRow?.reason ?? null}
          card={selRow?.card}
          onOpenCv={onOpenCv}
          explanation={entry?.explanation}
          strengths={entry?.strengths}
          gaps={entry?.gaps}
        />
      )}

      {/* Compare Modal */}
      {isCompareOpen && (
        <CompareModal
          selectedIds={selectedForCompare}
          run={run}
          config={config}
          onClose={() => setIsCompareOpen(false)}
          onOpenCv={onOpenCv}
        />
      )}
    </section>
  )
}

function Detail({
  id,
  a,
  weights,
  labels,
  reason,
  card,
  onOpenCv,
  explanation,
  strengths,
  gaps,
}: {
  id: string
  reason: string | null
  card?: Card | null
  onOpenCv: (id: string) => void
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
    try {
      setCv(await api.getCv(id))
    } catch (e) {
      setCv((e as Error).message)
    }
  }

  const review = a?.llm_review ?? {}
  const quotes = a
    ? [...CRITERIA.map((c) => a.criteria[c]), ...Object.values(review)].flatMap((s) =>
        s ? s.evidence.filter((e) => e.grounded).map((e) => e.quote) : []
      )
    : []

  return (
    <div className="detail modern-detail">
      <div className="detail-header">
        <div>
          <h3>Hồ sơ chi tiết: <span className="mono strong">{id}</span></h3>
        </div>
        <div className="detail-header-actions">
          <button className="ghost small-btn" onClick={showCv}>
            {cv ? 'Ẩn nội dung CV' : 'Xem nội dung CV'}
          </button>
          <button className="primary small-btn" onClick={() => onOpenCv(id)}>
            Bóc tách NLP
          </button>
        </div>
      </div>

      {a?.knocked_out && (
        <div className="banner rose">
          <b>Bị loại trừ (Knock-out):</b> {a.knockout_reason ?? reason ?? 'Không đáp ứng điều kiện tiên quyết'}
        </div>
      )}
      {!a?.knocked_out && reason && <p className="muted small">{reason}</p>}
      {a?.facts && <Facts f={a.facts} />}
      {explanation && <p className="lead-explanation">{explanation}</p>}

      {card && <CardView card={card} label={label} />}

      <div className="two strengths-gaps-row">
        {strengths && strengths.length > 0 && (
          <div className="card-box green-tint">
            <h4 className="text-emerald">✓ Điểm mạnh nổi bật</h4>
            <ul className="good">
              {strengths.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
        )}
        {gaps && gaps.length > 0 && (
          <div className="card-box amber-tint">
            <h4 className="text-amber">⚠ Điểm cần lưu ý / Còn thiếu</h4>
            <ul className="bad">
              {gaps.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {!a ? (
        <p className="muted">Chưa có dữ liệu đánh giá cho ứng viên này.</p>
      ) : (
        <div className="two radar-wrap">
          <Radar a={a} weights={weights} label={label} />
          <div>
            <h4 className="mb-2">Chi tiết từng tiêu chí tuyển dụng</h4>
            {CRITERIA.map((c) => {
              const s = a.criteria[c]
              return (
                <div key={c} className="crit">
                  <div className="wrow">
                    <span>{label(c).split(' (')[0]}</span>
                    <span className="bar">
                      <i style={{ width: `${s.score}%` }} />
                    </span>
                    <span className="mono strong">{Math.round(s.score)}/100</span>
                  </div>
                  <p className="small">{s.rationale}</p>
                  {s.evidence.map((e, i) => (
                    <Quote key={i} e={e} />
                  ))}
                  {review[c] && (
                    <div className="review-llm">
                      <p className="small">
                        <b>Thẩm định mở rộng (LLM review):</b> {review[c]!.rationale}
                      </p>
                      {review[c]!.evidence.map((e, i) => (
                        <Quote key={i} e={e} />
                      ))}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {cv === 'loading' && <p className="muted">Đang nạp văn bản hồ sơ...</p>}
      {typeof cv === 'string' && cv !== 'loading' && <p className="err">{cv}</p>}
      {cv && typeof cv === 'object' && (
        <div className="cv-preview-box">
          <h4>Văn bản hồ sơ đã che PII (Trích dẫn kiểm chứng được làm nổi bật)</h4>
          <pre className="cv">{highlight(cv.text, quotes)}</pre>
        </div>
      )}
    </div>
  )
}

function Quote({ e }: { e: Evidence }) {
  const kind =
    e.grounded === false
      ? 'u'
      : e.grounded === true && e.supported === false
      ? 'l'
      : e.grounded === true
      ? 'g'
      : 'x'
  const hint = {
    u: 'Không tìm thấy trích dẫn trong văn bản CV',
    l: 'Chỉ liệt kê kỹ năng, không có mô tả dự án thực tế',
    g: 'Được kiểm chứng chính xác từ kinh nghiệm thực chiến',
    x: 'Chưa kiểm chứng',
  }[kind]
  return (
    <blockquote className={`ev ${kind}`}>
      <b title={hint}>{{ u: '✗', l: '≈', g: '✓', x: '?' }[kind]}</b>
      <span className="q">{e.quote}</span> <SecTag section={e.section} />
      {kind === 'l' && <span className="ev-note">Chỉ liệt kê, thiếu chứng cứ dự án</span>}
    </blockquote>
  )
}

function Flags({ f }: { f?: string[] }) {
  if (!f?.length) return null
  return (
    <span className="badge amber" title="Cờ chất lượng dữ liệu">
      ⚠ {f.join(', ').replace(/_/g, ' ')}
    </span>
  )
}

function CardView({ card, label }: { card: Card; label: (k: string) => string }) {
  const name = (k: string) => label(k).split(' (')[0]
  return (
    <div className="card-x">
      {card.decomposition && (
        <table className="decomp">
          <thead>
            <tr>
              <th>Tiêu chí</th>
              <th>Trọng số</th>
              <th>Điểm</th>
              <th>Đóng góp điểm</th>
            </tr>
          </thead>
          <tbody>
            {card.decomposition.map((d) => (
              <tr key={d.criterion}>
                <td>{name(d.criterion)}</td>
                <td className="mono">{pct(d.weight)}</td>
                <td className="mono">{d.score}</td>
                <td className="mono">{(d.contribution * 100).toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {card.versus_next && (
        <p className="small">
          <b>Đứng trên {card.versus_next.vs} nhờ:</b>{' '}
          {card.versus_next.advantages
            .map((x) => `${name(x.criterion)} (+${(x.delta * 100).toFixed(1)})`)
            .join(', ') || 'độ tương đồng ngữ nghĩa'}
          {card.versus_next.disadvantages.length > 0 && (
            <>
              {' '}
              · <b>Kém hơn ở:</b>{' '}
              {card.versus_next.disadvantages
                .map((x) => `${name(x.criterion)} (${(x.delta * 100).toFixed(1)})`)
                .join(', ')}
            </>
          )}
        </p>
      )}
      {card.what_if && card.what_if.length > 0 && (
        <ul className="small what-if">
          {card.what_if.map((w) => (
            <li key={w.if_used.join()}>
              Nếu CV có thêm <b>{w.if_used.join(' + ')}</b> trong dự án thực tế:{' '}
              {w.passes_knockout ? (
                <>
                  điểm số đạt <b>{pct(w.total)}</b> ({w.delta >= 0 ? '+' : ''}
                  {(w.delta * 100).toFixed(1)} điểm)
                </>
              ) : (
                'vẫn bị loại'
              )}
              {w.must_have && w.passes_knockout && ' và sẽ vượt qua vòng knock-out'}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function Facts({ f }: { f: NonNullable<Assessment['facts']> }) {
  return (
    <p className="small facts-line">
      <b>Dữ liệu thực tế:</b> Cấp độ L{f.depth_level ?? '-'} · {f.years ?? '-'} năm tổng ·{' '}
      {f.years_automation ?? '-'} năm tự động hóa
    </p>
  )
}

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
    out.push(
      <Fragment key={`t${i}`}>{text.slice(pos, s)}</Fragment>,
      <mark key={`m${i}`}>{text.slice(s, e)}</mark>
    )
    pos = e
  })
  out.push(<Fragment key="end">{text.slice(pos)}</Fragment>)
  return out
}

function Radar({
  a,
  weights,
  label,
}: {
  a: Assessment
  weights: Values['weights'] | undefined
  label: (k: string) => string
}) {
  const C = 170,
    R = 105,
    n = CRITERIA.length
  const pt = (i: number, f: number, r = R) => {
    const t = (-90 + (360 / n) * i) * (Math.PI / 180)
    return [C + Math.cos(t) * r * f, C + Math.sin(t) * r * f] as const
  }
  const poly = (fs: number[]) =>
    fs.map((f, i) => pt(i, f).map((v) => v.toFixed(1)).join(',')).join(' ')
  const maxW = weights ? Math.max(...CRITERIA.map((c) => weights[c] ?? 0)) : 0
  return (
    <svg viewBox="0 0 340 340" className="radar" role="img" aria-label="Criterion scores radar">
      {[0.25, 0.5, 0.75, 1].map((f) => (
        <polygon key={f} points={poly(CRITERIA.map(() => f))} className="ring" />
      ))}
      {CRITERIA.map((c, i) => {
        const [x, y] = pt(i, 1),
          [lx, ly] = pt(i, 1, R + 18)
        return (
          <g key={c}>
            <line x1={C} y1={C} x2={x} y2={y} className="ring" />
            <text
              x={lx}
              y={ly + 3}
              textAnchor={lx < C - 5 ? 'end' : lx > C + 5 ? 'start' : 'middle'}
              className="rlabel"
            >
              {label(c).split(' (')[0].split(' / ')[0]}
            </text>
          </g>
        )
      })}
      {weights && maxW > 0 && (
        <polygon points={poly(CRITERIA.map((c) => (weights[c] ?? 0) / maxW))} className="poly-w" />
      )}
      <polygon points={poly(CRITERIA.map((c) => a.criteria[c].score / 100))} className="poly-s" />
      {CRITERIA.map((c, i) => {
        const [x, y] = pt(i, a.criteria[c].score / 100)
        return (
          <circle key={c} cx={x} cy={y} r={3} className="pt">
            <title>
              {label(c)}: {Math.round(a.criteria[c].score)}
            </title>
          </circle>
        )
      })}
      <g transform="translate(8 326)" className="rlabel">
        <rect width="10" height="3" y="-3" className="key-s" />
        <text x="14">Điểm số</text>
        <rect x="60" width="10" height="3" y="-3" className="key-w" />
        <text x="74">Trọng số JD</text>
      </g>
    </svg>
  )
}
