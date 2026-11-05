import AgentProposal from './AgentProposal'
import { Fragment, useState } from 'react'
import { api } from './api'
import CompareModal from './CompareModal'
import { SecTag } from './CvPreview'
import { backendText, useT, type T } from './i18n'
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

function tierName(t: T['t'], tier: 'strong' | 'possible' | 'weak'): string {
  return { strong: t('Strong', 'Xuất sắc'), possible: t('Possible', 'Tiềm năng'), weak: t('Weak', 'Cần xem xét') }[tier]
}

function groups({ t, term }: T): { status: Exclude<RankStatus, 'ranked'>; title: string; hint: string; open?: boolean }[] {
  return [
    { status: 'needs_review', title: t('Needs manual HR review', 'Cần HR rà soát thủ công'),
      hint: t('The agent hit an ambiguity; a person should read these', 'Agent gặp điểm mơ hồ; nên để người đọc lại'), open: true },
    { status: 'not_shortlisted', title: t('Not shortlisted', `Không vào ${term('Shortlist')}`),
      hint: t('Passed the knock-out, but the total is not high enough', 'Đã qua vòng loại trừ nhưng điểm tổng chưa đủ cao') },
    { status: 'knocked_out', title: t('Knocked out', term('Knock-out')),
      hint: t('A must-have tool is missing or only vaguely listed', 'Thiếu công cụ bắt buộc hoặc chỉ liệt kê mơ hồ') },
  ]
}

export default function Results({
  run,
  config,
  onOpenCv,
}: {
  run: RunState
  config: Config | null
  onOpenCv: (id: string) => void
}) {
  const tr = useT()
  const { t, term, lang } = tr
  const [sel, setSel] = useState<string | null>(null)
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([])
  const [isCompareOpen, setIsCompareOpen] = useState(false)
  const [activeTab, setActiveTab] = useState<'all' | 'ranked' | 'needs_review' | 'not_shortlisted' | 'knocked_out'>('all')
  const [decisions, setDecisions] = useState<Record<string, 'shortlist' | 'hold' | 'reject'>>({})

  const { ranking, report, assessments, weights } = run.values
  if (!ranking?.length) return null

  const labels = Object.fromEntries((config?.criteria ?? []).map((c) => [c.key, tr.crit(c.key, c.label)]))
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
        alert(t('You can compare at most 3 candidates side by side.', 'Bạn chỉ có thể chọn tối đa 3 ứng viên để so sánh trực diện.'))
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
    <button className="link mono" title={t('Open the parsed CV', 'Xem kết quả bóc tách CV')}
      onClick={(e) => { e.stopPropagation(); onOpenCv(id) }}>
      {id}
    </button>
  )

  const selRow = ranking.find((r) => r.cv_id === sel)

  return (
    <section className="card results-card">
      <div className="results-top-bar">
        <div>
          <h2>
            {t('Ranking & candidate assessment', 'Bảng xếp hạng & đánh giá ứng viên')}{' '}
            {run.done && <span className="badge green">{t('Done', 'Đã hoàn tất')}</span>}
          </h2>
          {report && !(run.mode === 'agent' && run.values.reply) && <p className="results-summary">{report.summary}</p>}
        </div>

        <div className="results-actions">
          {selectedForCompare.length >= 2 && (
            <button className="primary" onClick={() => setIsCompareOpen(true)}>
              ⚖️ {t('Compare side by side', 'So sánh đối đầu')} ({selectedForCompare.length})
            </button>
          )}
          {run.runId && (
            <button className="ghost" onClick={exportCsv}
              title={t('Download a CSV that opens in Excel', 'Tải file CSV (mở được bằng Excel, giữ dấu tiếng Việt)')}>
              📥 {t('Export CSV (Excel)', 'Xuất file CSV (Excel)')}
            </button>
          )}
        </div>
      </div>

      <AgentProposal run={run} labels={labels} onOpenCv={onOpenCv} selected={selectedForCompare}
        onToggleCompare={(id) => toggleSelectCompare(id, { stopPropagation() {} } as React.MouseEvent)} />

      {/* Filter Tabs */}
      <div className="tab-pills">
        <button className={`pill-btn ${activeTab === 'all' ? 'active' : ''}`} onClick={() => setActiveTab('all')}>
          {t('All', 'Tất cả')} ({ranking.length})
        </button>
        <button className={`pill-btn ${activeTab === 'ranked' ? 'active' : ''}`} onClick={() => setActiveTab('ranked')}>
          {t('Ranked', 'Đạt tiêu chuẩn')} ({ranked.length})
        </button>
        {needsReview.length > 0 && (
          <button className={`pill-btn ${activeTab === 'needs_review' ? 'active' : ''}`}
            onClick={() => setActiveTab('needs_review')}>
            {t('Needs review', 'Cần rà soát')} ({needsReview.length})
          </button>
        )}
        {notShortlisted.length > 0 && (
          <button className={`pill-btn ${activeTab === 'not_shortlisted' ? 'active' : ''}`}
            onClick={() => setActiveTab('not_shortlisted')}>
            {t('Not shortlisted', 'Chưa vào top')} ({notShortlisted.length})
          </button>
        )}
        {knockedOut.length > 0 && (
          <button className={`pill-btn ${activeTab === 'knocked_out' ? 'active' : ''}`}
            onClick={() => setActiveTab('knocked_out')}>
            {t('Knocked out', 'Bị loại')} ({knockedOut.length})
          </button>
        )}
      </div>

      {/* Main Ranking Table (when activeTab is 'all' or 'ranked') */}
      {(activeTab === 'all' || activeTab === 'ranked') && (
        <div className="table-wrapper">
          <table className="rank modern-rank-table">
            <thead>
              <tr>
                <th style={{ width: '40px' }}>{t('Compare', 'So sánh')}</th>
                <th style={{ width: '50px' }}>{t('Rank', 'Hạng')}</th>
                <th style={{ width: '100px' }}>{t('CV id', 'Mã CV')}</th>
                <th>{t('Headline / title', 'Tiêu đề / chức danh')}</th>
                <th style={{ width: '160px' }}>{t('Fit score', 'Điểm phù hợp')}</th>
                <th>{t('Grounds (AI arbiter)', `Căn cứ thẩm định (${term('Arbiter')})`)}</th>
                <th style={{ width: '140px' }} className="text-right">{t('HR decision', 'Quyết định của HR')}</th>
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
                        title={t('Select to compare (up to 3 candidates)', 'Chọn để so sánh trực diện (tối đa 3 ứng viên)')}
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
                          {r.tier && <span className={`badge ${TIER_BADGE[r.tier]}`}>{tierName(t, r.tier)}</span>}
                        </div>
                        <div className="meter-track">
                          <div className={`meter-fill tone-${r.tier || 'sky'}`} style={{ width: `${r.total * 100}%` }} />
                        </div>
                      </div>
                    </td>
                    <td className="small arbiter-cell">
                      {[r.reason, r.arbiter_note].filter(Boolean).map((x) => backendText(lang, x)).join(' · ')}
                    </td>
                    <td className="text-right nowrap" onClick={(e) => e.stopPropagation()}>
                      <div className="decision-btn-group">
                        <button
                          className={`dec-btn shortlist ${dec === 'shortlist' ? 'active' : ''}`}
                          title={t('Move to the interview list', 'Đưa vào danh sách phỏng vấn')}
                          onClick={(e) => setDecision(r.cv_id, 'shortlist', e)}
                        >
                          ✓ {t('Approve', 'Duyệt')}
                        </button>
                        <button
                          className={`dec-btn hold ${dec === 'hold' ? 'active' : ''}`}
                          title={t('Think it over', 'Cân nhắc thêm')}
                          onClick={(e) => setDecision(r.cv_id, 'hold', e)}
                        >
                          ? {t('Hold', 'Giữ')}
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
              {ranked.length === 0 && (
                <tr>
                  <td colSpan={7} className="muted text-center empty-cell">
                    {t('No candidate meets the minimum requirements.', 'Không có ứng viên nào vượt qua ngưỡng yêu cầu tối thiểu.')}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Subgroups (needs_review, not_shortlisted, knocked_out) */}
      {groups(tr).map((g) => {
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
                    <td className="small why">{r.reason ? backendText(lang, r.reason) : '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        )
      })}

      <p className="muted small note-line">
        💡 <b>{t('Tip:', 'Mẹo:')}</b>{' '}
        {t('Click a row for the detailed analysis, or tick 2–3 boxes to compare candidates side by side.',
          'Bấm vào một dòng ứng viên để xem phân tích chi tiết, hoặc đánh dấu 2–3 ô để so sánh đối đầu.')}
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
  const { t, term, lang, crit } = useT()
  const [cv, setCv] = useState<{ text: string } | 'loading' | string | null>(null)
  const label = (k: string) => labels[k] ?? crit(k)

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
          <h3>{t('Candidate detail:', 'Hồ sơ chi tiết:')} <span className="mono strong">{id}</span></h3>
        </div>
        <div className="detail-header-actions">
          <button className="ghost small-btn" onClick={showCv}>
            {cv ? t('Hide CV text', 'Ẩn nội dung CV') : t('Show CV text', 'Xem nội dung CV')}
          </button>
          <button className="primary small-btn" onClick={() => onOpenCv(id)}>
            {t('NLP parse', `Bóc tách ${term('NLP')}`)}
          </button>
        </div>
      </div>

      {a?.knocked_out && (
        <div className="banner rose">
          <b>{t('Knocked out:', `${term('Knock-out')}:`)}</b>{' '}
          {backendText(lang, a.knockout_reason ?? reason) || t('Does not meet a prerequisite', 'Không đáp ứng điều kiện tiên quyết')}
        </div>
      )}
      {!a?.knocked_out && reason && <p className="muted small">{backendText(lang, reason)}</p>}
      {a?.facts && <Facts f={a.facts} />}
      {explanation && <p className="lead-explanation">{explanation}</p>}

      {card && <CardView card={card} label={label} />}

      <div className="two strengths-gaps-row">
        {strengths && strengths.length > 0 && (
          <div className="card-box green-tint">
            <h4 className="text-emerald">✓ {t('Key strengths', 'Điểm mạnh nổi bật')}</h4>
            <ul className="good">
              {strengths.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
        )}
        {gaps && gaps.length > 0 && (
          <div className="card-box amber-tint">
            <h4 className="text-amber">⚠ {t('Gaps to probe', 'Điểm cần lưu ý / còn thiếu')}</h4>
            <ul className="bad">
              {gaps.map((s) => (
                <li key={s}>{s}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {!a ? (
        <p className="muted">{t('No assessment for this candidate yet.', 'Chưa có dữ liệu đánh giá cho ứng viên này.')}</p>
      ) : (
        <div className="two radar-wrap">
          <Radar a={a} weights={weights} label={label} />
          <div>
            <h4 className="mb-2">{t('Criterion by criterion', 'Chi tiết từng tiêu chí tuyển dụng')}</h4>
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
                  <p className="small">{backendText(lang, s.rationale)}</p>
                  {s.evidence.map((e, i) => (
                    <Quote key={i} e={e} />
                  ))}
                  {review[c] && (
                    <div className="review-llm">
                      <p className="small">
                        <b>{t('LLM review:', `Thẩm định bổ sung (${term('LLM')} review):`)}</b> {backendText(lang, review[c]!.rationale)}
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

      {cv === 'loading' && <p className="muted">{t('Loading the CV text…', 'Đang nạp văn bản hồ sơ…')}</p>}
      {typeof cv === 'string' && cv !== 'loading' && <p className="err">{cv}</p>}
      {cv && typeof cv === 'object' && (
        <div className="cv-preview-box">
          <h4>{t('Redacted CV text (verified quotes highlighted)',
            'Văn bản CV đã che thông tin cá nhân (các trích dẫn đã kiểm chứng được làm nổi bật)')}</h4>
          <pre className="cv">{highlight(cv.text, quotes)}</pre>
        </div>
      )}
    </div>
  )
}

function Quote({ e }: { e: Evidence }) {
  const { t } = useT()
  const kind =
    e.grounded === false
      ? 'u'
      : e.grounded === true && e.supported === false
      ? 'l'
      : e.grounded === true
      ? 'g'
      : 'x'
  const hint = {
    u: t('Quote not found in the CV text', 'Không tìm thấy trích dẫn trong văn bản CV'),
    l: t('Only listed as a skill, no project shows it', 'Chỉ liệt kê kỹ năng, không có dự án thực tế chứng minh'),
    g: t('Verified against real work in the CV', 'Đã kiểm chứng với kinh nghiệm làm việc thực tế trong CV'),
    x: t('Not verified', 'Chưa kiểm chứng'),
  }[kind]
  return (
    <blockquote className={`ev ${kind}`}>
      <b title={hint}>{{ u: '✗', l: '≈', g: '✓', x: '?' }[kind]}</b>
      <span className="q">{e.quote}</span> <SecTag section={e.section} />
      {kind === 'l' && <span className="ev-note">{t('Listed only, no project evidence', 'Chỉ liệt kê, thiếu bằng chứng dự án')}</span>}
    </blockquote>
  )
}

function Flags({ f }: { f?: string[] }) {
  const { t, flag } = useT()
  if (!f?.length) return null
  return (
    <span className="badge amber" title={t('Data-quality flags', 'Cờ chất lượng dữ liệu')}>
      ⚠ {f.map(flag).join(', ')}
    </span>
  )
}

function CardView({ card, label }: { card: Card; label: (k: string) => string }) {
  const { t, term } = useT()
  const name = (k: string) => label(k).split(' (')[0]
  return (
    <div className="card-x">
      {card.decomposition && (
        <table className="decomp">
          <thead>
            <tr>
              <th>{t('Criterion', 'Tiêu chí')}</th>
              <th>{t('Weight', term('Weight'))}</th>
              <th>{t('Score', 'Điểm')}</th>
              <th>{t('Contribution', 'Đóng góp vào điểm tổng')}</th>
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
          <b>{t(`Above ${card.versus_next.vs} thanks to:`, `Đứng trên ${card.versus_next.vs} nhờ:`)}</b>{' '}
          {card.versus_next.advantages
            .map((x) => `${name(x.criterion)} (+${(x.delta * 100).toFixed(1)})`)
            .join(', ') || t('passage relevance', 'độ liên quan ngữ nghĩa của các đoạn CV')}
          {card.versus_next.disadvantages.length > 0 && (
            <>
              {' '}
              · <b>{t('Weaker on:', 'Kém hơn ở:')}</b>{' '}
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
              {t('If the CV showed ', `${term('What-if')}: nếu CV có thêm `)}<b>{w.if_used.join(' + ')}</b>
              {t(' in real project work: ', ' trong dự án thực tế: ')}
              {w.passes_knockout ? (
                <>
                  {t('the score would be ', 'điểm sẽ đạt ')}<b>{pct(w.total)}</b> ({w.delta >= 0 ? '+' : ''}
                  {(w.delta * 100).toFixed(1)} {t('points', 'điểm')})
                </>
              ) : (
                t('still knocked out', 'vẫn bị loại')
              )}
              {w.must_have && w.passes_knockout && t(' and it would pass the knock-out', ' và sẽ vượt qua vòng knock-out (loại trừ)')}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function Facts({ f }: { f: NonNullable<Assessment['facts']> }) {
  const { t } = useT()
  return (
    <p className="small facts-line">
      <b>{t('Parsed facts:', 'Dữ liệu trích xuất từ CV:')}</b> {t('Depth', 'Mức độ')} L{f.depth_level ?? '-'} ·{' '}
      {f.years ?? '-'} {t('years total', 'năm kinh nghiệm')} · {f.years_automation ?? '-'} {t('years in automation', 'năm làm tự động hoá')}
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
  const { t } = useT()
  const C = 170,
    R = 105,
    n = CRITERIA.length
  const pt = (i: number, f: number, r = R) => {
    const th = (-90 + (360 / n) * i) * (Math.PI / 180)
    return [C + Math.cos(th) * r * f, C + Math.sin(th) * r * f] as const
  }
  const poly = (fs: number[]) =>
    fs.map((f, i) => pt(i, f).map((v) => v.toFixed(1)).join(',')).join(' ')
  const maxW = weights ? Math.max(...CRITERIA.map((c) => weights[c] ?? 0)) : 0
  return (
    <svg viewBox="0 0 340 340" className="radar" role="img" aria-label={t('Criterion scores radar', 'Biểu đồ radar điểm theo tiêu chí')}>
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
        <text x="14">{t('Score', 'Điểm số')}</text>
        <rect x="60" width="10" height="3" y="-3" className="key-w" />
        <text x="74">{t('JD weight', 'Trọng số JD')}</text>
      </g>
    </svg>
  )
}
