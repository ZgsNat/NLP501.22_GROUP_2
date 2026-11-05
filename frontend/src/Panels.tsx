import { useEffect, useRef, useState } from 'react'
import { backendText, useT } from './i18n'
import { CRITERIA, type Interrupt, type SubstituteProposal, type LogLine, type ResumeBody, type RunState } from './types'

export function Timeline({ logs }: { logs: LogLine[] }) {
  const { t, lang } = useT()
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
      <h2>
        {t('Event timeline', 'Dòng thời gian sự kiện')}{' '}
        <span className="muted small">{t('why the agent did what it did',
          'vì sao agent làm như vậy · nhật ký kỹ thuật, ghi bằng tiếng Anh')}</span>
      </h2>
      <div className="timeline" ref={box}>
        {groups.length === 0 && <p className="muted">{t('No events yet.', 'Chưa có sự kiện nào.')}</p>}
        {groups.map((g, i) => (
          <div key={i} className="group">
            <div className="group-node mono">{g.node}</div>
            <ul>
              {g.lines.map((l, j) => (
                <li key={j} className={`log ${l.level}`}>
                  {l.level === 'loop' && <b>↺ {t('loop', 'vòng lặp')} </b>}
                  {l.level === 'warn' && <b>⚠ </b>}
                  {backendText(lang, l.message)}
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
  const { t, term, crit } = useT()
  const { requirements: r, weights, query_terms, policy_context } = run.values
  if (!r && !weights && !query_terms && !policy_context) return null
  const list = (items: string[]) => (items.length ? items.map((x) => <span key={x} className="badge gray">{x}</span>) : <span className="muted">-</span>)
  return (
    <section className="card">
      <h2>{t('Requirements', 'Yêu cầu tuyển dụng')}</h2>
      {r && (
        <dl>
          <dt>{t('Title', 'Vị trí')}</dt><dd>{r.title}</dd>
          <dt>{t('Must have', term('Must-have'))}</dt><dd className="chips">{list(r.must_have)}</dd>
          <dt>{t('Nice to have', term('Nice-to-have'))}</dt><dd className="chips">{list(r.nice_to_have)}</dd>
          <dt>{t('Languages', 'Ngôn ngữ lập trình')}</dt><dd className="chips">{list(r.languages)}</dd>
          <dt>{t('Test types', 'Loại kiểm thử')}</dt><dd className="chips">{list(r.test_types)}</dd>
          <dt>{t('Min years / depth', 'Số năm tối thiểu / mức độ')}</dt>
          <dd>{r.min_years ?? '-'} {t('yrs', 'năm')} / {t('level', 'cấp')} {r.depth_level ?? '-'}</dd>
          {r.ambiguities.length > 0 && <><dt>{t('Ambiguities', 'Điểm mơ hồ')}</dt><dd><ul>{r.ambiguities.map((a) => <li key={a}>{a}</li>)}</ul></dd></>}
        </dl>
      )}
      {weights && (
        <>
          <h3>{t('Weights', term('Weight'))}</h3>
          {CRITERIA.map((c) => (
            <div key={c} className="wrow">
              <span>{crit(c).split(' (')[0]}</span>
              <span className="bar"><i style={{ width: `${(weights[c] ?? 0) * 100}%` }} /></span>
              <span className="mono">{Math.round((weights[c] ?? 0) * 100)}%</span>
            </div>
          ))}
        </>
      )}
      {query_terms && (
        <>
          <h3>{t('Query terms', 'Từ khoá truy vấn')}{' '}
            <span className="muted small">{t('highlighted = added by expansion', 'tô màu = được thêm khi mở rộng truy vấn')}</span></h3>
          <div className="chips">
            {query_terms.map((q) => <span key={q} className={`badge ${run.expanded.includes(q) ? 'sky' : 'gray'}`}>{q}</span>)}
          </div>
        </>
      )}
      {policy_context && policy_context.length > 0 && (
        <>
          <h3>{t('Policy context (RAG)', 'Chính sách tuyển dụng được truy hồi (RAG)')}</h3>
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
  const { t, term } = useT()
  const title = {
    clarification: t('Clarify the hiring requirements', 'Làm rõ yêu cầu tuyển dụng (clarification)'),
    substitutes: t('Accept equivalent tools?', 'Chấp nhận công cụ thay thế tương đương?'),
    review: t('Review the ranking & tune the weights', `Rà soát bảng xếp hạng & tinh chỉnh ${term('Weight')}`),
    chat: t('The recruiting assistant replies', 'Trợ lý tuyển dụng phản hồi'),
  }
  return (
    <section className="card interrupt modern-interrupt">
      <div className="interrupt-header">
        <span className="badge amber">{t('HR input needed', 'Cần HR phản hồi')}</span>
        <h2>{title[interrupt.kind]}</h2>
      </div>
      {interrupt.kind === 'clarification' && <Clarify questions={interrupt.payload.questions} onSubmit={onSubmit} />}
      {interrupt.kind === 'substitutes' && <Substitutes proposals={interrupt.payload.proposals} onSubmit={onSubmit} />}
      {interrupt.kind === 'review' && <Review payload={interrupt.payload} criteria={criteria} onSubmit={onSubmit} />}
      {interrupt.kind === 'chat' && <Chat payload={interrupt.payload} onSubmit={onSubmit} />}
    </section>
  )
}

function Clarify({ questions, onSubmit }: { questions: string[]; onSubmit: (b: ResumeBody) => void }) {
  const { t, lang } = useT()
  const [answers, setAnswers] = useState<Record<string, string>>({})
  return (
    <form onSubmit={(e) => { e.preventDefault(); onSubmit({ answers }) }} className="interrupt-form">
      <p className="muted small mb-2">{t('The JD has points to confirm before the scoring criteria can be set precisely:',
        'JD có một số điểm cần xác nhận để thiết lập tiêu chí chấm điểm chính xác nhất:')}</p>
      {questions.map((q) => (
        <label key={q} className="form-question">
          {/* the question text stays the answer's key; only what HR reads is translated */}
          <span className="question-text">{backendText(lang, q)}</span>
          <input
            className="text-input"
            placeholder={t('Your answer or a specific requirement…', 'Nhập câu trả lời hoặc yêu cầu cụ thể…')}
            value={answers[q] ?? ''}
            onChange={(e) => setAnswers({ ...answers, [q]: e.target.value })}
          />
        </label>
      ))}
      <button className="primary mt-2" type="submit">{t('Send answers', 'Gửi phản hồi xác nhận')}</button>
    </form>
  )
}

// L2: too few CVs show a must-have, so HR decides which related tools may stand in for it.
function Substitutes({ proposals, onSubmit }: { proposals: SubstituteProposal[]; onSubmit: (b: ResumeBody) => void }) {
  const { t } = useT()
  const [accept, setAccept] = useState<Record<string, string[]>>(() => Object.fromEntries(proposals.map((p) => [p.must_have, p.related])))
  const flip = (m: string, tool: string) =>
    setAccept((a) => ({ ...a, [m]: a[m].includes(tool) ? a[m].filter((x) => x !== tool) : [...a[m], tool] }))
  return (
    <form onSubmit={(e) => { e.preventDefault(); onSubmit({ accept }) }} className="interrupt-form">
      {proposals.map((p) => (
        <fieldset key={p.must_have} className="subst modern-subst">
          <legend>
            {t(`Only ${p.covered} CV(s) show `, `Chỉ có ${p.covered} hồ sơ đáp ứng trực tiếp `)}<b>{p.must_have}</b>
            {t('. Accept these equivalent tools?', '. Bạn có đồng ý chấp nhận các công cụ tương đương sau không?')}
          </legend>
          <div className="chips">
            {p.related.map((tool) => (
              <label key={tool} className="check-chip">
                <input type="checkbox" checked={accept[p.must_have].includes(tool)} onChange={() => flip(p.must_have, tool)} />
                <span>{tool}</span>
              </label>
            ))}
          </div>
        </fieldset>
      ))}
      <p className="muted small">{t('Untick everything if candidates must have exactly the original tool.',
        'Bỏ chọn tất cả nếu ứng viên bắt buộc phải có đúng công cụ gốc.')}</p>
      <button className="primary mt-2" type="submit">{t('Continue the search', 'Tiếp tục tìm kiếm & mở rộng')}</button>
    </form>
  )
}

function Review({ payload, criteria, onSubmit }: {
  payload: Extract<Interrupt, { kind: 'review' }>['payload']
  criteria: { key: string; label: string }[]
  onSubmit: (b: ResumeBody) => void
}) {
  const { t, term, crit } = useT()
  const [w, setW] = useState<Record<string, number>>(() => Object.fromEntries(CRITERIA.map((c) => [c, payload.weights[c] ?? 0])))
  const [say, setSay] = useState('')
  const [dec, setDec] = useState<Record<string, { action: 'approve' | 'reject'; reason: string }>>({})
  const decisions = Object.entries(dec).filter(([, d]) => d.reason.trim()).map(([cv_id, d]) => ({ cv_id, ...d }))
  const mark = (id: string, action: 'approve' | 'reject') =>
    setDec((d) => (d[id]?.action === action ? (({ [id]: _, ...rest }) => rest)(d) : { ...d, [id]: { action, reason: d[id]?.reason ?? '' } }))
  const left = payload.adjustments_left
  const shown = payload.ranking.filter((r) => (r.status ?? (r.knocked_out ? 'knocked_out' : 'ranked')) === 'ranked')
  const sum = CRITERIA.reduce((a, c) => a + w[c], 0)
  const label = (k: string) => crit(k, criteria.find((c) => c.key === k)?.label)
  const adjust = () => onSubmit({
    action: 'adjust',
    weights: Object.fromEntries(CRITERIA.map((c) => [c, Math.round((w[c] / sum) * 10000) / 10000])),
  })
  const n = decisions.length
  return (
    <div className="review">
      <div>
        <h3>{t('Current ranking', 'Thứ hạng hiện tại')}</h3>
        <ol className="mini-rank">
          {shown.map((r) => (
            <li key={r.cv_id}>
              <span className="mono">{r.cv_id}</span> {r.headline} <b>{Math.round(r.total * 100)}%</b>{' '}
              <button className={`chip ${dec[r.cv_id]?.action === 'approve' ? 'on' : ''}`} title={t('Approve', 'Duyệt')}
                onClick={() => mark(r.cv_id, 'approve')}>✓</button>
              <button className={`chip ${dec[r.cv_id]?.action === 'reject' ? 'on' : ''}`} title={t('Reject', 'Loại')}
                onClick={() => mark(r.cv_id, 'reject')}>✗</button>
              {dec[r.cv_id] && (
                <input className="reason" placeholder={t('reason (required)', 'lý do (bắt buộc)')} value={dec[r.cv_id].reason}
                       onChange={(e) => setDec({ ...dec, [r.cv_id]: { ...dec[r.cv_id], reason: e.target.value } })} />
              )}
            </li>
          ))}
        </ol>
        <button className="ghost" disabled={!n || left === 0} onClick={() => onSubmit({ action: 'decide', decisions })}>
          {t(`Learn weights from my ${n || ''} decision${n === 1 ? '' : 's'}`,
            `Học trọng số từ ${n || ''} quyết định của tôi`)}
        </button>
        <p className="muted small">{t('Approve or reject candidates with a reason: the weights that best explain your choices re-score the pool, and every decision goes to the audit log.',
          'Duyệt hoặc loại ứng viên kèm lý do: bộ trọng số giải thích tốt nhất các lựa chọn của bạn sẽ được dùng để chấm lại toàn bộ, và mọi quyết định đều được ghi vào nhật ký kiểm chứng.')}</p>
        {payload.ranking.length > shown.length && (
          <p className="muted small">{t(`+ ${payload.ranking.length - shown.length} more (not shortlisted, needs review or knocked out): see Results.`,
            `+ ${payload.ranking.length - shown.length} hồ sơ khác (chưa vào shortlist, cần rà soát hoặc bị loại): xem phần Kết quả.`)}</p>
        )}
      </div>
      <div>
        <h3>{t('Weights', term('Weight'))}{' '}
          <span className="muted small">{t('normalised to 100% on submit', 'tự chuẩn hoá về 100% khi gửi')}</span></h3>
        {CRITERIA.map((c) => (
          <div key={c} className="wrow slider">
            <span title={label(c)}>{label(c).split(' (')[0]}</span>
            <input type="range" min={0} max={1} step={0.01} value={w[c]} onChange={(e) => setW({ ...w, [c]: +e.target.value })} />
            <span className="mono">{sum ? Math.round((w[c] / sum) * 100) : 0}%</span>
          </div>
        ))}
        <h3>{t('…or say it in words', '…hoặc mô tả bằng lời')}</h3>
        <textarea rows={2} value={say}
                  placeholder={t('e.g. Certification matters more for this bank client; ignore domain',
                    'VD: Với khách hàng ngân hàng này, chứng chỉ quan trọng hơn; bỏ qua nghiệp vụ')}
                  onChange={(e) => setSay(e.target.value)} />
        <div className="row">
          <button className="primary" onClick={() => onSubmit({ action: 'approve' })}>{t('Approve', 'Duyệt kết quả')}</button>
          <button className="ghost" disabled={sum <= 0 || left === 0} onClick={adjust}>
            {t('Re-score with new weights', 'Chấm lại theo trọng số mới')}</button>
          <button className="ghost" disabled={!say.trim() || left === 0}
                  onClick={() => onSubmit({ action: 'adjust', instruction: say.trim() })}>
            {t('Re-score from my words', 'Chấm lại theo mô tả của tôi')}</button>
        </div>
        {left !== undefined && <p className="muted small">{t(`${left} re-score round(s) left; each costs 0 new assessments.`,
          `Còn ${left} lượt chấm lại; mỗi lượt không tốn thêm lượt đánh giá LLM nào.`)}</p>}
      </div>
    </div>
  )
}

// P7: the agent's answer, and HR's next instruction in plain words (it may ask, re-weight, change requirements, decide).
function Chat({ payload, onSubmit }: { payload: Extract<Interrupt, { kind: 'chat' }>['payload']; onSubmit: (b: ResumeBody) => void }) {
  const { t } = useT()
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
                placeholder={t('e.g. Why is CV-012 below CV-030? · Make Docker a must-have · Reject CV-030, too junior',
                  'VD: Vì sao CV-012 xếp dưới CV-030? · Đưa Docker thành bắt buộc · Loại CV-030, kinh nghiệm còn ít')} />
      <div className="row">
        <button className="primary" type="submit" disabled={!say.trim() || payload.turns_left <= 0}>{t('Send to agent', 'Gửi cho agent')}</button>
        <button className="ghost" type="button" onClick={() => onSubmit({ action: 'approve' })}>{t('Approve and finish', 'Duyệt và kết thúc')}</button>
      </div>
      <p className="muted small">{t(`${payload.turns_left} message(s) left in this run.`, `Còn ${payload.turns_left} tin nhắn trong đợt chạy này.`)}</p>
    </form>
  )
}
