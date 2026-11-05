import { backendText, useT, type T } from './i18n'
import type { RunState } from './types'

// The run as HR sees it: what the assistant has done, is doing, skipped, or is waiting for.
// A stage is recognised from the P7 tools that ran (run.tools) and from the pipeline nodes running inside them
// (node_start / node_end events), so the same bar also follows a P6 run.
interface Stage {
  key: string
  label: string
  hint: string
  tools: string[] // P7 tools that complete this stage
  nodes: string[] // nodes that run while this stage is in progress (inside a tool, or in P6)
  optional?: boolean // the assistant may decide the stage is not needed
  hr?: boolean // waits for HR
}

function stages({ t, term }: T): Stage[] {
  return [
    { key: 'jd', label: t('Read the JD', 'Đọc JD'), hint: t('Requirements, weights for the company type', 'Trích yêu cầu, trọng số theo loại công ty'),
      tools: ['parse_jd'], nodes: ['ingest'] },
    { key: 'ask', label: t('Clarify with HR', 'Làm rõ với HR'), hint: t('Only when the JD is ambiguous', 'Chỉ khi JD mơ hồ'),
      tools: ['ask_hr', 'propose_substitutes'], nodes: ['clarify', 'expand_query'], optional: true, hr: true },
    { key: 'screen', label: t('Screen the CV pool', 'Sàng lọc kho CV'),
      hint: t('Hybrid search, must-have knock-out, 8 rule criteria', `Tìm kiếm lai, ${term('Knock-out')} thiếu must-have, chấm 8 tiêu chí`),
      tools: ['screen', 'update_requirements', 'set_weights', 'record_decisions'], nodes: ['search', 'rerank', 'score'] },
    { key: 'investigate', label: t('Investigate odd CVs', 'Điều tra CV lỗi'), hint: t('Scans, misspelt tool names', 'CV scan, gõ sai tên công cụ'),
      tools: ['investigate'], nodes: ['investigate'], optional: true },
    { key: 'rank', label: t('AI ranking', 'Xếp hạng bằng AI'), hint: t('Reads each CV, two passes in opposite orders', 'Đọc từng CV, hai lượt đảo thứ tự'),
      tools: ['rank_shortlist'], nodes: [], optional: true },
    { key: 'verify', label: t('Verify evidence', 'Kiểm chứng bằng chứng'), hint: t('Every quote must be in the CV', 'Trích dẫn phải có thật trong CV'),
      tools: ['explain'], nodes: ['assess', 'verify_evidence'] },
    { key: 'compare', label: t('Compare near-ties', 'So sánh sát điểm'), hint: t('Only when two candidates are nearly level', 'Chỉ khi hai ứng viên gần bằng điểm'),
      tools: ['compare'], nodes: ['arbitrate'], optional: true },
    { key: 'answer', label: t('Answer HR', 'Trả lời HR'), hint: t('Proposal, reasons, interview questions', 'Đề xuất, lý do, câu hỏi phỏng vấn'),
      tools: ['respond_to_hr'], nodes: ['report', 'finalize'], hr: true },
  ]
}

type Status = 'pending' | 'active' | 'done' | 'skipped' | 'waiting'
const ICON: Record<Status, string> = { pending: '○', active: '◐', done: '✓', skipped: '–', waiting: '⏸' }

function statuses(run: RunState, list: Stage[]): Status[] {
  const ran = new Set(run.tools.map((x) => x.name))
  const running = (n: string) => run.nodes[n as keyof RunState['nodes']]?.status === 'running'
  const p6Done = (n: string) => run.mode === 'pipeline' && run.nodes[n as keyof RunState['nodes']]?.status === 'done'
  const waitingAsk = run.interrupt && run.interrupt.kind !== 'chat'
  const waitingChat = run.interrupt?.kind === 'chat'
  const out: Status[] = list.map((s) => {
    if (s.key === 'ask' && waitingAsk) return 'waiting'
    if (s.key === 'answer' && waitingChat) return 'waiting'
    if (s.nodes.some(running)) return 'active'
    if (s.tools.some((x) => ran.has(x)) || s.nodes.some(p6Done)) return 'done'
    return 'pending'
  })
  // An optional stage the assistant passed over is "not needed", not "not reached".
  const last = Math.max(...out.map((st, i) => (st === 'done' || st === 'waiting' || st === 'active' ? i : -1)))
  return out.map((st, i) => (st === 'pending' && list[i].optional && (i < last || run.done) ? 'skipped' : st))
}

function liveLine(run: RunState, { t, lang }: T): string | null {
  if (run.interrupt?.kind === 'chat') return t('The assistant replied: see the proposal below, then reply or approve.',
    'Trợ lý đã trả lời: xem đề xuất bên dưới, rồi nhắn tiếp hoặc duyệt.')
  if (run.interrupt) return t('The assistant is waiting for your answer below.', 'Trợ lý đang chờ bạn trả lời câu hỏi bên dưới.')
  if (!run.running) return null
  const thought = [...run.logs].reverse().find((l) => l.node === 'agent')
  if (thought) return backendText(lang, thought.message.split(' → ')[0]).slice(0, 180)
  const last = run.logs[run.logs.length - 1]
  return last ? backendText(lang, last.message).slice(0, 180) : t('The assistant is starting…', 'Trợ lý đang bắt đầu…')
}

export default function PipelineProgress({ run }: { run: RunState }) {
  const tr = useT()
  const { t } = tr
  if (!run.runId && !run.running) return null
  const list = stages(tr)
  const name: Record<Status, string> = {
    pending: t('not yet', 'chưa tới'), active: t('in progress', 'đang làm'), done: t('done', 'xong'),
    skipped: t('not needed', 'không cần'), waiting: t('waiting for you', 'chờ bạn'),
  }
  const st = statuses(run, list)
  const count = (s: Stage) => run.tools.filter((x) => s.tools.includes(x.name)).length
  const refused = run.logs.filter((l) => l.node === 'guard' && l.message.startsWith('Refused')).length
  const line = liveLine(run, tr)
  return (
    <section className="card pipeline-progress" aria-label={t('Progress', 'Tiến trình xử lý')}>
      <ol className="pp-stages">
        {list.map((s, i) => (
          <li key={s.key} className={`pp-stage ${st[i]}`} title={s.hint}>
            <span className="pp-icon" aria-hidden>{ICON[st[i]]}</span>
            <span className="pp-label">{s.label}{count(s) > 1 && <span className="pp-count"> ×{count(s)}</span>}</span>
            <span className="pp-state">{name[st[i]]}</span>
          </li>
        ))}
      </ol>
      {line && (
        <p className={`pp-live ${run.interrupt ? 'waiting' : ''}`}>
          {run.running && !run.interrupt && <span className="spinner small" />}
          {line}
        </p>
      )}
      {(run.tools.length > 0 || refused > 0) && (
        <p className="muted small pp-meta">
          {t(`${run.tools.length} steps run`, `${run.tools.length} bước đã thực hiện`)}
          {refused > 0 && <> · {t(`${refused} step(s) refused by the guard, and the assistant adjusted`,
            `${refused} bước bị lớp kiểm soát (guard) chặn và trợ lý đã tự điều chỉnh`)}</>}
          {run.done && <> · {t('finished', 'hoàn tất')}</>}
        </p>
      )}
    </section>
  )
}
