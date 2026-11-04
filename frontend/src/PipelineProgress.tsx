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

const STAGES: Stage[] = [
  { key: 'jd', label: 'Đọc JD', hint: 'Trích yêu cầu, trọng số theo loại công ty', tools: ['parse_jd'], nodes: ['ingest'] },
  { key: 'ask', label: 'Làm rõ với HR', hint: 'Chỉ khi JD mơ hồ', tools: ['ask_hr', 'propose_substitutes'], nodes: ['clarify', 'expand_query'], optional: true, hr: true },
  { key: 'screen', label: 'Sàng lọc kho CV', hint: 'Tìm kiếm lai, loại thiếu must-have, chấm 8 tiêu chí', tools: ['screen', 'update_requirements', 'set_weights', 'record_decisions'], nodes: ['search', 'rerank', 'score'] },
  { key: 'investigate', label: 'Điều tra CV lỗi', hint: 'CV scan, gõ sai tên công cụ', tools: ['investigate'], nodes: ['investigate'], optional: true },
  { key: 'rank', label: 'Xếp hạng bằng AI', hint: 'Đọc từng CV, hai lượt đảo thứ tự', tools: ['rank_shortlist'], nodes: [], optional: true },
  { key: 'verify', label: 'Kiểm chứng bằng chứng', hint: 'Trích dẫn phải có thật trong CV', tools: ['explain'], nodes: ['assess', 'verify_evidence'] },
  { key: 'compare', label: 'So sánh sát điểm', hint: 'Chỉ khi hai ứng viên gần bằng nhau', tools: ['compare'], nodes: ['arbitrate'], optional: true },
  { key: 'answer', label: 'Trả lời HR', hint: 'Đề xuất, lý do, câu hỏi phỏng vấn', tools: ['respond_to_hr'], nodes: ['report', 'finalize'], hr: true },
]

type Status = 'pending' | 'active' | 'done' | 'skipped' | 'waiting'
const ICON: Record<Status, string> = { pending: '○', active: '◐', done: '✓', skipped: '–', waiting: '⏸' }
const NAME: Record<Status, string> = { pending: 'chưa tới', active: 'đang làm', done: 'xong', skipped: 'không cần', waiting: 'chờ bạn' }

function statuses(run: RunState): Status[] {
  const ran = new Set(run.tools.map((t) => t.name))
  const running = (n: string) => run.nodes[n as keyof RunState['nodes']]?.status === 'running'
  const p6Done = (n: string) => run.mode === 'pipeline' && run.nodes[n as keyof RunState['nodes']]?.status === 'done'
  const waitingAsk = run.interrupt && run.interrupt.kind !== 'chat'
  const waitingChat = run.interrupt?.kind === 'chat'
  const out: Status[] = STAGES.map((s) => {
    if (s.key === 'ask' && waitingAsk) return 'waiting'
    if (s.key === 'answer' && waitingChat) return 'waiting'
    if (s.nodes.some(running)) return 'active'
    if (s.tools.some((t) => ran.has(t)) || s.nodes.some(p6Done)) return 'done'
    return 'pending'
  })
  // An optional stage the assistant passed over is "not needed", not "not reached".
  const last = Math.max(...out.map((st, i) => (st === 'done' || st === 'waiting' || st === 'active' ? i : -1)))
  return out.map((st, i) => (st === 'pending' && STAGES[i].optional && (i < last || run.done) ? 'skipped' : st))
}

function liveLine(run: RunState): string | null {
  if (run.interrupt?.kind === 'chat') return 'Trợ lý đã trả lời: xem đề xuất bên dưới, rồi nhắn tiếp hoặc duyệt.'
  if (run.interrupt) return 'Trợ lý đang chờ bạn trả lời câu hỏi bên dưới.'
  if (!run.running) return null
  const thought = [...run.logs].reverse().find((l) => l.node === 'agent')
  if (thought) return thought.message.split(' → ')[0].slice(0, 180)
  const last = run.logs[run.logs.length - 1]
  return last ? last.message.slice(0, 180) : 'Trợ lý đang bắt đầu…'
}

export default function PipelineProgress({ run }: { run: RunState }) {
  if (!run.runId && !run.running) return null
  const st = statuses(run)
  const count = (s: Stage) => run.tools.filter((t) => s.tools.includes(t.name)).length
  const refused = run.logs.filter((l) => l.node === 'guard' && l.message.startsWith('Refused')).length
  const line = liveLine(run)
  return (
    <section className="card pipeline-progress" aria-label="Tiến trình xử lý">
      <ol className="pp-stages">
        {STAGES.map((s, i) => (
          <li key={s.key} className={`pp-stage ${st[i]}`} title={s.hint}>
            <span className="pp-icon" aria-hidden>{ICON[st[i]]}</span>
            <span className="pp-label">{s.label}{count(s) > 1 && <span className="pp-count"> ×{count(s)}</span>}</span>
            <span className="pp-state">{NAME[st[i]]}</span>
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
          {run.tools.length} bước đã thực hiện
          {refused > 0 && <> · {refused} bước bị hệ thống kiểm soát chặn và trợ lý đã tự điều chỉnh</>}
          {run.done && <> · hoàn tất</>}
        </p>
      )}
    </section>
  )
}
