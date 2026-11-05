import { useEffect, useRef, useState } from 'react'
import AgentView from './AgentView'
import { api, isMock } from './api'
import CvLibrary from './CvLibrary'
import CvPool from './CvPool'
import CvPreview from './CvPreview'
import GraphView from './GraphView'
import { useLang, useT } from './i18n'
import PipelineProgress from './PipelineProgress'
import { InterruptPanel, Requirements, Timeline } from './Panels'
import Results from './Results'
import RunHistoryModal, { type HistoryItem } from './RunHistoryModal'
import RunPanel from './RunPanel'
import {
  AGENT_NODES,
  emptyRun,
  INTERRUPT_NODE,
  NODES,
  type Assessment,
  type Config,
  type LoopKey,
  type NodeName,
  type ResumeBody,
  type RunRequest,
  type LogLine,
  type RunState,
  type StreamEvent,
} from './types'

// Which loop edge a `level: "loop"` log belongs to, by emitting node.
const LOOP_OF: Partial<Record<NodeName, LoopKey>> = {
  ingest: 'clarify',
  clarify: 'clarify',
  search: 'expand',
  expand_query: 'expand',
  investigate: 'investigate',
  assess: 'verify',
  verify_evidence: 'verify',
  arbitrate: 'arbitrate',
  score: 'review',
  hr_review: 'review',
}

function applyEvent(s: RunState, ev: StreamEvent): RunState {
  switch (ev.type) {
    case 'run_started':
      return { ...s, runId: ev.run_id, traceUrl: ev.trace_url ?? null, mode: ev.mode ?? 'pipeline' }
    case 'tool':
      return { ...s, tools: [...s.tools, { name: ev.name, args: ev.args, result: ev.result }] }
    case 'node_start':
      return { ...s, interrupt: null, nodes: { ...s.nodes, [ev.node]: { status: 'running' } } }
    case 'node_end':
      return { ...s, nodes: { ...s.nodes, [ev.node]: { status: 'done', ms: ev.duration_ms } } }
    case 'log': {
      const loop = ev.level === 'loop' ? LOOP_OF[ev.node] : undefined
      const flash = loop ? { ...s.flash, [loop]: (s.flash[loop] ?? 0) + 1 } : s.flash
      return { ...s, flash, logs: [...s.logs, { node: ev.node, level: ev.level, message: ev.message }] }
    }
    case 'update': {
      const { assessments, loops, ...rest } = ev.update
      const values = { ...s.values, ...rest }
      if (loops) {
        const sum = { ...s.values.loops } as Record<string, number>
        for (const [k, n] of Object.entries(loops)) sum[k] = (sum[k] ?? 0) + (n ?? 0)
        values.loops = sum as typeof values.loops
      }
      if (assessments) {
        const merged: Record<string, Assessment> = { ...s.values.assessments }
        for (const [id, a] of Object.entries(assessments)) merged[id] = a
        values.assessments = merged
      }
      let expanded = s.expanded
      const prev = s.values.query_terms
      if (prev && rest.query_terms) {
        expanded = [...expanded, ...rest.query_terms.filter((t) => !prev.includes(t) && !expanded.includes(t))]
      }
      return { ...s, values, expanded }
    }
    case 'interrupt': {
      const node = s.mode === 'agent' && ev.kind !== 'chat' ? 'ask_hr' : INTERRUPT_NODE[ev.kind]
      const ms = s.nodes[node]?.ms
      const chat = ev.kind === 'chat' ? [...s.chat, { who: 'agent' as const, text: ev.payload.message }] : s.chat
      return {
        ...s,
        running: false,
        interrupt: { kind: ev.kind, payload: ev.payload } as RunState['interrupt'],
        nodes: { ...s.nodes, [node]: { status: 'waiting', ms } },
        chat,
      }
    }
    case 'done':
      return {
        ...s,
        running: false,
        done: true,
        interrupt: null,
        values: { ...s.values, report: ev.report },
        usage: ev.usage ?? null,
      }
    case 'error':
      return { ...s, running: false }
  }
}

// A run loaded from history has no live tool events: rebuild the tools the assistant ran from its narration
// ("… → screen({})"), minus the calls the guard refused, so the progress bar shows the run as it happened.
function toolsFromLogs(logs: LogLine[]): RunState['tools'] {
  const out: RunState['tools'] = []
  for (const l of logs) {
    if (l.node === 'agent') {
      const calls = l.message.includes(' → ') ? l.message.split(' → ').pop()! : l.message
      for (const m of calls.matchAll(/\b([a-z_]+)\(/g)) out.push({ name: m[1], args: {}, result: '' })
    } else if (l.node === 'guard' && l.message.startsWith('Refused ')) {
      const name = l.message.slice(8).split(':')[0]
      const i = out.map((t) => t.name).lastIndexOf(name)
      if (i >= 0) out.splice(i, 1)
    }
  }
  return out
}

export default function App() {
  const { t, term } = useT()
  const { lang, setLang } = useLang()
  const [tab, setTab] = useState<'screening' | 'library'>('screening')
  const [config, setConfig] = useState<Config | null>(null)
  const [run, setRun] = useState<RunState>(emptyRun)
  const [error, setError] = useState<string | null>(null)
  const [previewId, setPreviewId] = useState<string | null>(null)
  const [auditView, setAuditView] = useState(false)
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [history, setHistory] = useState<HistoryItem[]>(() => {
    try {
      return JSON.parse(localStorage.getItem('ts_run_history_v1') || '[]')
    } catch {
      return []
    }
  })
  const gen = useRef(0)

  useEffect(() => {
    api.config().then(setConfig).catch((e) => setError(`${t('Cannot load', 'Không nạp được')} /api/config: ${e.message}`))
    const shared = new URLSearchParams(location.search).get('run')
    if (shared) load(shared)
  }, [])

  const saveToHistory = (runId: string, title: string, topCandidate?: string, mode?: string) => {
    setHistory((prev) => {
      const filtered = prev.filter((h) => h.id !== runId)
      const updated = [
        {
          id: runId,
          timestamp: new Date().toISOString(),
          title: title || t('Hiring campaign', 'Chiến dịch tuyển dụng'),
          topCandidate,
          mode,
        },
        ...filtered,
      ].slice(0, 20)
      localStorage.setItem('ts_run_history_v1', JSON.stringify(updated))
      return updated
    })
  }

  const clearHistory = () => {
    localStorage.removeItem('ts_run_history_v1')
    setHistory([])
  }

  async function consume(stream: AsyncIterable<StreamEvent>, myGen: number) {
    try {
      for await (const ev of stream) {
        if (gen.current !== myGen) return
        if (ev.type === 'error') setError(ev.message)
        setRun((s) => {
          const nextState = applyEvent(s, ev)
          if (ev.type === 'done' && nextState.runId) {
            const top = nextState.values.ranking?.[0]?.cv_id
            saveToHistory(nextState.runId, nextState.values.requirements?.title || '', top, nextState.mode)
          }
          return nextState
        })
      }
    } catch (e) {
      if (gen.current === myGen) setError((e as Error).message)
    }
    if (gen.current === myGen) setRun((s) => (s.running ? { ...s, running: false } : s))
  }

  function start(req: RunRequest) {
    setError(null)
    setRun({ ...emptyRun, running: true, mode: req.mode ?? 'agent' })
    consume(api.start(req), ++gen.current)
  }

  function resume(body: ResumeBody) {
    if (!run.runId) return
    setError(null)
    setRun((s) => {
      const nodes = { ...s.nodes }
      for (const n of [...NODES, ...AGENT_NODES]) {
        if (nodes[n]?.status === 'waiting') nodes[n] = { status: 'running', ms: nodes[n]!.ms }
      }
      const chat = 'message' in body ? [...s.chat, { who: 'hr' as const, text: body.message }] : s.chat
      return { ...s, interrupt: null, running: true, nodes, chat }
    })
    consume(api.resume(run.runId, body), ++gen.current)
  }

  async function load(id: string) {
    if (!id.trim()) return
    setError(null)
    try {
      const snap = await api.getRun(id.trim())
      gen.current++
      const finished = snap.next.length === 0 && !snap.interrupt
      const nodes: RunState['nodes'] = {}
      const mode = snap.values.goal !== undefined ? 'agent' : 'pipeline'
      if (finished) for (const n of mode === 'agent' ? AGENT_NODES : NODES) nodes[n] = { status: 'done' }
      if (snap.interrupt) nodes[INTERRUPT_NODE[snap.interrupt.kind]] = { status: 'waiting' }
      const newRun: RunState = {
        ...emptyRun,
        runId: snap.run_id,
        values: snap.values,
        interrupt: snap.interrupt,
        nodes,
        done: finished,
        usage: snap.usage?.calls ? snap.usage : null,
        logs: snap.logs ?? [],
        tools: toolsFromLogs(snap.logs ?? []),
        mode,
      }
      setRun(newRun)
      saveToHistory(
        snap.run_id,
        snap.values.requirements?.title || '',
        snap.values.ranking?.[0]?.cv_id,
        mode
      )
    } catch (e) {
      setError(`${t('Cannot load the run', 'Không nạp được đợt chạy')}: ${(e as Error).message}`)
    }
  }

  return (
    <div className="app-shell">
      {/* Top Application Header */}
      <header className="app-header">
        <div className="brand-group">
          <div className="brand-logo">🎯</div>
          <div>
            <h1 className="brand-name">
              TalentScout AI{' '}
              {isMock && <span className="badge amber">Mock Demo</span>}
            </h1>
            <p className="brand-sub">{t('CV screening & candidate assessment for Automation Test roles',
              'Hệ thống sàng lọc & đánh giá ứng viên Automation Test')}</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="header-nav">
          <button
            className={`nav-tab ${tab === 'screening' ? 'active' : ''}`}
            onClick={() => setTab('screening')}
          >
            📋 {t('Screening', 'Chiến dịch sàng lọc')}
          </button>
          <button
            className={`nav-tab ${tab === 'library' ? 'active' : ''}`}
            onClick={() => setTab('library')}
          >
            📁 {t('CV library', 'Kho hồ sơ ứng viên')}
          </button>
        </nav>

        {/* Header Right Tools */}
        <div className="header-actions">
          <div className="lang-switch" role="group" aria-label={t('Language', 'Ngôn ngữ')}>
            {(['vi', 'en'] as const).map((l) => (
              <button
                key={l}
                type="button"
                className={`lang-btn ${lang === l ? 'active' : ''}`}
                aria-pressed={lang === l}
                title={l === 'vi' ? 'Tiếng Việt' : 'English'}
                onClick={() => setLang(l)}
              >
                {l.toUpperCase()}
              </button>
            ))}
          </div>

          <button
            className="ghost header-btn"
            onClick={() => setIsHistoryOpen(true)}
            title={t('Previous runs', 'Xem danh sách các đợt chạy trước')}
          >
            ⏱️ {t('History', 'Lịch sử')} ({history.length})
          </button>

          <button
            className={`ghost header-btn ${auditView ? 'audit-active' : ''}`}
            onClick={() => setAuditView(!auditView)}
            title={t('Show or hide the technical audit view and system log',
              'Bật/tắt chế độ kiểm chứng kỹ thuật & nhật ký hệ thống')}
          >
            {auditView ? `👁️ ${t('Audit view on', 'Đang xem kiểm chứng')}` : `🔍 ${t('Audit view', term('Audit'))}`}
          </button>

          {config?.langfuse_url && (
            <a
              href={config.langfuse_url}
              target="_blank"
              rel="noreferrer"
              className="ghost header-btn external-link"
              title={t('Open Langfuse traces', `Mở Langfuse ${term('Trace')}`)}
            >
              Langfuse ↗
            </a>
          )}
        </div>
      </header>

      {error && (
        <div className="banner" role="alert">
          <span>{error}</span>
          <button className="ghost small-btn" onClick={() => setError(null)}>{t('Close', 'Đóng')}</button>
        </div>
      )}

      {/* Main Content Area */}
      {tab === 'library' ? (
        <main className="library-container">
          <CvLibrary onOpenCv={setPreviewId} />
        </main>
      ) : (
        <div className="screening-workflow">
          <PipelineProgress run={run} />

          <div className="layout">
            <aside>
              <RunPanel config={config} running={run.running} onStart={start} />
              <CvPool onOpen={setPreviewId} />

              {run.runId && (
                <div className="run-info-card card">
                  <div className="flex-between">
                    <span className="small muted">{t('Run id:', 'Mã đợt chạy:')}</span>
                    <span className="mono strong">{run.runId}</span>
                  </div>
                  {run.usage && (
                    <div className="mt-1 small muted">
                      LLM: {run.usage.calls} {t('calls', 'lượt gọi')} · {run.usage.total_tokens.toLocaleString()} {t('tokens', 'token')}
                    </div>
                  )}
                </div>
              )}

              <Requirements run={run} />
            </aside>

            <main>
              {/* Technical Audit View toggleable */}
              {auditView && (
                <div className="audit-section">
                  <div className="audit-header">
                    <h4>🔍 {t('Execution graph (orchestrator)', `Sơ đồ quy trình thực thi (${term('Orchestrator')})`)}</h4>
                    <span className="badge slate">{run.mode === 'agent' ? 'P7 Autonomous Agent' : 'P6 Pipeline'}</span>
                  </div>
                  {run.mode === 'agent' ? <AgentView run={run} /> : <GraphView run={run} />}
                </div>
              )}

              {run.interrupt && (
                <InterruptPanel
                  interrupt={run.interrupt}
                  criteria={config?.criteria ?? []}
                  onSubmit={resume}
                />
              )}

              <Results run={run} config={config} onOpenCv={setPreviewId} />

              {auditView && <Timeline logs={run.logs} />}
            </main>
          </div>
        </div>
      )}

      {/* Modals */}
      {previewId && <CvPreview id={previewId} onClose={() => setPreviewId(null)} />}

      <RunHistoryModal
        isOpen={isHistoryOpen}
        currentRunId={run.runId}
        history={history}
        onClose={() => setIsHistoryOpen(false)}
        onSelectRun={load}
        onClearHistory={clearHistory}
      />
    </div>
  )
}
