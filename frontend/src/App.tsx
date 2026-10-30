import { useEffect, useRef, useState } from 'react'
import { api, isMock } from './api'
import CvPool from './CvPool'
import CvPreview from './CvPreview'
import GraphView from './GraphView'
import { InterruptPanel, Requirements, Timeline } from './Panels'
import Results from './Results'
import RunPanel from './RunPanel'
import {
  emptyRun, INTERRUPT_NODE, NODES,
  type Assessment, type Config, type LoopKey, type NodeName, type ResumeBody,
  type RunRequest, type RunState, type StreamEvent,
} from './types'

// Which loop edge a `level: "loop"` log belongs to, by emitting node.
const LOOP_OF: Partial<Record<NodeName, LoopKey>> = {
  ingest: 'clarify', clarify: 'clarify', search: 'expand', expand_query: 'expand', investigate: 'investigate', assess: 'verify',
  verify_evidence: 'verify', arbitrate: 'arbitrate', score: 'review', hr_review: 'review',
}

// Fold one stream event into the run state.
function applyEvent(s: RunState, ev: StreamEvent): RunState {
  switch (ev.type) {
    case 'run_started':
      return { ...s, runId: ev.run_id, traceUrl: ev.trace_url ?? null }
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
      const { assessments, ...rest } = ev.update
      const values = { ...s.values, ...rest }
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
      const node = INTERRUPT_NODE[ev.kind]
      const ms = s.nodes[node]?.ms
      return { ...s, running: false, interrupt: { kind: ev.kind, payload: ev.payload } as RunState['interrupt'],
        nodes: { ...s.nodes, [node]: { status: 'waiting', ms } } }
    }
    case 'done':
      return { ...s, running: false, done: true, interrupt: null, values: { ...s.values, report: ev.report }, usage: ev.usage ?? null }
    case 'error':
      return { ...s, running: false }
  }
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null)
  const [run, setRun] = useState<RunState>(emptyRun)
  const [error, setError] = useState<string | null>(null)
  const [loadId, setLoadId] = useState('')
  const [previewId, setPreviewId] = useState<string | null>(null) // CV shown in the parse-preview modal
  const gen = useRef(0) // ignore events from a stream that has been superseded

  useEffect(() => {
    api.config().then(setConfig).catch((e) => setError(`Cannot load /api/config: ${e.message}`))
    const shared = new URLSearchParams(location.search).get('run') // ?run=<id> → shareable link to a run
    if (shared) { setLoadId(shared); load(shared) }
  }, [])

  async function consume(stream: AsyncIterable<StreamEvent>, myGen: number) {
    try {
      for await (const ev of stream) {
        if (gen.current !== myGen) return
        if (ev.type === 'error') setError(ev.message)
        setRun((s) => applyEvent(s, ev))
      }
    } catch (e) {
      if (gen.current === myGen) setError((e as Error).message)
    }
    if (gen.current === myGen) setRun((s) => (s.running ? { ...s, running: false } : s))
  }

  function start(req: RunRequest) {
    setError(null)
    setRun({ ...emptyRun, running: true })
    consume(api.start(req), ++gen.current)
  }

  function resume(body: ResumeBody) {
    if (!run.runId) return
    setError(null)
    setRun((s) => {
      const nodes = { ...s.nodes }
      for (const n of NODES) if (nodes[n]?.status === 'waiting') nodes[n] = { status: 'running', ms: nodes[n]!.ms }
      return { ...s, interrupt: null, running: true, nodes }
    })
    consume(api.resume(run.runId, body), ++gen.current)
  }

  async function load(id = loadId) {
    if (!id.trim()) return
    setError(null)
    try {
      const snap = await api.getRun(id.trim())
      gen.current++
      const finished = snap.next.length === 0 && !snap.interrupt
      const nodes: RunState['nodes'] = {}
      if (finished) for (const n of NODES) nodes[n] = { status: 'done' }
      if (snap.interrupt) nodes[INTERRUPT_NODE[snap.interrupt.kind]] = { status: 'waiting' }
      setRun({ ...emptyRun, runId: snap.run_id, values: snap.values, interrupt: snap.interrupt, nodes, done: finished,
        usage: snap.usage?.calls ? snap.usage : null, logs: snap.logs ?? [] })
    } catch (e) {
      setError(`Cannot load run: ${(e as Error).message}`)
    }
  }

  return (
    <>
      <header className="top">
        <h1>Agentic CV Ranking {isMock && <span className="badge amber">mock</span>}</h1>
        <div className="grow" />
        <form className="row" onSubmit={(e) => { e.preventDefault(); load() }}>
          <input value={loadId} onChange={(e) => setLoadId(e.target.value)} placeholder="run id" />
          <button type="submit" className="ghost">Load run</button>
        </form>
        {config?.langfuse_url && (
          <a href={config.langfuse_url} target="_blank" rel="noreferrer">Open traces in Langfuse</a>
        )}
      </header>
      {error && (
        <div className="banner" role="alert">
          <span>{error}</span>
          <button className="ghost" onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}
      <div className="layout">
        <aside>
          <RunPanel config={config} running={run.running} onStart={start} />
          <CvPool onOpen={setPreviewId} />
          {run.runId && (
            <p className="muted mono">
              run {run.runId}
              {run.traceUrl && <> · <a href={run.traceUrl} target="_blank" rel="noreferrer">trace</a></>}
            </p>
          )}
          {run.usage && (
            <p className="muted mono" title={Object.entries(run.usage.by_node).map(([n, u]) => `${n}: ${u.calls} calls, ${u.input_tokens + u.output_tokens} tokens`).join('\n')}>
              LLM {run.usage.calls} calls · {run.usage.total_tokens.toLocaleString()} tokens
            </p>
          )}
          <Requirements run={run} />
        </aside>
        <main>
          <GraphView run={run} />
          {run.interrupt && <InterruptPanel interrupt={run.interrupt} criteria={config?.criteria ?? []} onSubmit={resume} />}
          <Results run={run} config={config} onOpenCv={setPreviewId} />
          <Timeline logs={run.logs} />
        </main>
      </div>
      {previewId && <CvPreview id={previewId} onClose={() => setPreviewId(null)} />}
    </>
  )
}
