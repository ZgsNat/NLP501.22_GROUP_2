import { mockApi } from './mock'
import type { Config, CV, CVListItem, CVProfileView, ResumeBody, RunRequest, RunSnapshot, StreamEvent } from './types'

const BASE = '/api'

async function check(r: Response): Promise<Response> {
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${(await r.text()).slice(0, 300)}`)
  return r
}

const getJson = async <T>(path: string): Promise<T> => (await check(await fetch(BASE + path))).json()

// POST and read the NDJSON response body incrementally, yielding one event per line.
async function* postStream(path: string, body: unknown): AsyncGenerator<StreamEvent> {
  const r = await check(await fetch(BASE + path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }))
  const reader = r.body!.getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    buf += dec.decode(value, { stream: !done })
    const lines = buf.split('\n')
    buf = done ? '' : lines.pop()!
    for (const l of lines) if (l.trim()) yield JSON.parse(l) as StreamEvent
    if (done) return
  }
}

// Multipart upload; surfaces the backend's `{detail}` message on 400.
async function postFile<T>(path: string, file: File): Promise<T> {
  const body = new FormData()
  body.append('file', file)
  const r = await fetch(BASE + path, { method: 'POST', body })
  if (!r.ok) {
    const t = await r.text()
    let detail = t.slice(0, 300)
    try { detail = JSON.parse(t).detail ?? detail } catch { /* not JSON */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return r.json()
}

export interface Api {
  config(): Promise<Config>
  start(req: RunRequest): AsyncIterable<StreamEvent>
  resume(runId: string, body: ResumeBody): AsyncIterable<StreamEvent>
  getRun(runId: string): Promise<RunSnapshot>
  getCv(id: string): Promise<CV>
  listCvs(): Promise<CVListItem[]>
  uploadCv(file: File): Promise<CVProfileView>
}

const realApi: Api = {
  config: () => getJson('/config'),
  start: (req) => postStream('/runs', req),
  resume: (id, body) => postStream(`/runs/${encodeURIComponent(id)}/resume`, body),
  getRun: (id) => getJson(`/runs/${encodeURIComponent(id)}`),
  getCv: (id) => getJson(`/cvs/${encodeURIComponent(id)}`),
  listCvs: () => getJson('/cvs'),
  uploadCv: (file) => postFile('/cvs', file),
}

// `?mock=1` replays a canned run (see mock.ts) so the UI works without the backend.
export const isMock = new URLSearchParams(location.search).has('mock')
export const api: Api = isMock ? mockApi : realApi
