import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import type { CVListItem } from './types'

// Sidebar card: upload a CV (.pdf .docx .md .txt) and open the parse preview of any CV in the pool.
export default function CvPool({ onOpen }: { onOpen: (id: string) => void }) {
  const [cvs, setCvs] = useState<CVListItem[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  const refresh = () => api.listCvs().then(setCvs).catch((e) => setErr(`Cannot list CVs: ${e.message}`))
  useEffect(() => { refresh() }, [])

  async function upload(file: File | undefined) {
    if (!file) return
    setBusy(true); setErr(null)
    try {
      const profile = await api.uploadCv(file)
      await refresh()
      onOpen(profile.id)
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
      if (input.current) input.current.value = ''
    }
  }

  return (
    <section className="card">
      <h2>CV pool <span className="muted small">{cvs.length} CVs</span></h2>
      <input ref={input} type="file" hidden accept=".pdf,.docx,.md,.txt" onChange={(e) => upload(e.target.files?.[0])} />
      <button className="ghost" disabled={busy} onClick={() => input.current?.click()}>{busy ? 'Parsing…' : 'Upload CV'}</button>
      <p className="muted small">.pdf, .docx, .md or .txt. You get a parse preview right after.</p>
      {err && <p className="err small">{err}</p>}
      {cvs.length > 0 && (
        <details>
          <summary className="small">Browse pool</summary>
          <ul className="pool">
            {cvs.map((c) => (
              <li key={c.id}>
                <button className="link mono" onClick={() => onOpen(c.id)}>{c.id}</button>
                <span className="small">{c.headline}</span>
                <span className="badge gray">{c.lang}</span>
                {c.uploaded && <span className="badge sky">uploaded</span>}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  )
}
