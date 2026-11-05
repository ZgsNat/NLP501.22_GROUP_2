import { useEffect, useRef, useState } from 'react'
import { api } from './api'
import { useT } from './i18n'
import type { CVListItem } from './types'

// Sidebar card: upload a CV (.pdf .docx .md .txt) and open the parse preview of any CV in the pool.
export default function CvPool({ onOpen }: { onOpen: (id: string) => void }) {
  const { t } = useT()
  const [cvs, setCvs] = useState<CVListItem[]>([])
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const input = useRef<HTMLInputElement>(null)

  const refresh = () => api.listCvs().then(setCvs).catch((e) => setErr(`${t('Cannot list CVs', 'Không tải được danh sách CV')}: ${e.message}`))
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
      <h2>{t('CV pool', 'Kho CV')} <span className="muted small">{cvs.length} CV</span></h2>
      <input ref={input} type="file" hidden accept=".pdf,.docx,.md,.txt,.png,.jpg,.jpeg" onChange={(e) => upload(e.target.files?.[0])} />
      <button className="ghost" disabled={busy} onClick={() => input.current?.click()}>
        {busy ? t('Parsing…', 'Đang bóc tách…') : t('Upload CV', 'Tải CV lên')}
      </button>
      <p className="muted small">{t('.pdf, .docx, .md or .txt. You get a parse preview right after.',
        '.pdf, .docx, .md hoặc .txt. Xem ngay kết quả bóc tách sau khi tải lên.')}</p>
      {err && <p className="err small">{err}</p>}
      {cvs.length > 0 && (
        <details>
          <summary className="small">{t('Browse pool', 'Xem danh sách CV')}</summary>
          <ul className="pool">
            {cvs.map((c) => (
              <li key={c.id}>
                <button className="link mono" onClick={() => onOpen(c.id)}>{c.id}</button>
                <span className="small">{c.headline}</span>
                <span className="badge gray">{c.lang}</span>
                {c.uploaded && <span className="badge sky">{t('uploaded', 'đã tải lên')}</span>}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  )
}
