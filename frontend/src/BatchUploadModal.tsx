import { useRef, useState } from 'react'
import { api } from './api'
import { useT } from './i18n'

interface Props {
  isOpen: boolean
  onClose: () => void
  onSuccess: () => void
}

interface QueuedFile {
  file: File
  status: 'pending' | 'uploading' | 'success' | 'error'
  resultId?: string
  error?: string
}

export default function BatchUploadModal({ isOpen, onClose, onSuccess }: Props) {
  const { t, term } = useT()
  const [queue, setQueue] = useState<QueuedFile[]>([])
  const [busy, setBusy] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  if (!isOpen) return null

  const addFiles = (files: FileList | File[]) => {
    const valid = Array.from(files).filter((f) => {
      const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase()
      return ['.pdf', '.docx', '.md', '.txt', '.png', '.jpg', '.jpeg'].includes(ext)
    })
    setQueue((prev) => [
      ...prev,
      ...valid.map((file) => ({ file, status: 'pending' as const })),
    ])
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    if (e.dataTransfer.files) addFiles(e.dataTransfer.files)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = () => setIsDragging(false)

  const removeFile = (idx: number) => {
    if (busy) return
    setQueue((prev) => prev.filter((_, i) => i !== idx))
  }

  const startUpload = async () => {
    if (!queue.length || busy) return
    setBusy(true)

    // Mark all pending as uploading
    setQueue((prev) => prev.map((q) => (q.status === 'pending' ? { ...q, status: 'uploading' } : q)))

    try {
      const pendingFiles = queue.filter((q) => q.status === 'uploading').map((q) => q.file)
      const res = await api.batchUploadCvs(pendingFiles)

      // Map back results
      setQueue((prev) =>
        prev.map((item) => {
          const successItem = res.items.find((i) => i.filename === item.file.name)
          if (successItem) {
            return { ...item, status: 'success', resultId: successItem.id }
          }
          const errItem = res.errors.find((e) => e.filename === item.file.name)
          if (errItem) {
            return { ...item, status: 'error', error: errItem.error }
          }
          return item
        })
      )
      onSuccess()
    } catch (e) {
      setQueue((prev) =>
        prev.map((item) =>
          item.status === 'uploading' ? { ...item, status: 'error', error: (e as Error).message } : item
        )
      )
    } finally {
      setBusy(false)
    }
  }

  const completedCount = queue.filter((q) => q.status === 'success').length
  const pending = queue.filter((q) => q.status === 'pending').length

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card batch-upload-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h3>{t('Batch CV upload', 'Tải lên hồ sơ ứng viên hàng loạt')}</h3>
            <p className="muted small">{t('.pdf, .docx, .md, .txt. Personal data is masked and the text is parsed (NLP) automatically.',
              `Hỗ trợ .pdf, .docx, .md, .txt. Tự động ${term('Redaction')} và bóc tách ${term('NLP')}.`)}</p>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label={t('Close', 'Đóng')}>✕</button>
        </div>

        <div
          className={`dropzone ${isDragging ? 'active' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            hidden
            accept=".pdf,.docx,.md,.txt,.png,.jpg,.jpeg"
            onChange={(e) => e.target.files && addFiles(e.target.files)}
          />
          <div className="dropzone-content">
            <span className="dropzone-icon">📁</span>
            <p className="dropzone-title">{t('Drop CVs here or click to choose files', 'Kéo thả CV vào đây hoặc bấm để chọn tệp')}</p>
            <p className="muted small">{t('Several files at once are processed together', 'Có thể chọn nhiều tệp cùng lúc để xử lý song song')}</p>
          </div>
        </div>

        {queue.length > 0 && (
          <div className="upload-queue">
            <div className="queue-header">
              <span>{t(`Queue (${queue.length} files)`, `Danh sách chờ (${queue.length} tệp)`)}</span>
              {completedCount > 0 && <span className="badge green">{t(`${completedCount} done`, `${completedCount} đã tải xong`)}</span>}
            </div>
            <ul className="queue-list">
              {queue.map((item, idx) => (
                <li key={idx} className={`queue-item status-${item.status}`}>
                  <span className="file-name">{item.file.name}</span>
                  <span className="file-size muted small">{(item.file.size / 1024).toFixed(1)} KB</span>
                  <span className="status-badge">
                    {item.status === 'pending' && <span className="badge gray">{t('Waiting', 'Chờ xử lý')}</span>}
                    {item.status === 'uploading' && <span className="badge amber">{t('Parsing…', 'Đang bóc tách…')}</span>}
                    {item.status === 'success' && <span className="badge green">✓ {item.resultId}</span>}
                    {item.status === 'error' && <span className="badge rose" title={item.error}>✕ {item.error || t('Error', 'Lỗi')}</span>}
                  </span>
                  {!busy && item.status === 'pending' && (
                    <button className="link remove-btn" onClick={() => removeFile(idx)}>✕</button>
                  )}
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="modal-footer">
          <button className="ghost" onClick={onClose} disabled={busy}>{t('Close', 'Đóng')}</button>
          <button className="primary" onClick={startUpload} disabled={busy || !queue.some((q) => q.status === 'pending')}>
            {busy ? t('Parsing & indexing…', 'Đang phân tích & lập chỉ mục…') : t(`Start (${pending} files)`, `Bắt đầu xử lý (${pending} tệp)`)}
          </button>
        </div>
      </div>
    </div>
  )
}
