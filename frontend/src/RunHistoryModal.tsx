import { useState } from 'react'
import { useT } from './i18n'

export interface HistoryItem {
  id: string
  timestamp: string
  title: string
  topCandidate?: string
  mode?: string
}

interface Props {
  isOpen: boolean
  currentRunId: string | null
  history: HistoryItem[]
  onClose: () => void
  onSelectRun: (id: string) => void
  onClearHistory: () => void
}

export default function RunHistoryModal({
  isOpen,
  currentRunId,
  history,
  onClose,
  onSelectRun,
  onClearHistory,
}: Props) {
  const { t, lang } = useT()
  const [manualId, setManualId] = useState('')

  if (!isOpen) return null

  const handleManualSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (manualId.trim()) {
      onSelectRun(manualId.trim())
      onClose()
    }
  }

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-card history-modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h3>{t('Run history', 'Lịch sử các đợt sàng lọc (run history)')}</h3>
            <p className="muted small">{t('Review or reopen earlier runs.', 'Xem lại hoặc mở lại các phiên làm việc trước đó.')}</p>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label={t('Close', 'Đóng')}>✕</button>
        </div>

        <form className="manual-load-box" onSubmit={handleManualSubmit}>
          <input
            type="text"
            placeholder={t('Enter a run id…', 'Nhập mã phiên (run id) cụ thể…')}
            value={manualId}
            onChange={(e) => setManualId(e.target.value)}
          />
          <button type="submit" className="primary" disabled={!manualId.trim()}>
            {t('Open', 'Mở phiên')}
          </button>
        </form>

        <div className="history-list-box">
          {history.length > 0 ? (
            <table className="history-table">
              <thead>
                <tr>
                  <th>{t('Run id', 'Mã phiên')}</th>
                  <th>{t('Time', 'Thời gian')}</th>
                  <th>{t('Position', 'Vị trí tuyển dụng')}</th>
                  <th>{t('Top candidate', 'Ứng viên dẫn đầu')}</th>
                  <th className="text-right">{t('Action', 'Thao tác')}</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.id} className={currentRunId === item.id ? 'current-row' : ''}>
                    <td>
                      <span className="mono strong">{item.id}</span>
                      {currentRunId === item.id && <span className="badge sky ml-1">{t('Viewing', 'Đang xem')}</span>}
                    </td>
                    <td className="muted small nowrap">
                      {new Date(item.timestamp).toLocaleString(lang === 'vi' ? 'vi-VN' : 'en-GB')}
                    </td>
                    <td>{item.title || t('Hiring campaign', 'Chiến dịch tuyển dụng')}</td>
                    <td>
                      {item.topCandidate ? (
                        <span className="badge green">{item.topCandidate}</span>
                      ) : (
                        <span className="muted">-</span>
                      )}
                    </td>
                    <td className="text-right">
                      <button
                        className="ghost small-btn"
                        onClick={() => {
                          onSelectRun(item.id)
                          onClose()
                        }}
                      >
                        {t('Reload', 'Tải lại')}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="empty-box muted text-center">
              {t('No runs saved in this browser yet.', 'Chưa có lịch sử phiên làm việc nào được lưu trên trình duyệt này.')}
            </div>
          )}
        </div>

        <div className="modal-footer flex-between">
          {history.length > 0 ? (
            <button className="ghost danger-text" onClick={onClearHistory}>
              {t('Clear all history', 'Xoá toàn bộ lịch sử')}
            </button>
          ) : <div />}
          <button className="primary" onClick={onClose}>{t('Close', 'Đóng')}</button>
        </div>
      </div>
    </div>
  )
}
