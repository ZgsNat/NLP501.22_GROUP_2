import { useState } from 'react'

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
            <h3>Lịch sử các đợt sàng lọc (Run History)</h3>
            <p className="muted small">Xem lại hoặc mở lại các phiên làm việc trước đó.</p>
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Close">✕</button>
        </div>

        <form className="manual-load-box" onSubmit={handleManualSubmit}>
          <input
            type="text"
            placeholder="Nhập mã phiên (Run ID) cụ thể..."
            value={manualId}
            onChange={(e) => setManualId(e.target.value)}
          />
          <button type="submit" className="primary" disabled={!manualId.trim()}>
            Mở phiên
          </button>
        </form>

        <div className="history-list-box">
          {history.length > 0 ? (
            <table className="history-table">
              <thead>
                <tr>
                  <th>Mã phiên</th>
                  <th>Thời gian</th>
                  <th>Vị trí tuyển dụng</th>
                  <th>Ứng viên dẫn đầu</th>
                  <th className="text-right">Hành động</th>
                </tr>
              </thead>
              <tbody>
                {history.map((item) => (
                  <tr key={item.id} className={currentRunId === item.id ? 'current-row' : ''}>
                    <td>
                      <span className="mono strong">{item.id}</span>
                      {currentRunId === item.id && <span className="badge sky ml-1">Đang xem</span>}
                    </td>
                    <td className="muted small nowrap">
                      {new Date(item.timestamp).toLocaleString('vi-VN')}
                    </td>
                    <td>{item.title || 'Chiến dịch tuyển dụng'}</td>
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
                        Tải lại
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <div className="empty-box muted text-center">
              Chưa có lịch sử phiên làm việc nào được lưu trên trình duyệt này.
            </div>
          )}
        </div>

        <div className="modal-footer flex-between">
          {history.length > 0 ? (
            <button className="ghost danger-text" onClick={onClearHistory}>
              Xóa toàn bộ lịch sử
            </button>
          ) : <div />}
          <button className="primary" onClick={onClose}>Đóng</button>
        </div>
      </div>
    </div>
  )
}
