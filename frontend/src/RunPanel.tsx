import { useRef, useState } from 'react'
import { api, type ParsedJd } from './api'
import type { Config, Mode, RunRequest } from './types'

interface Props {
  config: Config | null
  running: boolean
  onStart: (req: RunRequest) => void
}

export default function RunPanel({ config, running, onStart }: Props) {
  const [jd, setJd] = useState('')
  const [profile, setProfile] = useState('default')
  const [reasoning, setReasoning] = useState('')
  const [fast, setFast] = useState('')
  const [mode, setMode] = useState<Mode>('agent')
  const [goal, setGoal] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [parsedJd, setParsedJd] = useState<ParsedJd | null>(null)
  const [parsingJd, setParsingJd] = useState(false)
  const [jdUploadErr, setJdUploadErr] = useState<string | null>(null)
  const jdFileInputRef = useRef<HTMLInputElement>(null)

  if (!config) {
    return (
      <section className="card">
        <h3>Thiết lập vị trí tuyển dụng</h3>
        <p className="muted small">Đang nạp cấu hình hệ thống...</p>
      </section>
    )
  }

  const pickSample = (id: string) => {
    const s = config.sample_jds.find((j) => j.id === id)
    if (s) {
      setJd(s.text)
      setProfile(s.company_profile)
      setParsedJd(null)
    }
  }

  const handleJdFile = async (file?: File) => {
    if (!file) return
    setParsingJd(true)
    setJdUploadErr(null)
    try {
      const res = await api.parseJd(file)
      setJd(res.raw_text)
      setParsedJd(res)
      if (!goal.trim() && res.must_have.length > 0) {
        setGoal(`Cần ứng viên đáp ứng: ${res.must_have.join(', ')}; tối thiểu ${res.min_years || 2} năm kinh nghiệm.`)
      }
    } catch (e) {
      setJdUploadErr(`Lỗi đọc tệp JD: ${(e as Error).message}`)
    } finally {
      setParsingJd(false)
      if (jdFileInputRef.current) jdFileInputRef.current.value = ''
    }
  }

  const submit = () => {
    const models = {
      ...(reasoning.trim() && { reasoning: reasoning.trim() }),
      ...(fast.trim() && { fast: fast.trim() }),
    }
    onStart({
      jd_text: jd,
      company_profile: profile,
      mode,
      ...(mode === 'agent' && { goal: goal.trim() }),
      ...(Object.keys(models).length && { models }),
    })
  }

  return (
    <section className="card run-panel-card">
      <div className="card-header-clean">
        <div>
          <h3>Chiến dịch tuyển dụng mới</h3>
          <p className="muted small">Nhập yêu cầu từ file Word/PDF hoặc chọn mẫu tiêu chuẩn</p>
        </div>
        <span className="badge slate">Bước 1: JD</span>
      </div>

      <div className="jd-source-toolbar">
        <div className="form-group flex-1">
          <label className="field-label">Mẫu JD chuẩn có sẵn</label>
          <select defaultValue="" onChange={(e) => pickSample(e.target.value)} className="select-input">
            <option value="" disabled>Chọn mẫu JD Automation QA...</option>
            {config.sample_jds.map((j) => (
              <option key={j.id} value={j.id}>
                {j.title || j.id}
              </option>
            ))}
          </select>
        </div>

        <div className="upload-jd-btn-wrap">
          <input
            ref={jdFileInputRef}
            type="file"
            hidden
            accept=".pdf,.docx,.md,.txt"
            onChange={(e) => handleJdFile(e.target.files?.[0])}
          />
          <button
            type="button"
            className="ghost jd-file-btn"
            disabled={parsingJd || running}
            onClick={() => jdFileInputRef.current?.click()}
          >
            {parsingJd ? 'Đang bóc tách NLP...' : '📄 Tải lên tệp JD (.docx, .pdf, .txt)'}
          </button>
        </div>
      </div>

      {jdUploadErr && <p className="err small">{jdUploadErr}</p>}

      {/* NLP Parsed Requirements Preview Card */}
      {parsedJd && (
        <div className="nlp-preview-card">
          <div className="flex-between">
            <span className="nlp-tag">🧠 Bóc tách NLP (NLP Information Extraction)</span>
            {parsedJd.filename && <span className="muted small mono">{parsedJd.filename}</span>}
          </div>
          <h4 className="nlp-title">{parsedJd.title}</h4>

          <div className="nlp-fields">
            {parsedJd.must_have.length > 0 && (
              <div className="nlp-field">
                <span className="small strong text-rose">Bắt buộc (Must-have):</span>
                <div className="chips">
                  {parsedJd.must_have.map((s) => (
                    <span key={s} className="badge rose">{s}</span>
                  ))}
                </div>
              </div>
            )}

            {parsedJd.nice_to_have.length > 0 && (
              <div className="nlp-field">
                <span className="small strong text-sky">Ưu tiên (Nice-to-have):</span>
                <div className="chips">
                  {parsedJd.nice_to_have.map((s) => (
                    <span key={s} className="badge sky">{s}</span>
                  ))}
                </div>
              </div>
            )}

            <div className="nlp-metrics-row">
              {parsedJd.min_years != null && (
                <span className="metric-chip">Kinh nghiệm: <b>{parsedJd.min_years}+ năm</b></span>
              )}
              {parsedJd.depth_level != null && (
                <span className="metric-chip">Cấp độ: <b>L{parsedJd.depth_level}</b></span>
              )}
              {parsedJd.test_types.length > 0 && (
                <span className="metric-chip">Loại kiểm thử: <b>{parsedJd.test_types.join(', ')}</b></span>
              )}
              {parsedJd.domains.length > 0 && (
                <span className="metric-chip">Lĩnh vực: <b>{parsedJd.domains.join(', ')}</b></span>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="form-group">
        <label className="field-label">
          Nội dung bản mô tả công việc (JD Text) <span className="req">*</span>
        </label>
        <textarea
          rows={8}
          value={jd}
          onChange={(e) => {
            setJd(e.target.value)
            setParsedJd(null)
          }}
          placeholder="Dán hoặc tải lên nội dung JD (yêu cầu kỹ năng, công cụ, số năm kinh nghiệm)..."
          className="textarea-input"
        />
        {jd.trim() && (
          <span className="char-count muted small">
            Đã nhập {jd.trim().split(/\s+/).length} từ ({jd.length} ký tự)
          </span>
        )}
      </div>

      <div className="form-group">
        <label className="field-label">Chế độ phân tích & Đánh giá</label>
        <div className="mode-toggle-group">
          <button
            type="button"
            className={`mode-btn ${mode === 'agent' ? 'active' : ''}`}
            onClick={() => setMode('agent')}
          >
            <div className="mode-title">🤖 Trợ lý Agentic (Khuyên dùng)</div>
            <div className="mode-desc">Tự động lập kế hoạch đa bước, tương tác hỏi HR khi JD có điểm mơ hồ</div>
          </button>
          <button
            type="button"
            className={`mode-btn ${mode === 'pipeline' ? 'active' : ''}`}
            onClick={() => setMode('pipeline')}
          >
            <div className="mode-title">⚡ Quy trình chuẩn (Pipeline)</div>
            <div className="mode-desc">Luồng lọc cố định, tối ưu tốc độ và chi phí</div>
          </button>
        </div>
      </div>

      {mode === 'agent' && (
        <div className="form-group">
          <label className="field-label">
            Mục tiêu ưu tiên của bạn (Hiring Goal)
          </label>
          <input
            type="text"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder="VD: Cần 5 ứng viên giỏi Playwright và có kinh nghiệm CI/CD; ưu tiên có ISTQB"
            className="text-input"
          />
          <p className="muted small">Agent sẽ bám sát mục tiêu này trong các bước kiểm chứng và so sánh đối đầu.</p>
        </div>
      )}

      {/* Advanced configuration collapsible */}
      <div className="advanced-section">
        <button
          type="button"
          className="advanced-toggle"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          <span>{showAdvanced ? '▾ Thu gọn cấu hình kỹ thuật' : '▸ Cấu hình mô hình AI & Hồ sơ công ty'}</span>
        </button>

        {showAdvanced && (
          <div className="advanced-content">
            <div className="form-group">
              <label className="field-label">Hồ sơ công ty (Company Culture / Profile)</label>
              <select value={profile} onChange={(e) => setProfile(e.target.value)} className="select-input">
                {config.company_profiles.map((p) => (
                  <option key={p.key} value={p.key}>{p.label}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="field-label">Mô hình suy luận chính (Reasoning Model)</label>
              <input
                list="providers"
                value={reasoning}
                onChange={(e) => setReasoning(e.target.value)}
                placeholder={config.defaults.reasoning}
                className="text-input"
              />
            </div>

            <div className="form-group">
              <label className="field-label">Mô hình bóc tách nhanh (Fast Model)</label>
              <input
                list="providers"
                value={fast}
                onChange={(e) => setFast(e.target.value)}
                placeholder={config.defaults.fast}
                className="text-input"
              />
            </div>

            <datalist id="providers">
              {config.providers.map((p) => (
                <option key={p.name} value={`${p.name}:`} />
              ))}
            </datalist>

            <div className="provider-status-chips">
              <span className="small muted">Trạng thái cổng kết nối:</span>
              <div className="chips mt-1">
                {config.providers.map((p) => (
                  <span key={p.name} className={`badge ${p.configured ? 'green' : 'gray'}`}>
                    {p.name}: {p.configured ? 'Sẵn sàng' : 'Chưa có key'}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <button
        type="button"
        className="primary full-width-btn"
        disabled={running || !jd.trim()}
        onClick={submit}
      >
        {running ? 'Đang thực hiện sàng lọc & xếp hạng...' : '🚀 Bắt đầu sàng lọc hồ sơ'}
      </button>
    </section>
  )
}
