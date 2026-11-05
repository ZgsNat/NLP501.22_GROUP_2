import { useRef, useState } from 'react'
import { api, type ParsedJd } from './api'
import { useT } from './i18n'
import type { Config, Mode, RunRequest } from './types'

interface Props {
  config: Config | null
  running: boolean
  onStart: (req: RunRequest) => void
}

export default function RunPanel({ config, running, onStart }: Props) {
  const { t, term, lang } = useT()
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
        <h3>{t('New hiring campaign', 'Chiến dịch tuyển dụng mới')}</h3>
        <p className="muted small">{t('Loading the configuration…', 'Đang nạp cấu hình hệ thống…')}</p>
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
        const years = res.min_years || 2
        setGoal(t(`Candidates who meet: ${res.must_have.join(', ')}; at least ${years} years of experience.`,
          `Cần ứng viên đáp ứng: ${res.must_have.join(', ')}; tối thiểu ${years} năm kinh nghiệm.`))
      }
    } catch (e) {
      setJdUploadErr(`${t('Cannot read the JD file', 'Lỗi đọc tệp JD')}: ${(e as Error).message}`)
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
      lang, // the language the LLM writes its explanations, report and replies in
      ...(mode === 'agent' && { goal: goal.trim() }),
      ...(Object.keys(models).length && { models }),
    })
  }

  return (
    <section className="card run-panel-card">
      <div className="card-header-clean">
        <div>
          <h3>{t('New hiring campaign', 'Chiến dịch tuyển dụng mới')}</h3>
          <p className="muted small">{t('Load the requirements from a Word/PDF file or pick a sample',
            'Nhập yêu cầu từ file Word/PDF hoặc chọn mẫu có sẵn')}</p>
        </div>
        <span className="badge slate">{t('Step 1: JD', 'Bước 1: JD')}</span>
      </div>

      <div className="jd-source-toolbar">
        <div className="form-group flex-1">
          <label className="field-label">{t('Sample JDs', `Mẫu ${term('JD')} có sẵn`)}</label>
          <select defaultValue="" onChange={(e) => pickSample(e.target.value)} className="select-input">
            <option value="" disabled>{t('Pick an Automation QA JD…', 'Chọn mẫu JD Automation QA…')}</option>
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
            {parsingJd ? t('Parsing (NLP)…', 'Đang bóc tách (NLP)…') : `📄 ${t('Upload a JD file (.docx, .pdf, .txt)', 'Tải lên tệp JD (.docx, .pdf, .txt)')}`}
          </button>
        </div>
      </div>

      {jdUploadErr && <p className="err small">{jdUploadErr}</p>}

      {/* NLP Parsed Requirements Preview Card */}
      {parsedJd && (
        <div className="nlp-preview-card">
          <div className="flex-between">
            <span className="nlp-tag">🧠 {t('NLP information extraction', 'Bóc tách thông tin (NLP information extraction)')}</span>
            {parsedJd.filename && <span className="muted small mono">{parsedJd.filename}</span>}
          </div>
          <h4 className="nlp-title">{parsedJd.title}</h4>

          <div className="nlp-fields">
            {parsedJd.must_have.length > 0 && (
              <div className="nlp-field">
                <span className="small strong text-rose">{t('Must-have:', `${term('Must-have')}:`)}</span>
                <div className="chips">
                  {parsedJd.must_have.map((s) => (
                    <span key={s} className="badge rose">{s}</span>
                  ))}
                </div>
              </div>
            )}

            {parsedJd.nice_to_have.length > 0 && (
              <div className="nlp-field">
                <span className="small strong text-sky">{t('Nice-to-have:', `${term('Nice-to-have')}:`)}</span>
                <div className="chips">
                  {parsedJd.nice_to_have.map((s) => (
                    <span key={s} className="badge sky">{s}</span>
                  ))}
                </div>
              </div>
            )}

            <div className="nlp-metrics-row">
              {parsedJd.min_years != null && (
                <span className="metric-chip">{t('Experience', 'Kinh nghiệm')}: <b>{parsedJd.min_years}+ {t('years', 'năm')}</b></span>
              )}
              {parsedJd.depth_level != null && (
                <span className="metric-chip">{t('Depth', 'Mức độ')}: <b>L{parsedJd.depth_level}</b></span>
              )}
              {parsedJd.test_types.length > 0 && (
                <span className="metric-chip">{t('Test types', 'Loại kiểm thử')}: <b>{parsedJd.test_types.join(', ')}</b></span>
              )}
              {parsedJd.domains.length > 0 && (
                <span className="metric-chip">{t('Domains', 'Lĩnh vực')}: <b>{parsedJd.domains.join(', ')}</b></span>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="form-group">
        <label className="field-label">
          {t('Job description text', 'Nội dung bản mô tả công việc (JD)')} <span className="req">*</span>
        </label>
        <textarea
          rows={8}
          value={jd}
          onChange={(e) => {
            setJd(e.target.value)
            setParsedJd(null)
          }}
          placeholder={t('Paste or upload the JD (skills, tools, years of experience)…',
            'Dán hoặc tải lên nội dung JD (yêu cầu kỹ năng, công cụ, số năm kinh nghiệm)…')}
          className="textarea-input"
        />
        {jd.trim() && (
          <span className="char-count muted small">
            {t(`${jd.trim().split(/\s+/).length} words (${jd.length} characters)`,
              `Đã nhập ${jd.trim().split(/\s+/).length} từ (${jd.length} ký tự)`)}
          </span>
        )}
      </div>

      <div className="form-group">
        <label className="field-label">{t('Analysis mode', 'Chế độ phân tích & đánh giá')}</label>
        <div className="mode-toggle-group">
          <button
            type="button"
            className={`mode-btn ${mode === 'agent' ? 'active' : ''}`}
            onClick={() => setMode('agent')}
          >
            <div className="mode-title">🤖 {t('Agentic assistant (recommended)', `Trợ lý ${term('Agentic')} · khuyên dùng`)}</div>
            <div className="mode-desc">{t('Plans its own steps and asks HR when the JD is ambiguous',
              'Tự lập kế hoạch nhiều bước, hỏi lại HR khi JD có điểm mơ hồ')}</div>
          </button>
          <button
            type="button"
            className={`mode-btn ${mode === 'pipeline' ? 'active' : ''}`}
            onClick={() => setMode('pipeline')}
          >
            <div className="mode-title">⚡ {t('Fixed pipeline', term('Pipeline'))}</div>
            <div className="mode-desc">{t('A fixed screening flow, fastest and cheapest',
              'Luồng lọc cố định, nhanh nhất và tốn ít chi phí nhất')}</div>
          </button>
        </div>
      </div>

      {mode === 'agent' && (
        <div className="form-group">
          <label className="field-label">{t('Your hiring goal', `${term('Hiring goal')} của bạn`)}</label>
          <input
            type="text"
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            placeholder={t('e.g. 5 strong Playwright testers with CI/CD; ISTQB is a plus',
              'VD: Cần 5 ứng viên giỏi Playwright và có kinh nghiệm CI/CD; ưu tiên có ISTQB')}
            className="text-input"
          />
          <p className="muted small">{t('The agent keeps to this goal when it verifies and compares candidates.',
            'Agent sẽ bám sát mục tiêu này khi kiểm chứng và so sánh đối đầu các ứng viên.')}</p>
        </div>
      )}

      {/* Advanced configuration collapsible */}
      <div className="advanced-section">
        <button type="button" className="advanced-toggle" onClick={() => setShowAdvanced(!showAdvanced)}>
          <span>{showAdvanced
            ? `▾ ${t('Hide technical settings', 'Thu gọn cấu hình kỹ thuật')}`
            : `▸ ${t('AI models & company profile', 'Cấu hình mô hình AI & hồ sơ công ty')}`}</span>
        </button>

        {showAdvanced && (
          <div className="advanced-content">
            <div className="form-group">
              <label className="field-label">{t('Company profile', term('Company profile'))}</label>
              <select value={profile} onChange={(e) => setProfile(e.target.value)} className="select-input">
                {config.company_profiles.map((p) => (
                  <option key={p.key} value={p.key}>{p.label}</option>
                ))}
              </select>
            </div>

            <div className="form-group">
              <label className="field-label">{t('Reasoning model', term('Reasoning model'))}</label>
              <input
                list="providers"
                value={reasoning}
                onChange={(e) => setReasoning(e.target.value)}
                placeholder={config.defaults.reasoning}
                className="text-input"
              />
            </div>

            <div className="form-group">
              <label className="field-label">{t('Fast model', term('Fast model'))}</label>
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
              <span className="small muted">{t('Provider status:', 'Trạng thái cổng kết nối:')}</span>
              <div className="chips mt-1">
                {config.providers.map((p) => (
                  <span key={p.name} className={`badge ${p.configured ? 'green' : 'gray'}`}>
                    {p.name}: {p.configured ? t('ready', 'sẵn sàng') : t('no key', 'chưa có key')}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <button type="button" className="primary full-width-btn" disabled={running || !jd.trim()} onClick={submit}>
        {running ? t('Screening and ranking…', 'Đang sàng lọc & xếp hạng…') : `🚀 ${t('Start screening', 'Bắt đầu sàng lọc hồ sơ')}`}
      </button>
    </section>
  )
}
