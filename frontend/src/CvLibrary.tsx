import { useEffect, useMemo, useState } from 'react'
import { api } from './api'
import BatchUploadModal from './BatchUploadModal'
import { useT } from './i18n'
import type { CVListItem } from './types'

interface Props {
  onOpenCv: (id: string) => void
}

export default function CvLibrary({ onOpenCv }: Props) {
  const { t } = useT()
  const [cvs, setCvs] = useState<CVListItem[]>([])
  const [search, setSearch] = useState('')
  const [selectedSkill, setSelectedSkill] = useState<string>('all')
  const [sourceFilter, setSourceFilter] = useState<'all' | 'uploaded' | 'benchmark'>('all')
  const [isBatchOpen, setIsBatchOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [deleteId, setDeleteId] = useState<string | null>(null)

  const loadData = () => {
    setLoading(true)
    setError(null)
    api.listCvs()
      .then(setCvs)
      .catch((e) => setError(`${t('Cannot load the CV list', 'Không thể tải danh sách CV')}: ${e.message}`))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadData()
  }, [])

  // Collect all unique skills across all CVs
  const allSkills = useMemo(() => {
    const set = new Set<string>()
    for (const c of cvs) {
      if (c.skills) for (const s of c.skills) set.add(s)
    }
    return Array.from(set).sort()
  }, [cvs])

  // Filtered CV list
  const filtered = useMemo(() => {
    return cvs.filter((c) => {
      // Source filter
      if (sourceFilter === 'uploaded' && !c.uploaded) return false
      if (sourceFilter === 'benchmark' && c.uploaded) return false

      // Skill filter
      if (selectedSkill !== 'all' && (!c.skills || !c.skills.includes(selectedSkill))) {
        return false
      }

      // Search term
      if (search.trim()) {
        const q = search.toLowerCase()
        const matchId = c.id.toLowerCase().includes(q)
        const matchHeadline = (c.headline || '').toLowerCase().includes(q)
        const matchFilename = (c.filename || '').toLowerCase().includes(q)
        const matchSkills = (c.skills || []).some((s) => s.toLowerCase().includes(q))
        if (!matchId && !matchHeadline && !matchFilename && !matchSkills) return false
      }

      return true
    })
  }, [cvs, sourceFilter, selectedSkill, search])

  const handleDelete = async (id: string) => {
    if (!window.confirm(t(`Delete CV ${id}?`, `Bạn có chắc chắn muốn xoá hồ sơ ${id}?`))) return
    setDeleteId(id)
    try {
      await api.deleteCv(id)
      await loadData()
    } catch (e) {
      alert(`${t('Delete failed', 'Xoá thất bại')}: ${(e as Error).message}`)
    } finally {
      setDeleteId(null)
    }
  }

  const handleResetUploads = async () => {
    if (!window.confirm(t('Delete ALL uploaded CVs and go back to the benchmark pool?',
      'Bạn có chắc muốn xoá TẤT CẢ hồ sơ đã tải lên để hoàn nguyên kho mẫu chuẩn?'))) return
    setLoading(true)
    try {
      const res = await api.resetCvs()
      alert(t(`Pool restored (${res.deleted_count} uploaded CVs removed).`,
        `Đã hoàn nguyên kho dữ liệu (đã xoá ${res.deleted_count} hồ sơ tải lên).`))
      await loadData()
    } catch (e) {
      alert(`${t('Error', 'Lỗi')}: ${(e as Error).message}`)
    } finally {
      setLoading(false)
    }
  }

  const uploadedCount = cvs.filter((c) => c.uploaded).length
  const benchmarkCount = cvs.length - uploadedCount

  return (
    <div className="cv-library-page">
      <header className="page-header">
        <div>
          <h2>{t('Candidate pool', 'Kho hồ sơ ứng viên (candidate pool)')}</h2>
          <p className="muted">{t('Every CV used for testing, screening and automated assessment.',
            'Quản lý toàn bộ hồ sơ ứng viên dùng cho kiểm thử, sàng lọc và đánh giá tự động.')}</p>
        </div>
        <div className="header-actions">
          {uploadedCount > 0 && (
            <button className="ghost danger-text" onClick={handleResetUploads}
              title={t('Delete every CV users uploaded', 'Xoá tất cả CV người dùng đã tải lên')}>
              {t('Clear uploads', 'Dọn dẹp tải lên')} ({uploadedCount})
            </button>
          )}
          <button className="primary" onClick={() => setIsBatchOpen(true)}>
            + {t('Batch upload', 'Tải lên hàng loạt')}
          </button>
        </div>
      </header>

      {error && <div className="banner" role="alert">{error}</div>}

      <div className="stats-cards-grid">
        <div className="stat-card">
          <span className="stat-title">{t('CVs in total', 'Tổng số hồ sơ')}</span>
          <span className="stat-value">{cvs.length}</span>
          <span className="stat-desc">{t(`${uploadedCount} uploaded, ${benchmarkCount} benchmark`,
            `Gồm ${uploadedCount} tải lên, ${benchmarkCount} mẫu chuẩn`)}</span>
        </div>
        <div className="stat-card">
          <span className="stat-title">{t('Skills detected', 'Kỹ năng nhận diện được')}</span>
          <span className="stat-value">{allSkills.length}</span>
          <span className="stat-desc">{t('Technical skills & QA tools', 'Kỹ năng công nghệ & công cụ QA')}</span>
        </div>
        <div className="stat-card">
          <span className="stat-title">{t('CVs shown', 'Hồ sơ đang hiển thị')}</span>
          <span className="stat-value">{filtered.length}</span>
          <span className="stat-desc">{t('With the current filters', 'Theo bộ lọc tìm kiếm hiện tại')}</span>
        </div>
      </div>

      <div className="library-toolbar">
        <div className="search-box">
          <input
            type="text"
            placeholder={t('Search by id, title, skill, file name…', 'Tìm theo mã, chức danh, kỹ năng, tên tệp…')}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          {search && <button className="clear-btn" onClick={() => setSearch('')}>✕</button>}
        </div>

        <div className="filters-row">
          <select value={selectedSkill} onChange={(e) => setSelectedSkill(e.target.value)}>
            <option value="all">{t('All skills', 'Tất cả kỹ năng')} ({allSkills.length})</option>
            {allSkills.map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>

          <select value={sourceFilter} onChange={(e) => setSourceFilter(e.target.value as any)}>
            <option value="all">{t('All sources', 'Tất cả nguồn')} ({cvs.length})</option>
            <option value="uploaded">{t('Uploaded by users', 'Người dùng tải lên')} ({uploadedCount})</option>
            <option value="benchmark">{t('Benchmark pool', 'Kho chuẩn benchmark')} ({benchmarkCount})</option>
          </select>

          <button className="ghost" onClick={loadData} disabled={loading} title={t('Refresh', 'Làm mới')}>
            {loading ? t('Loading…', 'Đang tải…') : `↻ ${t('Refresh', 'Làm mới')}`}
          </button>
        </div>
      </div>

      <div className="card table-container">
        <table className="library-table">
          <thead>
            <tr>
              <th>{t('CV id', 'Mã hồ sơ')}</th>
              <th>{t('Headline / title', 'Tiêu đề / chức danh')}</th>
              <th>{t('Main skills', 'Kỹ năng chính')}</th>
              <th>{t('Experience', 'Kinh nghiệm')}</th>
              <th>{t('Language', 'Ngôn ngữ')}</th>
              <th>{t('Source', 'Nguồn')}</th>
              <th className="text-right">{t('Actions', 'Thao tác')}</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((c) => (
              <tr key={c.id}>
                <td>
                  <button className="link mono strong" onClick={() => onOpenCv(c.id)}>
                    {c.id}
                  </button>
                  {c.filename && c.filename !== `${c.id}.md` && (
                    <span className="file-hint muted small">{c.filename}</span>
                  )}
                </td>
                <td className="headline-cell">{c.headline || <span className="muted">{t('No title', 'Chưa có tiêu đề')}</span>}</td>
                <td>
                  <div className="table-skills">
                    {(c.skills || []).slice(0, 4).map((s) => (
                      <span key={s} className="badge gray skill-badge">{s}</span>
                    ))}
                    {(c.skills || []).length > 4 && (
                      <span className="badge gray skill-badge" title={(c.skills || []).slice(4).join(', ')}>
                        +{(c.skills || []).length - 4}
                      </span>
                    )}
                  </div>
                </td>
                <td className="nowrap">
                  {c.years_automation != null ? `${c.years_automation} ${t('yrs', 'năm')}` : '-'}
                </td>
                <td>
                  <span className={`badge ${c.lang === 'vi' ? 'emerald' : c.lang === 'en' ? 'sky' : 'amber'}`}>
                    {c.lang.toUpperCase()}
                  </span>
                </td>
                <td>
                  {c.uploaded ? (
                    <span className="badge sky">{t('Uploaded', 'Đã tải lên')}</span>
                  ) : (
                    <span className="badge slate">{t('Benchmark', 'Chuẩn')}</span>
                  )}
                </td>
                <td className="text-right nowrap">
                  <button className="ghost small-btn" onClick={() => onOpenCv(c.id)}>
                    {t('Details', 'Chi tiết')}
                  </button>
                  {c.uploaded && (
                    <button
                      className="ghost small-btn danger-text"
                      disabled={deleteId === c.id}
                      onClick={() => handleDelete(c.id)}
                    >
                      {deleteId === c.id ? '...' : t('Delete', 'Xoá')}
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={7} className="text-center muted empty-cell">
                  {loading ? t('Loading…', 'Đang nạp dữ liệu…') : t('No CV matches the filters.', 'Không tìm thấy hồ sơ phù hợp với bộ lọc.')}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <BatchUploadModal
        isOpen={isBatchOpen}
        onClose={() => setIsBatchOpen(false)}
        onSuccess={loadData}
      />
    </div>
  )
}
