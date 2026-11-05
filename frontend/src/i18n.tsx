import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'

// UI language. Every string is written as t('English', 'Tiếng Việt') where it is used, so the two never drift apart.
// Vietnamese convention: a technical term keeps its English name and gets a short gloss in parentheses (term()).
export type UiLang = 'en' | 'vi'
const KEY = 'ts_ui_lang'

export const GLOSS: Record<string, string> = {
  'JD': 'mô tả công việc',
  'CV': 'hồ sơ ứng viên',
  'HR': 'nhân sự tuyển dụng',
  'Shortlist': 'danh sách rút gọn',
  'Knock-out': 'loại trừ',
  'Must-have': 'bắt buộc',
  'Nice-to-have': 'ưu tiên',
  'Pipeline': 'quy trình cố định',
  'Agent': 'tác tử AI',
  'Agentic': 'tự lập kế hoạch',
  'Rerank': 'xếp hạng lại',
  'Cross-encoder': 'mô hình chấm cặp JD–CV',
  'Arbiter': 'trọng tài LLM',
  'Near-tie': 'gần như hoà điểm',
  'Evidence': 'bằng chứng',
  'Grounded': 'có thật trong CV',
  'Supported': 'chứng minh được',
  'Tier': 'nhóm xếp loại',
  'Token': 'đơn vị tính chi phí LLM',
  'LLM': 'mô hình ngôn ngữ lớn',
  'Reasoning model': 'mô hình suy luận',
  'Fast model': 'mô hình nhanh',
  'Company profile': 'hồ sơ công ty',
  'Hiring goal': 'mục tiêu tuyển dụng',
  'Audit': 'kiểm chứng kỹ thuật',
  'Trace': 'vết thực thi',
  'Orchestrator': 'bộ điều phối',
  'Guard': 'lớp kiểm soát',
  'Harness': 'khung kiểm soát',
  'NLP': 'xử lý ngôn ngữ tự nhiên',
  'OCR': 'nhận dạng chữ trong ảnh',
  'Redaction': 'che thông tin cá nhân',
  'Prompt injection': 'chèn lệnh vào CV',
  'Chunk': 'đoạn CV',
  'Weight': 'trọng số',
  'Re-score': 'chấm lại',
  'What-if': 'giả định nếu',
  'Hedged': 'chỉ nói qua / đang học',
  'Listed': 'chỉ liệt kê',
  'Used': 'đã dùng trong công việc',
  'Depth': 'mức độ tự động hoá',
  'CI/CD': 'tích hợp & triển khai liên tục',
  'ISTQB': 'chứng chỉ kiểm thử',
}

// The 8 criteria (data/criteria_profiles.yaml keys). English labels come from the API; Vietnamese ones from here.
export const CRITERION_VI: Record<string, string> = {
  tools: 'Công cụ / framework (tools)',
  automation_depth: 'Mức độ tự động hoá (automation depth)',
  programming_language: 'Ngôn ngữ lập trình',
  test_types: 'Loại kiểm thử (test types)',
  cicd_infra: 'CI/CD & hạ tầng kiểm thử',
  experience_level: 'Số năm kinh nghiệm & cấp bậc',
  istqb: 'Chứng chỉ ISTQB',
  domain_language: 'Kiến thức nghiệp vụ & tiếng Anh',
}

export const FLAG_VI: Record<string, string> = {
  short_text: 'CV quá ngắn',
  garbled_text: 'văn bản lỗi / OCR kém',
  few_skills: 'ít kỹ năng',
  no_dated_roles: 'không có mốc thời gian',
  prompt_injection: 'Prompt injection (chèn lệnh vào CV)',
  duplicate_of: 'trùng với',
  implied_evidence: 'bằng chứng gián tiếp',
}

interface LangCtx { lang: UiLang; setLang: (l: UiLang) => void }
const LangContext = createContext<LangCtx>({ lang: 'vi', setLang: () => {} })

function stored(): UiLang {
  try {
    return localStorage.getItem(KEY) === 'en' ? 'en' : 'vi'
  } catch {
    return 'vi'
  }
}

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<UiLang>(stored)
  useEffect(() => {
    document.documentElement.lang = lang
  }, [lang])
  const setLang = (l: UiLang) => {
    setLangState(l)
    try {
      localStorage.setItem(KEY, l)
    } catch {
      // private mode: the choice lasts for this page only
    }
  }
  return <LangContext.Provider value={{ lang, setLang }}>{children}</LangContext.Provider>
}

export function useLang(): LangCtx {
  return useContext(LangContext)
}

export type Translate = (en: string, vi: string) => string

/** term('Shortlist') → 'Shortlist' in English, 'Shortlist (danh sách rút gọn)' in Vietnamese. */
export function glossed(lang: UiLang, en: string): string {
  return lang === 'vi' && GLOSS[en] ? `${en} (${GLOSS[en]})` : en
}

export interface T {
  t: Translate
  term: (en: string) => string
  lang: UiLang
  /** criterion label: the API's (English) label, or the Vietnamese one */
  crit: (key: string, apiLabel?: string) => string
  /** a data-quality flag such as "duplicate_of:CV-007" */
  flag: (f: string) => string
}

export function useT(): T {
  const { lang } = useLang()
  return {
    t: (en, vi) => (lang === 'vi' ? vi : en),
    term: (en) => glossed(lang, en),
    lang,
    crit: (key, apiLabel) => (lang === 'vi' ? CRITERION_VI[key] : undefined) ?? apiLabel ?? key.replace(/_/g, ' '),
    flag: (f) => {
      const [k, rest] = f.split(/:(.*)/s)
      const name = lang === 'vi' ? FLAG_VI[k] ?? k.replace(/_/g, ' ') : k.replace(/_/g, ' ')
      return rest ? `${name}: ${rest}` : name
    },
  }
}

// Deterministic text the backend writes (statuses, reasons, cards) is English. In Vietnamese the recurring templates
// are translated here; the CV ids, tool names and numbers inside them are kept as they are.
type Replacer = string | ((substring: string, ...groups: string[]) => string)
const BACKEND_VI: [RegExp, Replacer][] = [
  // scoring.knockout_reason: "Missing must-have: x; only hedged mention of y: “…”" (capitalized, ";"-joined)
  [/missing must-have: ([\w.+#-]+)/gi, 'thiếu kỹ năng bắt buộc (must-have): $1'],
  [/only hedged mention of ([\w.+#-]+):/gi, 'chỉ nhắc kiểu "đang học / làm quen" (hedged) về $1:'],
  [/^thiếu/, 'Thiếu'],
  [/^chỉ nhắc/, 'Chỉ nhắc'],
  [/ \(investigated: found (.+); knock-out confirmed\)$/i, ' (đã điều tra: tìm thấy $1; vẫn Knock-out (loại trừ))'],
  [/ \(investigated; knock-out confirmed\)$/i, ' (đã điều tra; vẫn Knock-out (loại trừ))'],
  [/^Not retrieved: relevance rank (\d+) of (\d+) is below the search limit \((\d+)\)$/i,
    'Không được truy hồi: hạng liên quan $1/$2, thấp hơn giới hạn tìm kiếm ($3)'],
  [/^Not shortlisted: rank (\d+) of (\d+) on passage relevance \+ rule score \(shortlist = top (\d+)\)/i,
    'Không vào Shortlist (danh sách rút gọn): hạng $1/$2 theo độ liên quan đoạn CV + điểm luật (shortlist = top $3)'],
  [/; via substitutes for (\[.*\])$/i, '; nhờ công cụ thay thế cho $1'],
  [/^Ranked after direct matches: meets must-haves only via substitutes \((.+)\)$/i,
    'Xếp sau các CV khớp trực tiếp: chỉ đạt yêu cầu bắt buộc nhờ công cụ thay thế ($1)'],
  [/^Assessment failed \(invalid LLM output\) — needs manual review$/i,
    'Đánh giá lỗi (LLM trả kết quả không hợp lệ) — cần HR xem lại thủ công'],
  [/^Near-tie \(Δ=([\d.\-]+)\) resolved by arbiter in both orders: (\S+) — (.+)$/i,
    'Near-tie (gần như hoà điểm, Δ=$1), Arbiter (trọng tài LLM) thống nhất ở cả hai thứ tự: $2 — $3'],
  [/^Placed below (\S+) by arbiter, consistent in both orders \(Δ=([\d.\-]+)\)$/i,
    'Xếp dưới $1 theo Arbiter (trọng tài LLM), nhất quán ở cả hai thứ tự (Δ=$2)'],
  [/^Near-tie (\S+) \/ (\S+) \(Δ=([\d.\-]+)\): the verdict flipped when the order was swapped \(position bias\) → rule order kept$/i,
    'Near-tie (gần như hoà điểm) $1 / $2 (Δ=$3): kết luận đổi khi đảo thứ tự (thiên lệch vị trí) → giữ thứ tự theo luật'],
  // clarification questions written by rules (agent/nodes.py: vagueness_questions, seniority_conflict, ask_hr_gate)
  [/^How many years of test automation experience are required \(not total QA\)\?$/,
    'Yêu cầu bao nhiêu năm kinh nghiệm kiểm thử tự động (không tính tổng số năm làm QA)?'],
  [/^What level of automation ownership is expected: writing test scripts on an existing framework, or building\/maintaining the framework and CI integration\?$/,
    'Kỳ vọng mức độ làm chủ tự động hoá nào: viết test script trên framework có sẵn, hay xây dựng / duy trì framework và tích hợp CI?'],
  [/^Which types of testing: web UI, mobile, API, performance, or a combination\?$/,
    'Những loại kiểm thử nào: web UI, mobile, API, performance (hiệu năng), hay kết hợp?'],
  [/^Is (.+) a hard requirement or just preferred\?$/, '$1 là yêu cầu bắt buộc hay chỉ là ưu tiên?'],
  [/^The JD lists (\d+) must-have tools; which are truly required vs nice-to-have\?$/,
    'JD liệt kê $1 công cụ bắt buộc (must-have); công cụ nào thực sự bắt buộc, công cụ nào chỉ là ưu tiên (nice-to-have)?'],
  [/^Which programming language does the team use for test automation\?$/,
    'Nhóm dùng ngôn ngữ lập trình nào cho kiểm thử tự động?'],
  [/^Which tools\/frameworks are must-have for this role\?$/,
    'Những công cụ / framework nào là bắt buộc (must-have) cho vị trí này?'],
  [/^The title says junior but the JD asks for (\d+)\+ years: which one applies\?$/,
    'Chức danh ghi junior nhưng JD yêu cầu $1+ năm: nên áp dụng điều nào?'],
  [/^The title says senior but the JD asks for only (\d+) year\(s\): which one applies\?$/,
    'Chức danh ghi senior nhưng JD chỉ yêu cầu $1 năm: nên áp dụng điều nào?'],
  // rule-criterion rationales (agent/scoring.py, each ends with " [rule-based]")
  [/ \[rule-based\]$/, ' [tính theo luật]'],
  [/^Must-have demonstrated (\d+)\/(\d+)/, 'Đã chứng minh $1/$2 kỹ năng bắt buộc (must-have)'],
  [/; nice-to-have (\d+)\/(\d+)/, '; ưu tiên (nice-to-have) $1/$2'],
  [/\bmissing ([\w.+#-]+)/g, 'thiếu $1'],
  [/([\w.+#-]+) (listed|hedged) only/g, (_m: string, k: string, s: string) => `${k} chỉ ${s === 'listed' ? 'liệt kê' : 'nhắc kiểu đang học'}`],
  [/^No dated roles and no years stated/, 'Không có vị trí nào ghi thời gian và không nêu số năm'],
  [/^([\d.]+) years in automation roles \(([\d.]+) total\)/, '$1 năm ở vị trí tự động hoá (tổng $2 năm)'],
  [/^L(\d) from the candidate's own action verbs/, 'L$1 suy ra từ động từ hành động của chính ứng viên'],
  [/ \(JD asks L(\d)\)/, ' (JD yêu cầu L$1)'],
  [/^best of (\[.*?\]): credit/, 'tốt nhất trong $1: điểm'],
  [/^no language named by the JD; best language shown: credit/, 'JD không nêu ngôn ngữ; lấy ngôn ngữ tốt nhất trong CV: điểm'],
  [/^no test type named by the JD; best shown/, 'JD không nêu loại kiểm thử; lấy loại tốt nhất trong CV'],
  [/^no CI\/infra named by the JD; best shown/, 'JD không nêu CI / hạ tầng; lấy mức tốt nhất trong CV'],
  [/^JD names no foreign language or domain: neutral/, 'JD không nêu ngoại ngữ hay lĩnh vực: điểm trung lập'],
  [/^No ISTQB certification mentioned/, 'Không nhắc tới chứng chỉ ISTQB'],
  [/ \[unverified: (.+)\]/, ' [chưa kiểm chứng: $1]'],
  [/ \[capped: evidence not found in CV\]/, ' [bị giới hạn điểm: không tìm thấy bằng chứng trong CV]'],
  [/ \[capped: listed, not demonstrated\]/, ' [bị giới hạn điểm: chỉ liệt kê, chưa chứng minh]'],
]

/** Translate one backend template string for the UI language (unknown text is returned unchanged). */
export function backendText(lang: UiLang, text: string | null | undefined): string {
  if (!text || lang === 'en') return text ?? ''
  let out = text
  for (const [rx, vi] of BACKEND_VI) out = out.replace(rx, vi as string)
  return out
}
