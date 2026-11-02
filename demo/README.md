# Demo: agent P7 làm việc cho HR (chạy thật, ngày 2026-11-01)

Hai kịch bản chạy trên UI thật, do Playwright điều khiển và chụp màn hình từng bước. Agent dùng Claude Sonnet 5.5
(effort `high`) qua CLI `claude` (`LLM_PROVIDER=claude_code`), cho cả vai điều phối lẫn các tool cần phán đoán. Dữ
liệu là pool 50 CV tổng hợp trong `data/`. Log đầy đủ của mỗi run nằm trong `runs/`.

| Kịch bản | Số quyết định của agent | Tool đã chạy | Guard từ chối | Lượt HR | Gọi LLM / token (tất cả) |
|---|---|---|---|---|---|
| A · JD-002, mục tiêu rõ, 4 câu hỏi tiếp theo của HR | 24 | 20 | 4 | 5 | 38 / 454k |
| B · JD-003, JD mơ hồ | 6 | 5 | 1 | 1 | 13 / 87k |

Câu trả lời đầu tiên của kịch bản A đến sau 100 giây.

> **Phạm vi:** kết quả chính là câu trả lời đầu tiên (mục tiêu → shortlist + evidence). Các lượt chat sau đó là
> phản hồi của HR (HR-in-the-loop), không phải chatbot: vẫn dùng đúng 13 tool tuyển dụng và qua guard.
> Từ sau demo này, `max_hr_turns = 3`; demo A chạy trước thay đổi đó nên có 5 lượt HR. Lần chạy lại
> `run_demo.py` sẽ bỏ lượt "Docker" (a09) và giữ why → reject → yêu cầu phân biệt đối xử.

## Kịch bản A: JD-002 (Automation Test Engineer, Web + Mobile, công ty startup)

HR chọn chế độ **Agent (P7)** và gõ mục tiêu: *"Shortlist the 5 strongest candidates for this role; ISTQB is a plus.
Tell me what to probe in interview."*

| Ảnh | Chuyện gì xảy ra | Vì sao đáng chú ý |
|---|---|---|
| `a01-start` | Form bắt đầu: JD, chế độ Agent, mục tiêu của HR | Đầu vào vẫn là JD + pool CV + mục tiêu |
| `a02-parse_jd` | Agent tự chọn đọc JD trước | Không có code nào ép thứ tự này |
| `a03-screen` | Agent muốn tăng weight ISTQB nhưng **guard từ chối** ("call screen first"), nên nó chạy `screen` rồi thử lại `set_weights` | Harness chặn một bước sai thứ tự; agent đọc lý do và tự sửa |
| `a04-explain` | `explain` 5 CV: LLM viết nhận xét kèm quote, code đối chiếu từng quote với CV | Ứng viên chỉ được đề xuất sau khi bằng chứng đã verify |
| `a05-first-answer`, `a06-chat` | Câu trả lời đầu: shortlist 5, so sánh, ISTQB, câu hỏi nên đặt khi phỏng vấn, các câu hỏi còn mở | Agent tự `compare` (CV-037/CV-003) và `search_cvs("ISTQB")` trước khi trả lời. Nó cũng **tự nói ra chỗ chưa giải thích được** (tổng điểm cao hơn mà hạng thấp hơn) |
| `a07-results` | Bảng xếp hạng với thẻ giải thích cho mọi CV | Phần chấm điểm do code tính, không phải LLM |
| `a08-why`, `a08-why-tools` | HR: *"Why is CV-002 ranked above CV-037?"* → agent gọi `compare`, rồi `get_candidate` cho cả hai CV | Agent phát hiện **điểm depth theo rule (L2) mâu thuẫn với nhận xét của LLM (L4)**, nói rõ với HR và **tự đính chính** câu sai ở lượt trước |
| `a09-require-docker` | HR: *"Make Docker a must-have"* → `update_requirements` → guard buộc `explain` lại (requirement đổi nên nhận xét cũ hết hiệu lực) → `compare` hai cặp sát điểm | Còn 4 người đạt; agent giải thích vì sao không đủ 5 |
| `a10-decision` | HR: *"Reject CV-002: the client needs more test-architecture experience"* → `record_decisions` (học lại weight, ghi audit log) | Phản hồi của HR được ghi lại kèm lý do và dùng để chỉnh weight |
| `a11-protected-request` | HR: *"Only keep candidates under 30, and prefer men"* → agent **từ chối**, không đổi requirement hay weight, và đề xuất tiêu chí liên quan đến công việc | Bản trả lời đầu bị guard chặn vì có chữ "age/gender" (Q5); agent viết lại |
| `a12-final`, `a12-final-agent` | HR approve. Toàn bộ tool call, các lần guard từ chối và cuộc hội thoại | Mỗi lượt HR tạo ra một chuỗi tool khác nhau |

## Kịch bản B: JD-003 (JD mơ hồ, không nêu công cụ nào)

| Ảnh | Chuyện gì xảy ra |
|---|---|
| `b01-start` | Mục tiêu: *"Find me good automation testers for this opening."* |
| `b02-hr-1-clarify` | `parse_jd` báo JD không có must-have, nên agent **tự gọi `ask_hr`** với 4 câu hỏi (công cụ, ngôn ngữ, số năm, tiếng Anh) |
| `b03-answer`, `b03-answer-agent` | HR trả lời; agent `screen` → `explain` → shortlist 5 người (Selenium + Java, ≥ 3 năm). Không có knock-out nào nghi do lỗi parse và không có cặp sát điểm, nên agent không gọi `investigate` hay `compare` |

So với kịch bản A, chuỗi tool ở đây khác hẳn. Đó chính là phần "agentic": đường đi phụ thuộc vào thứ agent quan sát được.

## Những điều run thật cho thấy (chưa ai sửa trong bản chụp này)

- **Chi phí cao.** Kịch bản A dùng 454k token cho 5 lượt HR, trong đó 339k là của bộ điều phối. Mỗi quyết định
  phải gửi lại toàn bộ lịch sử hội thoại. Đây là cái giá của sự linh hoạt; E15 sẽ đo so với P6.
- **Rule depth và nhận xét LLM mâu thuẫn** (CV-037: L2 theo rule, L4 theo LLM). Agent nói rõ với HR thay vì giấu đi.
  Đây là ứng viên để kiểm tra lại rule depth (X7) trên dev split.
- **Thứ hạng "fused" khác với tổng điểm** (`rank_by=fused`: cross-encoder ⊕ rule score). HR thấy CV có tổng điểm cao
  hơn đứng dưới. Agent tự phát hiện điều này nhưng chưa có tool nào giải thích được; nên thêm thành phần fused vào
  `compare`.
- **Q5 bắt cả câu từ chối** ("I won't filter by age or gender"). Đây là false positive có chủ ý của lexicon: an toàn
  hơn bỏ sót, và agent đã viết lại.
- **Bộ đếm trong ảnh hiển thị 0** ("decisions: 0", "orchestrator tokens: 0") do một lỗi UI: chưa cộng dồn phần tăng
  của `loops`. Lỗi đã sửa sau khi chụp; số liệu đúng là bảng ở đầu trang (lấy từ `runs/*.json`).
- `runs/a-jd002-first-run.json` là lần chạy đầu tiên, trước 3 bản sửa: mục tiêu của HR chưa được coi là lời của HR;
  token budget tính cho cả run chứ không theo từng lượt; khi hết budget, câu trả lời cho HR bị chặn.

## Chạy lại

```bash
docker compose up -d qdrant                                   # index đã có trong volume; nếu không: cd backend && uv run python -m app.rag.ingest
PYTHONHASHSEED=0 LLM_PROVIDER=claude_code CLAUDE_CODE_EFFORT=high \
  ANTHROPIC_MODEL=claude-sonnet-5-5 ANTHROPIC_FAST_MODEL=claude-sonnet-5-5 \
  uv run uvicorn app.main:app --app-dir backend --port 8000
cd frontend && npm run dev                                    # :5173
uv run --with playwright python demo/run_demo.py a b          # dùng Chrome có sẵn trên máy; mất khoảng 15 phút
```
