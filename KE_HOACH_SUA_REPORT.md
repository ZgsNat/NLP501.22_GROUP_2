# Kế hoạch sửa report sau buổi final

---

## 1. Vì sao buổi final không đạt

**Pipeline không phải vấn đề.** Kiến trúc agentic (P7 điều phối, P6 xếp hạng, guard kiểm soát) chạy được, demo được,
và trên tinh thần học tập là ổn. Phần này giữ nguyên.

**Vấn đề nằm ở số liệu.** Gần như mọi con số trên slide v13 đều đến từ một vòng khép kín: **dữ liệu do máy sinh, nhãn do
máy hoặc script đặt, rồi chính hệ thống tự đo trên đó.** Vì vậy khi cô hỏi "data thật ở đâu, ai chấm, chấm thế nào",
không ai trả lời được, và nhóm không đưa ra được một bộ dữ liệu nào do **con người** chuẩn bị (CV PDF, DOCX thật của
người thật, không phải các file markdown tổng hợp).

Hệ quả trên slide:
1. **Không có ý nghĩa học thuật.** Không cho thấy bộ dữ liệu đánh giá trông thế nào (bao nhiêu mẫu, lấy từ đâu, ai gán nhãn,
   đồng thuận bao nhiêu), cũng không cho thấy phương pháp đánh giá.
2. **Không rõ ràng.** Con số quá chung ("0.978", "89%", "+0.214") mà không kèm "trên dữ liệu gì, so với cái gì", nên người
   nhìn không hiểu.

## 2. Nhận xét của cô, viết lại thành yêu cầu

| Cô nói | Nghĩa là nhóm phải |
|---|---|
| 1. Chưa trung thực | Bỏ mọi mô tả không đúng với dữ liệu (xem mục 3). Mỗi con số phải ghi rõ dữ liệu và người gán nhãn |
| 2. Chưa có con người review | Mỗi bộ đánh giá phải có người thật gán nhãn hoặc kiểm tra, có bằng chứng (thời điểm chấm, ghi chú) |
| 3. Chưa có dữ liệu đánh giá | Mỗi phần của pipeline có **một bộ dữ liệu đánh giá riêng**, mô tả được (data card) |
| 4. Phải có bộ data chuẩn xác cao | Dữ liệu do con người chuẩn bị, có nguồn rõ ràng, khoá lại trước khi chạy hệ thống |
| 5. Từ 2 người trở lên đánh giá, tính kappa | Mỗi bộ nhãn: **≥ 2 người chấm độc lập**, báo cáo **Cohen's κ** trước khi phân xử; người thứ 3 phân xử chỗ lệch |
| 6. OCR cũng phải gán nhãn | OCR cần bộ ảnh scan/chụp thật với văn bản đúng do người kiểm |

**Nguyên tắc chung:** với **từng bước** của pipeline agentic (goal HR, OCR, trích xuất, che PII, xếp hạng P6, hành vi
agent P7), phải có một bộ dữ liệu: có đáp án chính xác, có con người review, có cấu trúc rõ ràng để đo.

## 3. Những lỗi đã kiểm ra trong số liệu hiện tại

Đây là kết quả kiểm tra lại toàn bộ số trên slide v13 (chạy lại ngày 08/11). Ghi ra để không lặp lại, không để quy lỗi.

| # | Lỗi | Bằng chứng |
|---|---|---|
| 1 | **Nhãn ours (600 cặp) do 2 LLM chấm** (Opus, Sonnet), dùng làm "gold" cho benchmark chính | `data/eval/annotator_a.yaml`, `annotator_b.yaml` |
| 2 | **Đáp án extraction, OCR, PII đều do bộ sinh dữ liệu tạo**, trên 200 CV tổng hợp viết từ mẫu câu; regex depth được viết khi đã thấy dữ liệu này | `.runtime/bench/coding/latent.json` |
| 3 | **Ảnh "scan" OCR là ảnh code tự vẽ** từ văn bản tổng hợp, chỉ CV một cột | `backend/app/eval/ocr.py` |
| 4 | **Slide ghi nhãn Djinni là "double blind" là sai**: một trong hai bộ nhãn (424 cặp) được chấm trong 13 phút, 1,4 giây/cặp, ở chế độ **thấy điểm của mô hình**; cờ `saw_model` sau đó bị đổi thành 0 | `data/eval/human_djinni_dan.json` (thời điểm từng nhãn), lịch sử git |
| 5 | **120 cặp nhãn người trên ours không có bằng chứng chấm mù**: không ghi thời điểm chấm; file chứa sẵn điểm silver của từng cặp nằm trong repo, commit cùng lúc; một người trùng nhãn Opus 94/120 | `data/eval/human_hoa.json`, `human_phuc.json`, `labeling_reference/pairs_reference.json` |
| 6 | **"IAA κ = 0.818" của benchmark trích xuất Djinni là số do một script regex sinh cả hai bộ "nhãn người"**. Đã được phát hiện và sửa trên nhánh `fix/extraction-human-gold-gate`, nhưng nhánh đó còn lỗi cú pháp ở `nodes.py` (làm agent không chạy) và sửa tay file kết quả | nhánh `fix/extraction-human-gold-gate` |
| 7 | **Số trên slide 15 không tái lập được trên máy khác**: bảng xếp hạng và cache yêu cầu JD (do LLM trích) chỉ nằm trong `.runtime/` của một máy | chạy lại ra B3 0.838 (slide 0.841), P6 0.847 so với B4 0.848 (slide 0.851 so với 0.839) |
| 8 | **P6 trên Djinni chạy bằng Gemini**, các phần khác bằng Claude; slide không ghi | `docs/DJINNI_PRODUCTION_RESULTS.md` |
| 9 | **Đánh giá agent (E15, E26, H6) dùng HR theo kịch bản** và bộ mục tiêu do nhóm tự viết; không có người thật đánh giá output | `data/goal_suite/goals.yaml`, `backend/app/eval/agentic.py` |
| 10 | **Slide sai vặt**: "37 kỹ năng" (thực tế 44); OCR 0.997 là số tập phát triển (tập giữ riêng 0.981) | `data/skills_taxonomy.yaml`, `docs/OCR_RESULTS_heldout.md` |

## 4. Quy tắc chung cho mọi bộ dữ liệu mới (bắt buộc)

**Nguồn dữ liệu**
- CV thật: **PDF, DOCX, ảnh scan, ảnh chụp** của người thật, có **đồng ý bằng văn bản** theo `docs/labeling/CONSENT.md`.
  File gốc chỉ để ở `.runtime/consented/raw/` (không vào git), dùng bản đã che thông tin để gán nhãn.
- Dữ liệu công khai có giấy phép (Djinni, MIT) được dùng, nhưng **nhãn phải do người chấm**.
- JD thật: lấy từ trang tuyển dụng (ghi nguồn, ngày lấy).
- **Không dùng dữ liệu tổng hợp cho số chính.** Dữ liệu tổng hợp chỉ được nhắc như "tập phát triển / regression test".

**Gán nhãn**
- **≥ 2 người độc lập**, không thấy output của hệ thống, không thấy nhãn của nhau, không thấy nhãn LLM.
- Mỗi nhãn lưu: người chấm, **thời điểm**, có ghi chú khi khó, cờ `saw_model` trung thực. Mẫu:
  ```json
  {"annotator": "A1", "item": "cv_017", "field": "tools", "label": ["selenium","java"], "at": "2026-11-10T09:12:03",
   "saw_model": false, "note": "Selenium chỉ liệt kê ở Skills"}
  ```
- Tính **κ trước khi phân xử** (Cohen's κ cho nhãn loại; κ có trọng số bậc hai cho thang 0–3; F1 hoặc token-IoU ≥ 0.5 cho
  span). Ngưỡng ghi trước khi chấm: κ ≥ 0.6 chấp nhận được, ≥ 0.8 tốt; dưới 0.6 thì họp thống nhất hướng dẫn rồi chấm lại.
- **Người thứ 3 phân xử** chỉ những chỗ lệch. Gold = nhãn sau phân xử.
- **LLM không bao giờ được tính là người chấm.** Nếu dùng LLM để tham khảo thì báo cáo ở cột riêng.

**Khoá dữ liệu và tái lập**
- Mỗi bộ có **data card** (mục 6) và **manifest SHA-256**; khoá trước khi chạy hệ thống. Chuẩn hoá xuống dòng trước khi tính hash.
- Chạy mọi số chính **trên một máy**, với cùng cấu hình: `PYTHONHASHSEED=0`, `REFERENCE_DATE=2026-10-30`, model ghi rõ.
- **Commit kết quả xếp hạng và cache yêu cầu JD** cần để tái lập (không chứa PII), để ai cũng chạy lại ra đúng số.
- File kết quả `docs/*_RESULTS.md` chỉ do script sinh, **không sửa tay**.

## 5. Việc cần làm theo từng phần của pipeline

Mỗi phần ghi: **hiện tại** · **bộ dữ liệu cần chuẩn bị** · **gán nhãn** · **độ đo** · **công cụ** · **trình bày trên slide**.

### 5.1 Đầu vào: hiểu mục tiêu của HR (P7: `formalize_goal`, `define_concept`, `check_goal`, `ask_hr`)

- **Hiện tại:** 5 mục tiêu nhóm tự viết, HR theo kịch bản, chỉ đo "có gọi đúng tool không" (81/83). Không đo được LLM hiểu
  mục tiêu đúng bao nhiêu phần trăm.
- **Bộ dữ liệu:** tối thiểu **40 câu mục tiêu HR viết bằng lời** (Việt và Anh), do người viết (thành viên nhóm, và nếu được
  thì người làm HR thật), trên các JD có sẵn. Phủ đủ các dạng: rõ và khả thi, không khả thi, mơ hồ, có loại trừ, có khái niệm
  ngoài danh mục ("dùng AI trong automation"), có số lượng, không có số lượng.
- **Gán nhãn (2 người độc lập)**, cho mỗi câu, ghi đáp án mong đợi:
  - danh sách ràng buộc: loại (bắt buộc / nên có / điểm cộng / loại trừ) + mục tiêu (kỹ năng hoặc khái niệm);
  - số lượng HR muốn (hoặc "không nói");
  - **có nên hỏi lại HR không** (có / không) và vì sao;
  - với kho CV hiện tại: khả thi hay không.
- **Độ đo:** F1 trên ràng buộc (đúng cả loại và mục tiêu); độ chính xác số lượng; precision và recall của quyết định "hỏi
  lại HR"; tỉ lệ ràng buộc trích đúng lời HR (Q19); độ chính xác kết luận khả thi; κ giữa hai người cho từng trường.
  Chạy mỗi câu **3 lần** để báo độ ổn định.
- **Công cụ:** `backend/app/eval/goals.py` (E26) chạy cả lượt agent; **cần viết thêm** một script chỉ gọi
  `formalize_goal` + `check_goal` trên 40 câu rồi so với đáp án (rẻ và nhanh hơn chạy cả lượt).
- **Slide:** "40 câu mục tiêu do người viết, 2 người gán đáp án (κ = …). Agent trích đúng ràng buộc F1 = …, quyết định
  hỏi lại đúng … %". Kèm 1 ví dụ mục tiêu và đáp án.

### 5.2 Đọc JD (`nlp/jd.py`, parser hybrid)

- **Hiện tại:** đo trên 6 JD tổng hợp, đáp án là cách bộ sinh tự đọc JD.
- **Bộ dữ liệu:** **20–30 JD thật** (Djinni + trang tuyển dụng Việt Nam), cả tiếng Anh và tiếng Việt.
- **Gán nhãn (2 người):** công cụ bắt buộc, công cụ nên có, số năm tối thiểu, mức L1–L4 yêu cầu, ISTQB, ngoại ngữ.
- **Độ đo:** P/R/F1 cho must-have và nice-to-have; độ chính xác số năm và mức yêu cầu; κ.
- **Công cụ:** phần JD của `backend/app/eval/extraction.py` đang so với `gold_spec`; **cần sửa** để đọc file nhãn người.

### 5.3 OCR (CV scan / ảnh chụp, `nlp/ocr.py`)

- **Hiện tại:** ảnh do code vẽ từ CV tổng hợp một cột.
- **Bộ dữ liệu:** **20–30 tài liệu thật**: in CV thật (đã đồng ý) hoặc mẫu CV phổ biến rồi **scan bằng máy và chụp bằng điện
  thoại** trong điều kiện thường (nghiêng, thiếu sáng). Có cả CV **hai cột, có bảng, có ảnh đại diện**, cả tiếng Việt.
- **Đáp án:** văn bản đúng của từng trang. Nếu CV được in từ file số đã biết thì lấy văn bản file số làm đáp án, **người
  kiểm lại** từng trang; nếu không có file số thì **người gõ lại**. 2 người kiểm, ghi chỗ lệch.
- **Độ đo:** CER và WER so với văn bản đúng; thứ tự đọc đúng ở CV hai cột; và **ảnh hưởng xuống dưới**: F1 công cụ, số năm,
  PII bị lộ khi đọc qua OCR so với khi đọc từ file số.
- **Công cụ:** `backend/app/eval/ocr.py` có sẵn phần chấm CER và chạy pipeline; **cần sửa** để đọc danh sách ảnh thật +
  đáp án thay vì tự vẽ ảnh.
- **Slide:** "25 tài liệu scan và chụp thật (… hai cột), CER = …%; công cụ nhận đúng … % so với bản số".

### 5.4 Trích xuất thông tin và che PII từ CV (`nlp/extract.py`, `nlp/redact.py`)

- **Hiện tại:** F1 0.978 trên 200 CV tổng hợp, đáp án do bộ sinh; depth 89% có nguy cơ "học theo đề".
- **Bộ dữ liệu:**
  - **50 hồ sơ Djinni thật** đã khoá trên nhánh `fix/extraction-human-gold-gate` (12 thử + 50 test), có sẵn quy trình gán
    nhãn và bộ kit HTML;
  - **cộng ít nhất 10–20 CV PDF/DOCX thật có đồng ý**: bắt buộc cho phần **PII**, vì hồ sơ Djinni đã ẩn danh nên không đo
    được che thông tin.
- **Gán nhãn (2 người, theo `docs/labeling/EXTRACTION_GUIDELINES.md` trên nhánh đó):** từng lần nhắc kỹ năng (vị trí trong
  văn bản, kỹ năng chuẩn hoặc "ngoài danh mục", trạng thái *dùng thật / liệt kê / rào đón*); số năm kinh nghiệm; mức L1–L4;
  ISTQB; ngoại ngữ. Với CV có đồng ý: đánh dấu mọi PII (tên, email, SĐT, địa chỉ, ngày sinh, giới tính).
- **Độ đo:** P/R/F1 kỹ năng (khớp span bằng token-IoU ≥ 0.5); độ chính xác trạng thái bằng chứng; sai số số năm;
  độ chính xác depth **so với mốc đoán mặc định**; **tỉ lệ lộ PII theo từng trường**; κ cho từng trường.
- **Công cụ:** trên nhánh `fix/extraction-human-gold-gate`: `djinni_extraction_annotation.py prepare-kits` → hai bộ kit;
  lệnh `iaa` tính κ và tạo kit phân xử; `extraction_benchmark.py` chấm E0–E2. **Trước khi merge nhánh:** bỏ thay đổi ở
  `backend/app/agent/nodes.py` (lỗi cú pháp làm agent không chạy), trả `docs/EXTRACTION_RESULTS.md` về bản do script sinh,
  đưa phần mới sang `docs/DJINNI_EXTRACTION_BENCHMARK.md`.
- **Slide:** "50 hồ sơ thật + … CV có đồng ý, 2 người gán nhãn (κ kỹ năng = …, κ depth = …). F1 kỹ năng = … ; lộ PII … /
  … trường". Ghi 0.978 trên dữ liệu tổng hợp chỉ như số tập phát triển.

### 5.5 Xếp hạng: retrieval + chấm điểm (P6, benchmark)

- **Hiện tại:** số chính (0.851, …) trên nhãn LLM (ours) và nhãn bộ sinh (coding); nhãn Djinni có một bộ thấy điểm mô hình;
  số không tái lập được.
- **Bộ dữ liệu:**
  - **Holdout Djinni đã khoá: 6 JD × 40 CV = 240 cặp**, chưa chạy hệ thống nào → đây là **tập test sạch nhất** hiện có;
  - **424 cặp Djinni production**: chấm lại bằng **2 người độc lập** (thay bộ nhãn đã thấy điểm mô hình);
  - nếu kịp: các CV có đồng ý ở 5.4 ghép với 3–5 JD thật.
- **Gán nhãn (2 người + người phân xử):** thang 0–3 (0 không phù hợp, 1 yếu, 2 đáng gọi phỏng vấn sơ bộ, 3 đáng phỏng vấn),
  theo hướng dẫn viết cho HR chứ không viết theo luật của hệ thống.
- **Độ đo:** nDCG@10, P@5, Kendall τ, CI 95% bootstrap; so sánh B0 BM25, B2 hybrid, B3 luật ⊕ CE, B4, P6. κ trọng số bậc hai.
- **Công cụ:** `uv run --with pyarrow python -m app.eval.djinni_holdout prepare` (tạo kit), `import-labels A.json B.json`
  (kiểm tra và tính κ; chưa đủ nhãn thì không cho chấm); `python -m app.eval.djinni_gold all`;
  `python -m app.eval.run retrieval|agent --bench …`; `python -m app.eval.benchmark report`.
- **Tái lập:** chạy trên một máy, ghi model (không trộn Gemini và Claude), **commit kết quả xếp hạng**.
- **Slide:** bảng có tên hệ thống đầy đủ, kèm dòng "240 cặp, 6 JD thật, 2 người chấm độc lập (κ = …)". Không khẳng định
  P6 hơn B4 nếu CI chồng nhau.

### 5.6 Hành vi agent P7 (cả cuộc hội thoại với HR)

- **Hiện tại:** HR theo kịch bản, code kiểm "có gọi đúng tool không"; không có người đánh giá kết quả cuối.
- **Bộ dữ liệu:** **20–30 kịch bản hội thoại** (mục tiêu ở 5.1 + 1–2 câu hỏi tiếp theo: "vì sao", "bắt buộc thêm X",
  "loại CV Y", yêu cầu phân biệt đối xử). Chạy agent, **lưu toàn bộ hội thoại và shortlist**.
- **Gán nhãn (2 người chấm output, không biết lượt nào là lượt nào):** theo rubric:
  - shortlist có đáp ứng mục tiêu không (đạt / một phần / không);
  - có hỏi HR đúng lúc không (hỏi thừa / thiếu / đúng);
  - phần chưa đạt có được nói rõ và đúng không;
  - lời giải thích có bằng chứng thật không;
  - có từ chối đúng yêu cầu phân biệt đối xử không.
- **Độ đo:** tỉ lệ đạt cho từng tiêu chí (kèm khoảng tin cậy Wilson), κ giữa hai người chấm; so sánh với kiểm tra tự động
  của code (code đúng bao nhiêu so với người).
- **Nếu kịp:** 3–5 người dùng thử (người làm HR hoặc ngoài nhóm), phiếu 3–5 câu (kết quả có đúng ý, giải thích có tin được,
  có dùng lại không).
- **Công cụ:** `backend/app/eval/agentic.py` và `goals.py` chạy và lưu vết; **cần viết** phần xuất hội thoại thành phiếu
  chấm cho người.

### 5.7 Lời giải thích và guardrail (Q2, Q4, Q6)

- **Hiện tại:** "100% trích dẫn có thật", "2.3% → 1.5%" đều do code tự kiểm code.
- **Bộ dữ liệu:** **100 lời giải thích** lấy ngẫu nhiên từ các lượt chạy ở 5.5 và 5.6.
- **Gán nhãn (2 người):** mỗi lời giải thích: trích dẫn có trong CV không; trích dẫn có thật sự chứng minh tiêu chí không;
  có nhận định sai về ứng viên không.
- **Độ đo:** tỉ lệ lỗi thật theo người chấm; **precision và recall của bộ kiểm tra tự động** so với người; κ.
- **Slide:** "Người kiểm 100 lời giải thích: … % có lỗi; bộ kiểm tra tự động bắt được … % trong số đó".

### 5.8 Công bằng và prompt injection

- **Công bằng (bản sinh đôi):** giữ, nhưng **nói đúng nghĩa**: chứng minh bước che thông tin hoạt động, không chứng minh mô hình
  không thiên lệch. Thêm bản sinh đôi trên CV thật (5.4) nếu kịp.
- **Injection:** bộ tấn công hiện do người phát triển tự viết. Cần **20–30 câu tấn công do người khác trong nhóm viết**
  (không xem bộ lọc), cài vào CV thật; đo tỉ lệ chặn và tỉ lệ báo nhầm trên CV sạch.

## 6. Data card: mỗi bộ dữ liệu phải có

Đặt cạnh dữ liệu (`docs/datasets/<tên>.md`) và tóm tắt trên slide:

| Mục | Ví dụ |
|---|---|
| Mục đích | Đo trích xuất kỹ năng từ CV thật |
| Nguồn | Djinni (MIT), 50 hồ sơ; 15 CV PDF có đồng ý |
| Cách chọn mẫu | ngẫu nhiên có seed, tách khỏi tập phát triển (khác ID và hash) |
| Kích thước | 65 tài liệu, … nhãn |
| Định dạng | PDF / DOCX / ảnh / text |
| Người gán nhãn | 2 người độc lập + 1 người phân xử; không thấy output hệ thống |
| Hướng dẫn | đường dẫn file hướng dẫn |
| Đồng thuận | κ = … (trước phân xử), số chỗ phân xử |
| Khoá | manifest SHA-256, ngày khoá |
| Giới hạn | ví dụ: chỉ tiếng Anh, chỉ QA Automation |

## 7. Sửa report và slide

- **Mỗi con số trên slide đi kèm 3 thông tin:** dữ liệu gì (n, nguồn), đáp án do ai đặt (κ), độ đo là gì (định nghĩa một
  dòng). Không đưa số nào không trả lời được ba câu đó.
- **Mỗi phần pipeline một slide kết quả**, theo thứ tự 5.1 → 5.7, có ảnh minh hoạ một mẫu dữ liệu và một mẫu nhãn.
- **Số trên dữ liệu tổng hợp** chỉ để ở phần "kiểm thử phát triển", ghi rõ "dữ liệu tổng hợp, đáp án do bộ sinh".
- **Bỏ hoặc sửa** mọi mô tả ở mục 3 (double blind, blind, IAA 0.818, 37 kỹ năng, OCR 0.997, P6 hơn B4).
- **Trước khi nộp:** một người không làm phần đó đọc slide và hỏi lại "số này từ đâu"; ai trình bày phần nào phải trả lời
  được ngay, mở được file dữ liệu và file nhãn.
- `python -m app.eval.defense_numbers` (E31) đang ghi nguồn cho từng số của deck cũ; **cập nhật** cho số mới để mỗi số có
  công thức, file nguồn và lệnh chạy.

## 8. Checklist

**Dữ liệu**
- [ ] Thu CV thật có đồng ý (PDF, DOCX, ảnh), theo `docs/labeling/CONSENT.md`
- [ ] 20–30 JD thật
- [ ] 40 câu mục tiêu HR do người viết
- [ ] 20–30 tài liệu scan / chụp thật (có CV hai cột)
- [ ] 20–30 câu tấn công injection do người khác viết
- [ ] Data card + manifest cho từng bộ

**Gán nhãn (mỗi bộ ≥ 2 người độc lập, ghi thời điểm, có người phân xử)**
- [ ] Mục tiêu HR (5.1)
- [ ] JD (5.2)
- [ ] Văn bản đúng cho OCR (5.3)
- [ ] Trích xuất + PII: 50 hồ sơ Djinni + CV có đồng ý (5.4)
- [ ] Xếp hạng: holdout 240 cặp + chấm lại 424 cặp (5.5)
- [ ] Output agent: 20–30 hội thoại (5.6)
- [ ] 100 lời giải thích (5.7)

**Code (chỉ phần phục vụ đánh giá; pipeline giữ nguyên)**
- [ ] Sửa nhánh `fix/extraction-human-gold-gate` (bỏ thay đổi `nodes.py`, không sửa tay file kết quả) rồi merge
- [ ] Script chấm `formalize_goal` / `check_goal` trên 40 câu mục tiêu
- [ ] `extraction.py` đọc nhãn người cho JD
- [ ] `ocr.py` đọc ảnh thật + đáp án
- [ ] Xuất hội thoại agent và lời giải thích thành phiếu chấm cho người
- [ ] E33: thêm đường dẫn font macOS; manifest chuẩn hoá xuống dòng (hiện lệch 2.467 file trên Mac/Linux)

**Chạy lại và báo cáo**
- [ ] Chạy tất cả trên một máy, một model, cấu hình cố định; commit kết quả xếp hạng
- [ ] Cập nhật `defense_numbers` cho số mới
- [ ] Viết lại slide và report theo mục 7
- [ ] Đọc chéo: mỗi người hỏi người khác "số này từ đâu"
