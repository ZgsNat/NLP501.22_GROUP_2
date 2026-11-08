# Script thuyết trình · NLP501 Nhóm 2 · deck v13

Deck: `NLP501_Nhom2_BaoCaoCuoiKy_v13.pptx` (24 slide trình bày + 1 slide phụ lục, khoảng 23 phút). Mỗi người nói một khối slide liền nhau, theo mạch nội dung: bài toán → giải pháp → dữ liệu và kết quả → thảo luận. Bài toán được phát biểu lại ở slide 2 vì đã thay đổi để trở thành agentic. Số liệu khớp với slide và [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md).

| Thứ tự | Người nói | Slide | Nội dung | Thời lượng |
|---|---|---|---|---|
| 1 | Hoàng Xuân Sơn | 1–4 | bài toán, khoảng trống, tiêu chí chấm điểm | 3:35 |
| 2 | Nguyễn Thái Thịnh | 5–9 | kiến trúc, NLP pipeline, retrieval | 4:45 |
| 3 | Nguyễn Đình Đán | 10–12 | agent P7, P6, guardrails | 3:30 |
| 4 | Trần Việt Hoa | 13–18 | dữ liệu, kết quả thực nghiệm | 7:00 |
| 5 | Nguyễn Văn Vũ | 19–24 | nghiên cứu liên quan, giả thuyết, hạn chế, kết luận | 5:00 |

Thuật ngữ kỹ thuật giữ nguyên tiếng Anh như trên slide. Câu chuyển lời chỉ có ở cuối mỗi khối.

---

## 01 · Bài toán và tiêu chí

### Slide 1 · Tiêu đề · Sơn · 0:20

Kính thưa Hội đồng, nhóm 2 trình bày hệ thống Agentic AI xếp hạng và giải thích mức độ phù hợp giữa CV và JD cho vị
trí Automation Test Engineer: điểm số do luật xác định, LLM có guardrail, quyết định thuộc về HR.

### Slide 2 · Bài toán: từ xếp hạng theo JD đến làm việc theo mục tiêu của HR · Sơn · 1:30

Bài toán của nhóm đã thay đổi để trở thành agentic, nên cần phát biểu lại.

- **Trước: xếp hạng theo JD.** Đầu vào là JD và kho CV, đầu ra là danh sách xếp hạng. Khó khăn là keyword ATS không
  thấy automation depth (chạy, viết, xây framework hay thiết kế kiến trúc kiểm thử, L1–L4), không phân biệt công cụ đã
  dùng trong dự án với công cụ chỉ liệt kê, và bị đánh lừa bởi hedging như "familiar with", "đang tự học".
- **Nay: mục tiêu của HR bằng lời.** Ví dụ "5 CV, bắt buộc Selenium, có Playwright càng tốt, dùng AI trong automation
  là điểm cộng lớn". Mục tiêu như vậy trộn ràng buộc cứng và mềm, chứa khái niệm không có trong danh mục kỹ năng, và có
  thể không khả thi trên kho CV hiện có. HR còn hỏi tiếp: "vì sao #1 trên #2?", "bắt buộc Docker".
- **Vì vậy hệ thống phải agentic:** hiểu mục tiêu, kiểm tra kho CV có đáp ứng được không, hỏi lại HR khi cần, hành
  động bằng công cụ, tự kiểm kết quả và nêu rõ phần chưa đạt.

Đầu ra: danh sách có điểm theo tám tiêu chí và trích dẫn đã kiểm chứng; mọi CV khác có trạng thái và lý do; phần mục
tiêu chưa đạt được nêu rõ. Ràng buộc: không dùng thông tin định danh, HR quyết định.

Điểm số vẫn tính bằng luật theo công thức dưới slide: H là knock-out theo công cụ bắt buộc, w là trọng số theo loại
doanh nghiệp, S là điểm 0–100 của tám tiêu chí. Cái mới không nằm ở công thức mà ở chỗ agent chịu trách nhiệm về mục
tiêu của HR.

### Slide 3 · Khoảng trống nghiên cứu G1–G5 · Sơn · 0:45

Giải pháp nhắm vào năm khoảng trống rút ra từ tổng quan tài liệu: G1 đặc thù tiêu chí của từng doanh nghiệp; G2 nghịch
lý từ vựng, với kiểm thử phần mềm keyword ATS vượt embedding (Resume2Vec); G3 LLM hallucination; G4 chi phí khi dùng
LLM cho mọi ứng viên (RankGPT); G5 data leakage khi test split nhỏ và nhãn do chính LLM gán. Cuối bài nhóm đánh giá mức
độ giải quyết từng khoảng trống.

### Slide 4 · Rule-based scoring và explanation card · Sơn · 1:00

Ví dụ là JD-010 (ngân hàng) và CV-033. Mỗi dòng là một tiêu chí: trọng số, điểm, đóng góp và bằng chứng trích từ CV.
Công cụ: 3/3 yêu cầu bắt buộc, đóng góp 0.184. Depth L3, xác định từ động từ của chính ứng viên. CI/CD đạt 90 vì
Jenkins chỉ được liệt kê, tính 0.7. Tổng 0.924.

Explanation card được tính từ chính các luật chấm điểm, nên luôn nhất quán với thứ hạng; LLM không thể giải thích lệch
với điểm. Trọng số theo loại doanh nghiệp, tổng bằng 1: gia công phần mềm nâng lĩnh vực và tiếng Anh lên 0.20, ngân
hàng tăng ISTQB, startup ưu tiên loại kiểm thử và CI/CD. Trọng số có thể hiệu chỉnh từ quyết định của HR bằng pairwise
logistic regression. Với CV bị loại, what-if analysis cho HR thấy ứng viên còn cách ngưỡng bao xa. Mời bạn Thịnh trình bày các kỹ thuật NLP.

---

## 02 · Giải pháp

### Slide 5 · Kiến trúc tổng thể · Thịnh · 1:00

Luồng chính đi từ trái sang phải. CV và JD được NLP pipeline X1–X11 xử lý, hoàn toàn tất định, không dùng LLM. Hybrid
retrieval tìm ứng viên, rule-based scoring tính knock-out H và tám điểm S, rồi trả về HR danh sách xếp hạng kèm lý do và
trạng thái.

LLM chỉ làm các việc cần phán đoán: hỏi lại HR khi JD mơ hồ, đọc CV bị lỗi ký tự, phân xử cặp near-tie theo hai thứ tự,
sinh giải thích có trích dẫn, và điều phối theo mục tiêu của HR ở P7. Mọi output của LLM đi qua harness gồm schema check,
precondition, HR authority, 20 tiêu chí và budget về bước và token.

Lý do thiết kế đến từ số đo: ở v2, khi LLM đặt 5 trên 8 điểm S, nDCG@10 giảm 0.033. Từ v3, luật xác định cả tám điểm.

### Slide 6 · NLP pipeline cho CV · Thịnh · 0:45

Pipeline gồm các bước: đọc tệp (pypdf, python-docx, PaddleOCR), Unicode normalization, PII redaction, lọc prompt
injection, sectioning và date grammar, structural chunking, nhận diện kỹ năng kèm hedge và depth, cuối cùng là
CVProfile.

Nguyên tắc là CV phải được xử lý như văn bản bán cấu trúc chứ không phải túi từ. Cùng là chữ "Selenium": trong mục Kinh
nghiệm là bằng chứng sử dụng, trong mục Kỹ năng là tự nhận, sau "familiar with" là hedge, còn trong "our developers run
Selenium" là việc của người khác. Vì vậy cấu trúc được khôi phục trước, và mọi bước sau chỉ đọc CVProfile, không đọc văn
bản thô.

### Slide 7 · Preprocessing song ngữ · Thịnh · 0:45

- **NFC normalization:** PDF lưu tiếng Việt ở dạng tổ hợp, chữ "ê" có thể là e cộng dấu mũ cộng dấu sắc. Không chuẩn
  hoá thì "kiểm thử" trích từ PDF và từ form web là hai chuỗi khác nhau, khiến BM25, regex và kiểm tra trích dẫn sai mà
  không báo lỗi.
- **PII redaction:** xoá theo luật các dòng nhãn–giá trị (họ tên, ngày sinh, giới tính, hôn nhân, tôn giáo), email,
  URL, số điện thoại, và tên đứng một mình ở đầu CV. Khoảng năm như "2019.05–2021" không bị nhận nhầm là số điện thoại.
- **Prompt injection:** câu nhắm vào mô hình bị thay bằng [REDACTED] và CV bị gắn cờ; một classifier làm lớp thứ hai.
- **Sectioning và date grammar:** khoảng 90 biến thể tiêu đề Anh–Việt; date grammar xử lý "Jan 2020 – Present",
  "03/2021 – nay", "Tháng 6/2019 – Tháng 2/2021", các khoảng chồng lấn được hợp nhất.
- **Language detection và OCR:** ngưỡng 35% từ đặc trưng tiếng Việt; ảnh quét qua PaddleOCR, tiêu đề và nhãn được so
  khớp ở dạng không dấu.

### Slide 8 · Information extraction · Thịnh · 1:15

- **Gazetteer-based NER:** 37 kỹ năng và các biến thể, so khớp theo alphanumeric boundary nên "java" không khớp trong
  "javascript". Mỗi lần xuất hiện có evidence strength: đã dùng trong vai trò hoặc dự án là 1.0, chỉ liệt kê là 0.7,
  hedged là 0.2. Tên công cụ là tập từ vựng đóng, nên gazetteer chính xác hơn statistical NER.
- **Hedge scope theo NegEx:** NegEx vốn phát hiện phủ định trong văn bản y khoa; nhóm mở rộng cho tiếng Anh và tiếng
  Việt. Từ báo hiệu đứng trước như "familiar with", "đang tự học" có phạm vi đến hết câu hoặc đến "but / nhưng"; từ báo
  hiệu đứng sau chỉ tác động trong segment của nó: "Java (basic), Selenium" chỉ hedge Java.
- **Automation depth L1–L4:** suy từ động từ đầu mệnh đề do chính ứng viên thực hiện. "Executed the existing automated
  suite" là L1 vì "automated" là tính từ; "Built a POM framework" là L3; chức danh như "Automation Architect" không được
  tính.

Tác dụng chống keyword stuffing: một yêu cầu bắt buộc chỉ xuất hiện ở dạng hedge thì không qua knock-out. Tỉ lệ trap CV
lọt top 10 là 0.00 khi có xử lý hedge, 0.17 khi không có.

### Slide 9 · Hybrid retrieval và reranking · Thịnh · 1:00

Retrieval gồm năm bước:

1. BM25 bắt chính xác tên công cụ như RestAssured, k6, Appium.
2. Dense embedding bge-small-en-v1.5 bắt ngữ nghĩa: "test harness" gần với "framework kiểm thử".
3. Reciprocal Rank Fusion gộp hai danh sách.
4. Mỗi requirement là một query riêng, có mở rộng biến thể; yêu cầu bắt buộc nhân trọng số 2; lấy giá trị lớn nhất theo
   CV.
5. Cross-encoder MiniLM-L6 chấm đồng thời cặp (yêu cầu, evidence chunk).

Hai kết quả đo: BM25 vượt dense retrieval trên CV kiểm thử, khớp với Resume2Vec; cross-encoder đóng góp lớn nhất,
+0.05 đến +0.08 nDCG@10. Fine-tune cross-encoder (E20) tăng 0.046 trên dữ liệu tổng hợp nhưng giảm 0.027 trên CV thực tế,
nên không dùng. Mời bạn Đán trình bày phần agent.

### Slide 10 · P7: agent làm việc theo mục tiêu của HR · Đán · 1:30

Đây là phần làm hệ thống trở thành agentic. Khi HR đưa mục tiêu bằng lời, P7 chọn bước tiếp theo trong 17 typed tool,
theo sáu bước trên slide:

1. **Hiểu mục tiêu:** `formalize_goal` biến lời HR thành GoalSpec gồm ràng buộc bắt buộc, nên có, điểm cộng, loại trừ
   và số lượng. Mỗi ràng buộc phải trích nguyên văn lời HR (Q19), nên LLM không tự thêm yêu cầu.
2. **Khái niệm mới:** với tiêu chí không có trong danh mục như "AI trong automation", `define_concept` để LLM đề xuất từ
   khoá; code tìm và chấm bằng chứng trong CV theo cùng thang dùng thật, liệt kê, hedge.
3. **Kiểm tra khả thi:** `check_goal` chạy trước khi xếp hạng, 0 token: đếm CV đáp ứng từng ràng buộc và đồng thời,
   chỉ ra điểm nghẽn và nới ràng buộc nào thì được bao nhiêu CV.
4. **Hỏi lại HR:** chỉ khi mục tiêu không khả thi hoặc mơ hồ; câu hỏi không cần thiết bị harness từ chối.
5. **Hành động:** `screen` chạy P6 đến bước xếp hạng, 0 token LLM; `explain` sinh giải thích có trích dẫn đã kiểm chứng;
   `compare`, `investigate` khi cần.
6. **Tự kiểm và trả lời:** `goal_report` so shortlist với mục tiêu; `respond_to_hr` bị từ chối cho đến khi nêu rõ phần
   chưa đạt (Q20).

Bao quanh mọi bước là guard: schema và precondition của từng tool, HR authority (chỉ lời HR mới đổi được yêu cầu, Q16),
trả lời HR phải trung thực với CV (Q15), và budget bước, token, lượt hội thoại; hết budget thì code trả lời bằng các CV
đã kiểm chứng. Luật vẫn chấm điểm; LLM quyết định thứ tự công việc; code quyết định điều gì là dữ kiện.

### Slide 11 · P6: sáu bounded loop bên trong công cụ screen · Đán · 1:00

Bên trong công cụ `screen` là P6, đồ thị cố định với sáu bounded loop, mỗi loop có giới hạn trong cấu hình:

- **L1 clarification:** luật quyết định trước (JD không có công cụ bắt buộc thì hỏi HR), sau đó LLM với typed output.
- **L2 query expansion:** luật đề xuất công cụ thay thế khi yêu cầu bắt buộc khan hiếm; HR phê duyệt.
- **L3 arbiter:** chỉ đổi chỗ cặp near-tie (|Δ| < 0.05); LLM đánh giá theo hai thứ tự, nếu kết quả đảo thì giữ thứ tự
  của luật.
- **L4 verify evidence:** code check Q2–Q5; sai thì regenerate.
- **L5 HR review:** HR chấp nhận, chỉnh trọng số hoặc loại CV.
- **L6 investigate:** vòng ReAct đọc CV lỗi ký tự; LLM chọn hành động, code xác minh từng dữ kiện.

Ba số đo: trên 18 JD, P6 đi 8 execution path khác nhau, fixed pipeline chỉ 1; 17/17 verdict near-tie ổn định khi đảo
thứ tự; investigator có 0 false recovery trên 42 ca thử.

### Slide 12 · 20 tiêu chí guardrail · Đán · 1:00

Mỗi loại output của LLM có một tiêu chí, một phép kiểm tra bằng code và một hành động khi vi phạm. Ví dụ: Q2, trích dẫn
phải có trong CV, sai thì regenerate và nêu rõ trích dẫn sai; Q4, nhận định phải nhất quán với CV đã phân tích, sai thì
regenerate rồi gắn [unverified]; Q5, câu chứa protected attribute bị xoá; Q14, tool call không hợp lệ bị từ chối kèm lý
do; Q19 và Q20 như vừa trình bày. Q1–Q7 và Q14–Q20 chạy online trên mọi output; còn lại đánh giá offline.

Kết quả đo:

- 100% trích dẫn có trong CV.
- Nhận định mâu thuẫn với CV ở P6 giảm từ 2.3% xuống 1.5%; fixed pipeline không regenerate giữ ở 3.5%.
- Ứng viên trong báo cáo có nhận định không có căn cứ giảm từ 6.7% xuống 1.1%.
- 10/10 tấn công prompt injection bị chặn, 0/150 báo nhầm trên CV sạch.
- 0 điểm thay đổi khi đổi họ tên, giới tính, tuổi, hôn nhân (6 biến thể × 1.800 cặp). Mời bạn Hoa trình bày dữ liệu và kết quả thực nghiệm.

---

## 03 · Dữ liệu và kết quả thực nghiệm

### Slide 13 · Dữ liệu và NLP501-bench · Hoa · 1:00

Nhóm dùng bốn nguồn dữ liệu với vai trò cố định:

- **ours:** 50 CV × 12 JD, nhãn silver do hai LLM gán, cùng 120 cặp do người chấm mù.
- **coding:** 200 CV × 6 JD, nhãn rubric từ data generator.
- **Djinni:** dữ liệu thực tế, 250 CV × 8 JD, 424 cặp có nhãn **Dual-annotator Aggregate Human Gold** do anh Vũ và anh Đán chấm mù độc lập, dùng cho test v2 và external validation.
- **Kaggle:** 2.484 CV PDF thuộc 24 ngành, không có nhãn xếp hạng, chỉ dùng để kiểm tra NLP pipeline.

NLP501-bench v1 gồm 250 CV tổng hợp ẩn danh và 18 JD; 10 JD test chỉ báo cáo một lần; 268 tệp đầu vào được cố định bằng
SHA-256. Tập holdout Djinni gồm 6 JD × 40 CV mới, khoá trước mọi lần chạy và đang chờ hai bộ nhãn mù.

Metric chính là nDCG@10, kèm P@5, R@20, MRR và Kendall τ; khoảng tin cậy 95% bằng bootstrap theo JD; so sánh bằng
one-sided paired bootstrap; inter-annotator agreement bằng quadratic-weighted Cohen's κ. Chống leakage (G5): chỉ tune
trên dev, TREC-style pooling, holdout chỉ mở khi có gold labels.

### Slide 14 · Chất lượng NLP pipeline & Benchmark trích xuất · Hoa · 0:45

- **Benchmark trích xuất thực tế (50 profile Djinni unseen, có nhãn Human-Adjudicated Reference):** Micro-F1 đạt **0.943 [0.872, 0.986]**, vượt trội baseline từ khóa E0 (0.847); độ đồng thuận gán nhãn unweighted κ = **0.818**, overlap F1 = **0.991** (giải quyết 53 ca bất đồng status); ghi nhận trung thực 17.54% công cụ OOV ngoài từ điển (Allure, Gradle, Swagger, Jira...).
- **Đánh giá Trade-off NegEx (E1 vs E2):** Trên hồ sơ CV thực tế, No-Context E1 (0.947) nhỉnh hơn nhẹ Full Pipeline E2 (0.943) với p = 0.627 (không có ý nghĩa thống kê) do hồ sơ thực tế rất ít dùng từ lấp liếm (~3.5%); NegEx phát huy vai trò quyết định trong việc chặn trap CV trên tập kiểm thử bẫy (`ours`).
- **Kiểm định lan truyền sai số (Extraction-induced Knockouts):** Pipeline E2 đạt **0/308 (0.0%)** trường hợp loại nhầm ứng viên đủ điều kiện (so với 7/308 tương đương 2.3% của E0); tương quan thứ hạng kỹ năng với nhãn người đạt Kendall τ = **0.900** (Spearman ρ = 0.916).
- **Độ bền vững & Bảo mật:** Không truyền văn bản CV ra ngoài môi trường máy chủ nội bộ ở cấu hình deterministic; 0 trường hợp rò rỉ PII; thời gian xử lý 12.7 ms/CV với chi phí 0 token.

### Slide 15 · Benchmark chính thức v1 · Hoa · 1:15

Trên 10 JD test:

- Tám tiêu chí luật vượt phiên bản v2 +0.093 (p = 0.009).
- P6 vượt rubric ranker +0.041, nhưng p = 0.12, chưa có ý nghĩa thống kê.
- P6 vượt fixed pipeline +0.012 (p = 0.064); chênh lệch nằm chủ yếu ở JD mơ hồ duy nhất, nơi agent hỏi lại HR (0.68 so
  với 0.58).
- Dữ kiện có cấu trúc vượt text retrieval: 0.841 so với 0.715.

Khi chấm lại bằng 120 cặp nhãn người chấm mù (v1.1), thứ tự của 8 hệ thống không đổi, Kendall τ bằng 1.00, và nDCG thay
đổi không quá 0.007. So sánh giữa các hệ thống vì vậy đứng vững khi người chấm thay cho LLM.

### Slide 16 · External validation trên CV thực tế · Hoa · 1:15

Trên 8 JD Djinni thật với nhãn người: BM25 0.422, dense 0.464, hybrid 0.482, tám tiêu chí luật 0.636, agent P6 0.643.
Luật vượt BM25 +0.214 với p < 0.001; P@5 của P6 là 0.725. Luật và agent gần như bằng nhau vì điểm do luật quyết định.

Inter-annotator agreement: κ 0.696 trên Djinni (424 cặp), κ 0.965 trên bộ ours (120 cặp chấm mù). Giá trị tuyệt đối
trên dữ liệu thật thấp hơn dữ liệu tổng hợp (0.64 so với 0.85): điều đáng tin là thứ tự giữa các hệ thống, không phải
con số tuyệt đối.

### Slide 17 · Đóng góp của agent · Hoa · 1:30

Thí nghiệm có kiểm soát trên phiên bản mơ hồ của 18 JD: không hỏi thì nDCG giảm từ 0.851 xuống 0.626; hỏi và HR trả
lời thì lên 0.756, tức +0.129 (p < 0.001), khôi phục 57% mức giảm; Kendall τ tăng 0.056 (p = 0.012).

Hành vi được kiểm tra tự động bằng code: 40/42 check trên các tác vụ HR (E15), 81/83 trên mục tiêu phức tạp chạy 3 lần
mỗi mục tiêu (E26). Ví dụ: "bắt buộc Docker" thì mọi CV xếp hạng đều có Docker; "loại CV hạng 1" thì CV rời bảng xếp
hạng; "dưới 30 tuổi, ưu tiên nam" thì hệ thống từ chối.

Một nguyên tắc đánh giá: bước xếp hạng bằng LLM chỉ được chấm bằng nhãn do chính họ mô hình đó gán thì con số không
được công bố, và tính năng đó để tắt.

### Slide 18 · Hai lượt chạy thật · Hoa · 1:15

Bên trái là P6 trên JD mơ hồ JD-003. Luật thấy không có công cụ bắt buộc nên hỏi HR, không cần LLM. Câu trả lời
"Selenium hoặc Playwright" được chuẩn hoá thành nhóm one-of. Verify bắt được 1/40 nhận định mâu thuẫn và regenerate.
Arbiter gặp cặp near-tie Δ 0.022 cho kết quả ngược khi đảo thứ tự, nên tự giữ thứ tự của luật.

Bên phải là P7 với mục tiêu không khả thi: 5 CV phải biết cả Cypress và Appium. Feasibility check trước khi xếp hạng
cho thấy chỉ 1/56 CV đáp ứng, agent hỏi lại HR; HR chấp nhận ít hơn, agent đề xuất đúng CV-026 và khai báo "1 of 5".
Mô hình chọn hành động; code quyết định điều gì trở thành dữ kiện. Mời bạn Vũ trình bày phần thảo luận và kết luận.

---

## 04 · Thảo luận và kết luận

### Slide 19 · So sánh với nghiên cứu liên quan · Vũ · 0:45

Đặt giải pháp cạnh ba nghiên cứu gần nhất. Resume2Vec cho thấy ATS tốt hơn embedding ở kiểm thử phần mềm; nhóm kiểm chứng lại: BM25 0.794 so với dense 0.741 trên ours, 0.542 so với 0.333 trên coding. ConFit v2 dùng bi-encoder contrastive, nDCG@10 82.20 trên AliYun; trên CV kiểm thử dense đơn thuần yếu, nên nhóm dùng hybrid retrieval, cross-encoder và luật, được +0.214 so với BM25 trên CV thực tế. RankGPT để LLM xếp hạng toàn bộ ứng viên; nhóm chỉ dùng LLM cho near-tie và giải thích, vì để LLM chấm điểm làm giảm 0.033, còn dùng LLM có chọn lọc tiết kiệm 47% token.

### Slide 20 · Giả thuyết và kết quả âm · Vũ · 1:15

- **H1, agent vượt fixed pipeline:** 0.851 so với 0.839, p = 0.064: không được ủng hộ.
- **H2, selective arbiter giảm token:** −47% (P6), −67% (B4) so với dùng LLM cho mọi CV: ủng hộ một phần, chưa đạt mục
  tiêu −78%.
- **H3, verification giữ giải thích grounded:** 100% trích dẫn có căn cứ: được ủng hộ.
- **H4, company profile khớp HR:** τ 0.554 so với 0.542: không được ủng hộ.
- **H6, hỏi lại HR cải thiện JD mơ hồ:** +0.129 (p < 0.001): được ủng hộ.

Error analysis trên test: P6 bỏ sót 6 và báo nhầm 1 trên ours, 4 và 1 trên coding; rubric ranker 10 và 1, 4 và 0. Hai
nguồn lỗi chính là JD mơ hồ và CV mô tả công cụ mà không gọi tên.

Kết quả âm được giữ lại: structural chunking không tăng nDCG; ablation từng bước NLP gần như không đổi nDCG nhưng xử lý
hedge giữ trap CV ngoài top 10; fine-tune cross-encoder không transfer sang CV thật; HR learning qua nhiều session không
cải thiện (−0.008); single agent không có harness tốn 127k–230k token mỗi JD và xếp hạng cả CV chưa đọc; LLM shortlist
ranking chỉ cải thiện trên nhãn do LLM tạo nên không được tính và bị tắt mặc định.

### Slide 21 · Hạn chế và đạo đức · Vũ · 0:30

Dữ liệu xếp hạng chủ yếu là tổng hợp, giá trị tuyệt đối không chuyển sang dữ liệu thật (0.85 so với 0.64); test split
10 JD nên P6 so với rubric ranker chưa có ý nghĩa thống kê; nhãn Djinni là Dual-annotator Aggregate Human Gold do hai người chấm mù độc lập (metadata saw_model nhập nhầm trước đó đã được đính chính); điểm lấy trung bình thực không làm tròn, phân tầng độ tin cậy và kiểm định độ nhạy với cận min/max; HR trong E15 và E26 là kịch bản. Về đạo đức: protected attribute bị xoá trước
khi lập chỉ mục, hệ thống chỉ gợi ý, HR quyết định, và yêu cầu lọc theo tuổi hay giới tính bị harness từ chối.

### Slide 22 · Mức độ giải quyết G1–G5 · Vũ · 1:00

- **G2 đã giải quyết:** luật kết hợp hybrid retrieval vượt BM25 +0.214 trên CV thực tế có nhãn người.
- **G3 kiểm soát được:** 100% trích dẫn grounded, nhận định mâu thuẫn giảm còn 1.5% nhưng chưa về 0.
- **G1 một phần:** hiệu quả của trọng số theo doanh nghiệp chưa đo được vì nhãn không gắn bối cảnh doanh nghiệp (A2
  −0.004, HR learning −0.008).
- **G4 một phần:** token giảm 47–67%, chưa đạt 78%.
- **G5 một phần:** benchmark cố định bằng SHA-256, test báo cáo một lần, v1.1 cho τ 1.00 với nhãn người; holdout đã khoá
  nhưng chưa có nhãn nên chưa đo.

### Slide 23 · Kết luận và hướng phát triển · Vũ · 1:15

Bốn kết quả chính:

- P6 đạt 0.851 nDCG@10 trên test split, cao nhất trong các phiên bản, và thứ tự không đổi khi dùng nhãn người.
- Trên CV thực tế có nhãn người: 0.422 với BM25, 0.643 với agent.
- Hỏi lại HR trên JD mơ hồ cải thiện +0.129.
- Mọi output của LLM được kiểm tra bằng code theo 20 tiêu chí; không điểm nào thay đổi theo protected attribute.

Kết luận kỹ thuật: điểm số nên do luật xác định trên dữ liệu đã được xử lý tốt; agent có giá trị ở những khâu cần phán
đoán và tương tác với HR.

Hướng phát triển: gán nhãn mù cho holdout Djinni và đánh giá một lần; adjudication để có gold labels cho hồ sơ thật ngắn;
lặp lại E15 với HR thật; thu thập CV thật có đồng ý và mở rộng test split; statistical NER cho CV mô tả công cụ mà không
gọi tên.

### Slide 24 · Hỏi đáp · Vũ · 0:15

Nhóm xin cảm ơn Hội đồng đã lắng nghe. Mã nguồn, benchmark và toàn bộ kết quả có trong repository; báo cáo đầy đủ ở
`docs/FINAL_REPORT.md`. Nhóm sẵn sàng nhận câu hỏi.

---

## Phụ lục (chỉ mở khi có câu hỏi)

### Slide 25 · Câu hỏi nghiên cứu và đóng góp

Nhóm đặt năm câu hỏi nghiên cứu; cột phải là câu trả lời đo được:

- **RQ1:** structured extraction vượt raw-text matching: nDCG@10 0.841 so với 0.715.
- **RQ2:** trọng số theo doanh nghiệp chưa thể hiện trên bộ nhãn hiện có (Δ −0.004), vì nhãn không gắn bối cảnh
  doanh nghiệp.
- **RQ3:** cross-encoder là thành phần retrieval đóng góp lớn nhất, +0.05 đến +0.08.
- **RQ4:** giá trị của agent so với fixed pipeline nằm ở việc hỏi lại HR khi JD mơ hồ: +0.129.
- **RQ5:** lỗi tập trung ở JD mơ hồ và ở CV mô tả công cụ mà không nêu tên.

Bốn đóng góp: một NLP pipeline song ngữ tất định; một agent có kiểm soát gồm LangGraph StateGraph 12 node, 6 bounded
loop (P6) và một goal-driven orchestrator với 17 tool (P7); bộ 20 tiêu chí cho output của LLM, mỗi tiêu chí có phép
kiểm tra tự động; và benchmark NLP501-bench ở các phiên bản v1, v1.1, v2 cùng một tập holdout khoá trước.

---

## Chuẩn bị câu hỏi phản biện

| Câu hỏi | Người trả lời | Câu trả lời ngắn |
|---|---|---|
| Hệ thống có thật sự agentic không, hay chỉ là pipeline? | Đán | P7: LLM chọn bước trong 17 tool theo mục tiêu HR, kiểm tra tính khả thi trước khi xếp hạng, hỏi lại khi cần, tự kiểm kết quả và khai phần chưa đạt (slide 18, hai lượt chạy thật). P6 đi 8 path khác nhau trên 18 JD, fixed pipeline đi 1. |
| Vì sao không để LLM chấm điểm? | Thịnh | Đã đo: ở v2 LLM chấm 5/8 tiêu chí làm nDCG giảm 0.033. LLM cũng có position bias (arbiter tự lật khi đảo thứ tự). Luật chấm, LLM giải thích và phân xử near-tie. |
| Agent không hơn fixed pipeline về nDCG, vậy agent để làm gì? | Hoa | Đúng theo thiết kế, vì luật chấm cả hai. Giá trị nằm ở JD mơ hồ (+0.129), CV lỗi ký tự, kiểm soát hallucination và mục tiêu HR phức tạp. |
| Nhãn do LLM gán thì đánh giá có vòng lặp không? | Hoa | Có rủi ro, nên: 120 cặp chấm mù bởi người cho v1.1 (thứ tự giữ nguyên, τ 1.00); Djinni dùng Dual-annotator Aggregate Human Gold (chấm mù độc lập 100%, kiểm định độ nhạy min/max); con số nào chỉ có nhãn cùng họ mô hình chấm thì không công bố và tính năng để tắt. |
| Dữ liệu tổng hợp có đại diện không? | Vũ | Không cho giá trị tuyệt đối (0.85 so với 0.64). Kết luận dựa trên thứ tự giữa các hệ thống, đã được xác nhận trên 8 JD thật với nhãn người. |
| Bước trích xuất có benchmark uy tín không, sai số có ảnh hưởng xếp hạng không? | Thịnh | Có benchmark độc lập: 50 profile Djinni unseen có nhãn Human-Adjudicated Reference (unweighted κ = 0.818, overlap F1 = 0.991), Micro-F1 đạt 0.943 [0.872, 0.986] (vượt E0 keyword 0.847, OOV 17.5%). Kiểm định trên 308 cặp đủ điều kiện cho 0% loại nhầm và Kendall τ = 0.900; Agent Investigator hỗ trợ mở rộng context cho các ca nghi ngờ. |
| Vì sao gazetteer mà không phải NER học máy? | Thịnh | Tên công cụ là tập từ khóa có ranh giới rõ; gazetteer kết hợp NegEx và boundary đạt Micro-F1 0.943 trên 50 CV thật và 0.978 trên dev set, chi phí 0 token và không rò rỉ dữ liệu. NER học máy là hướng mở rộng để khắc phục 17.5% OOV cho các công cụ chưa có trong từ điển. |
| Có học từ phản hồi của HR không? | Đán | Có hiệu chỉnh trọng số trong phiên bằng pairwise logistic regression; học qua nhiều phiên đã đo (E24) và không cải thiện (−0.008), nên quyết định loại của HR được giữ như quyết định cứng. |
