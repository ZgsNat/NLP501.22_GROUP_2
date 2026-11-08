# Script thuyết trình · NLP501 Nhóm 2 · deck v12

Deck: `NLP501_Nhom2_BaoCaoCuoiKy_v12.pptx` (25 slide, khoảng 27 phút). Số liệu khớp với slide và [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md).

| Người nói | Slide | Thời lượng |
|---|---|---|
| Hoàng Xuân Sơn | 1–4, 24–25 | 5:00 |
| Nguyễn Đình Đán | 5, 12–14, 23 | 6:15 |
| Trần Việt Hoa | 6, 15–19 | 6:45 |
| Nguyễn Thái Thịnh | 7–11 | 5:45 |
| Nguyễn Văn Vũ | 20–22 | 3:30 |

Quy ước: thuật ngữ kỹ thuật giữ nguyên tiếng Anh như trên slide. Mỗi đoạn kết thúc bằng một câu chuyển lời.

---

## Phần 01 · Đặt vấn đề

### Slide 1 · Tiêu đề · Sơn · 0:30

Kính thưa Hội đồng, nhóm 2 trình bày đề tài: hệ thống Agentic AI xếp hạng và giải thích mức độ phù hợp giữa CV và Job
Description cho vị trí Automation Test Engineer. Nguyên tắc thiết kế nằm ở dòng phụ đề: điểm số do luật xác định, LLM
có guardrail, quyết định cuối cùng thuộc về con người. Năm thành viên trình bày theo phần việc mỗi người phụ trách.

### Slide 2 · Bố cục · Sơn · 0:30

Báo cáo gồm sáu phần: đặt vấn đề, dữ liệu và benchmark, phương pháp với trọng tâm là các kỹ thuật NLP, guardrail cho
output của LLM, kết quả thực nghiệm, và thảo luận cùng kết luận.

### Slide 3 · Bài toán · Sơn · 1:15

Bài toán được phát biểu như sau.

- **Đầu vào:** một JD tiếng Anh hoặc tiếng Việt, thường mơ hồ; mục tiêu của HR diễn đạt bằng lời, ví dụ "5 CV, bắt
  buộc Selenium, có Playwright càng tốt, có dùng AI trong automation là điểm cộng lớn"; một kho 50 đến 250 CV dạng
  PDF, DOCX hoặc ảnh quét.
- **Đầu ra:** danh sách ứng viên xếp hạng, mỗi người có điểm theo tám tiêu chí và trích dẫn đã kiểm chứng trong CV;
  mọi CV còn lại có trạng thái và lý do; phần mục tiêu của HR không đáp ứng được phải được nêu rõ.
- **Ràng buộc:** không dùng thông tin định danh, điểm số do luật xác định, HR ra quyết định cuối cùng.

Keyword screening (ATS) không giải được bài toán này vì ba lý do:

1. **Automation depth.** Chạy kịch bản (L1), viết kịch bản (L2), xây framework (L3) và thiết kế kiến trúc kiểm thử
   (L4) là bốn năng lực khác nhau, dù CV có thể dùng chung một từ khoá.
2. **Mức độ sử dụng.** "Selenium" trong mục Kinh nghiệm là bằng chứng đã dùng; trong mục Kỹ năng chỉ là tự nhận.
3. **Hedging.** "Familiar with Cypress", "đang tự học Playwright" có nhắc tên công cụ nhưng không chứng minh năng lực.

Điểm của một CV với một JD được tính theo công thức trên slide. H là knock-out theo công cụ bắt buộc: thiếu thì điểm
bằng 0. w là trọng số theo loại doanh nghiệp. S là điểm 0–100 của tám tiêu chí, đều tính bằng luật trên hồ sơ đã phân
tích.

### Slide 4 · Câu hỏi nghiên cứu và đóng góp · Sơn · 1:15

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
Mời bạn Đán trình bày các khoảng trống nghiên cứu.

### Slide 5 · Khoảng trống nghiên cứu G1–G5 · Đán · 1:15

Từ tổng quan tài liệu, nhóm rút ra năm khoảng trống:

- **G1, đặc thù doanh nghiệp:** cùng một hồ sơ, ngân hàng và startup đánh giá khác nhau; hệ thống so khớp CV–JD hiện
  có ít khi mô hình hoá điều này.
- **G2, nghịch lý từ vựng:** Resume2Vec cho thấy kiểm thử phần mềm là lĩnh vực mà keyword ATS vượt embedding, vì tên
  công cụ chính xác mang tính quyết định.
- **G3, LLM hallucination:** khi xếp hạng hoặc giải thích, LLM có thể nêu kỹ năng không có trong CV.
- **G4, chi phí:** reranking bằng LLM trên toàn bộ ứng viên, như RankGPT, rất tốn kém.
- **G5, evaluation data leakage:** test split nhỏ, đã được nhìn thấy trước và được gán nhãn bởi chính LLM thì dễ đánh
  giá quá cao.

Cuối bài, nhóm sẽ quay lại đánh giá mức độ giải quyết từng khoảng trống. Mời bạn Hoa trình bày dữ liệu.

## Phần 02 · Dữ liệu và benchmark

### Slide 6 · Dữ liệu và NLP501-bench · Hoa · 1:15

Nhóm dùng bốn nguồn dữ liệu với vai trò cố định:

- **ours:** 50 CV × 12 JD, nhãn silver do hai LLM gán, cùng 120 cặp do người chấm mù.
- **coding:** 200 CV × 6 JD, nhãn rubric từ data generator.
- **Djinni:** dữ liệu thực tế, 250 CV × 8 JD, 424 cặp có nhãn người (Vũ và Đán), dùng cho test v2 và external
  validation.
- **Kaggle:** 2.484 CV PDF thuộc 24 ngành, không có nhãn xếp hạng, chỉ dùng để kiểm tra NLP pipeline.

NLP501-bench v1 gồm 250 CV tổng hợp ẩn danh và 18 JD; 10 JD test chỉ báo cáo một lần; 268 tệp đầu vào được cố định bằng
SHA-256. Tập holdout Djinni gồm 6 JD × 40 CV mới, khoá trước mọi lần chạy và đang chờ hai bộ nhãn mù.

Metric chính là nDCG@10, kèm P@5, R@20, MRR và Kendall τ; khoảng tin cậy 95% bằng bootstrap theo JD; so sánh bằng
one-sided paired bootstrap; inter-annotator agreement bằng quadratic-weighted Cohen's κ. Chống leakage (G5): chỉ tune
trên dev, TREC-style pooling, holdout chỉ mở khi có gold labels. Mời bạn Thịnh trình bày phương pháp.

## Phần 03 · Phương pháp

### Slide 7 · Kiến trúc tổng thể · Thịnh · 1:15

Luồng chính đi từ trái sang phải. CV và JD được NLP pipeline X1–X11 xử lý, hoàn toàn tất định, không dùng LLM. Hybrid
retrieval tìm ứng viên, rule-based scoring tính knock-out H và tám điểm S, rồi trả về HR danh sách xếp hạng kèm lý do và
trạng thái.

LLM chỉ làm các việc cần phán đoán: hỏi lại HR khi JD mơ hồ, đọc CV bị lỗi ký tự, phân xử cặp near-tie theo hai thứ tự,
sinh giải thích có trích dẫn, và điều phối theo mục tiêu của HR ở P7. Mọi output của LLM đi qua harness gồm schema check,
precondition, HR authority, 20 tiêu chí và budget về bước và token.

Lý do thiết kế đến từ số đo: ở v2, khi LLM đặt 5 trên 8 điểm S, nDCG@10 giảm 0.033. Từ v3, luật xác định cả tám điểm.

### Slide 8 · NLP pipeline cho CV · Thịnh · 1:00

Pipeline gồm các bước: đọc tệp (pypdf, python-docx, PaddleOCR), Unicode normalization, PII redaction, lọc prompt
injection, sectioning và date grammar, structural chunking, nhận diện kỹ năng kèm hedge và depth, cuối cùng là
CVProfile.

Nguyên tắc là CV phải được xử lý như văn bản bán cấu trúc chứ không phải túi từ. Cùng là chữ "Selenium": trong mục Kinh
nghiệm là bằng chứng sử dụng, trong mục Kỹ năng là tự nhận, sau "familiar with" là hedge, còn trong "our developers run
Selenium" là việc của người khác. Vì vậy cấu trúc được khôi phục trước, và mọi bước sau chỉ đọc CVProfile, không đọc văn
bản thô.

### Slide 9 · Preprocessing song ngữ · Thịnh · 1:00

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

### Slide 10 · Information extraction · Thịnh · 1:15

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

### Slide 11 · Hybrid retrieval và reranking · Thịnh · 1:15

Retrieval gồm năm bước:

1. BM25 bắt chính xác tên công cụ như RestAssured, k6, Appium.
2. Dense embedding bge-small-en-v1.5 bắt ngữ nghĩa: "test harness" gần với "framework kiểm thử".
3. Reciprocal Rank Fusion gộp hai danh sách.
4. Mỗi requirement là một query riêng, có mở rộng biến thể; yêu cầu bắt buộc nhân trọng số 2; lấy giá trị lớn nhất theo
   CV.
5. Cross-encoder MiniLM-L6 chấm đồng thời cặp (yêu cầu, evidence chunk).

Hai kết quả đo: BM25 vượt dense retrieval trên CV kiểm thử, khớp với Resume2Vec; cross-encoder đóng góp lớn nhất,
+0.05 đến +0.08 nDCG@10. Fine-tune cross-encoder (E20) tăng 0.046 trên dữ liệu tổng hợp nhưng giảm 0.027 trên CV thực tế,
nên không dùng. Mời bạn Đán trình bày phần chấm điểm và agent.

### Slide 12 · Rule-based scoring và explanation card · Đán · 1:15

Ví dụ là JD-010 (ngân hàng) và CV-033. Mỗi dòng là một tiêu chí: trọng số, điểm, đóng góp và bằng chứng trích từ CV.
Công cụ: 3/3 yêu cầu bắt buộc, đóng góp 0.184. Depth L3, xác định từ động từ của chính ứng viên. CI/CD đạt 90 vì
Jenkins chỉ được liệt kê, tính 0.7. Tổng 0.924.

Explanation card được tính từ chính các luật chấm điểm, nên luôn nhất quán với thứ hạng; LLM không thể giải thích lệch
với điểm. Trọng số theo loại doanh nghiệp, tổng bằng 1: gia công phần mềm nâng lĩnh vực và tiếng Anh lên 0.20, ngân
hàng tăng ISTQB, startup ưu tiên loại kiểm thử và CI/CD. Trọng số có thể hiệu chỉnh từ quyết định của HR bằng pairwise
logistic regression. Với CV bị loại, what-if analysis cho HR thấy ứng viên còn cách ngưỡng bao xa.

### Slide 13 · Thiết kế agent · Đán · 1:30

Agent có hai tầng.

**P6 là đồ thị cố định với sáu bounded loop**, mỗi loop có giới hạn trong cấu hình:

- **L1 clarification:** luật quyết định trước (JD không có công cụ bắt buộc thì hỏi HR), sau đó LLM với typed output.
- **L2 query expansion:** luật đề xuất công cụ thay thế khi yêu cầu bắt buộc khan hiếm; HR phê duyệt.
- **L3 arbiter:** chỉ đổi chỗ cặp near-tie (|Δ| < 0.05); LLM đánh giá theo hai thứ tự, nếu kết quả đảo thì giữ thứ tự
  của luật.
- **L4 verify evidence:** code check Q2–Q5; sai thì regenerate.
- **L5 HR review:** HR chấp nhận, chỉnh trọng số hoặc loại CV.
- **L6 investigate:** vòng ReAct đọc CV lỗi ký tự; LLM chọn hành động, code xác minh từng dữ kiện.

**P7 là orchestrator, cấu hình mặc định.** LLM chọn bước tiếp theo trong 17 typed tool. Mục tiêu của HR được formalize
thành GoalSpec; mỗi ràng buộc phải trích nguyên văn lời HR (Q19). Feasibility check chạy trước khi xếp hạng; nếu không
đạt, agent hỏi lại HR và câu trả lời phải nêu phần chưa đáp ứng (Q20).

Ba số đo cho thấy đây là agent chứ không phải pipeline tuyến tính: trên 18 JD, P6 đi 8 execution path khác nhau, fixed
pipeline chỉ đi 1; 17/17 verdict near-tie ổn định khi đảo thứ tự; investigator có 0 false recovery trên 42 ca thử.

### Slide 14 · 20 tiêu chí guardrail · Đán · 1:15

Mỗi loại output của LLM có một tiêu chí, một phép kiểm tra bằng code và một hành động khi vi phạm. Ví dụ: Q2, trích dẫn
phải có trong CV, sai thì regenerate và nêu rõ trích dẫn sai; Q4, nhận định phải nhất quán với CV đã phân tích, sai thì
regenerate rồi gắn [unverified]; Q5, câu chứa protected attribute bị xoá; Q14, tool call không hợp lệ bị từ chối kèm lý
do; Q19 và Q20 như vừa trình bày. Q1–Q7 và Q14–Q20 chạy online trên mọi output; còn lại đánh giá offline.

Kết quả đo:

- 100% trích dẫn có trong CV.
- Nhận định mâu thuẫn với CV ở P6 giảm từ 2.3% xuống 1.5%; fixed pipeline không regenerate giữ ở 3.5%.
- Ứng viên trong báo cáo có nhận định không có căn cứ giảm từ 6.7% xuống 1.1%.
- 10/10 tấn công prompt injection bị chặn, 0/150 báo nhầm trên CV sạch.
- 0 điểm thay đổi khi đổi họ tên, giới tính, tuổi, hôn nhân (6 biến thể × 1.800 cặp).

Mời bạn Hoa trình bày kết quả thực nghiệm.

## Phần 05 · Kết quả thực nghiệm

### Slide 15 · Chất lượng NLP pipeline · Hoa · 1:00

- Tool extraction F1 0.978 (tiếng Anh 0.983, tiếng Việt 0.969); trên CV quét qua OCR F1 0.997, CER 0.031.
- Số năm kinh nghiệm sai số trung bình 0.79 năm; depth đúng 89%.
- 0% lộ thông tin cá nhân sau redaction.
- Depth trên CV tiếng Việt gõ không dấu từ 0% lên 87.5%.

Trên 2.484 CV PDF thật thuộc 24 ngành, pipeline đọc được 100% tệp và lộ ra ba lỗi dữ liệu tổng hợp không có: tỉ lệ vai
trò xác định được khoảng ngày tăng từ 41.9% lên 97.7%, nhận nhầm tên công cụ trùng từ thông dụng giảm từ 17.2% xuống
1.6%. Giới hạn: CV bị nhiễu ký tự chỉ đạt F1 0.157; trường hợp này chuyển cho investigator.

### Slide 16 · Benchmark chính thức v1 · Hoa · 1:15

Trên 10 JD test:

- Tám tiêu chí luật vượt phiên bản v2 +0.093 (p = 0.009).
- P6 vượt rubric ranker +0.041, nhưng p = 0.12, chưa có ý nghĩa thống kê.
- P6 vượt fixed pipeline +0.012 (p = 0.064); chênh lệch nằm chủ yếu ở JD mơ hồ duy nhất, nơi agent hỏi lại HR (0.68 so
  với 0.58).
- Dữ kiện có cấu trúc vượt text retrieval: 0.841 so với 0.715.

Khi chấm lại bằng 120 cặp nhãn người chấm mù (v1.1), thứ tự của 8 hệ thống không đổi, Kendall τ bằng 1.00, và nDCG thay
đổi không quá 0.007. So sánh giữa các hệ thống vì vậy đứng vững khi người chấm thay cho LLM.

### Slide 17 · External validation trên CV thực tế · Hoa · 1:15

Trên 8 JD Djinni thật với nhãn người: BM25 0.422, dense 0.464, hybrid 0.482, tám tiêu chí luật 0.636, agent P6 0.643.
Luật vượt BM25 +0.214 với p < 0.001; P@5 của P6 là 0.725. Luật và agent gần như bằng nhau vì điểm do luật quyết định.

Inter-annotator agreement: κ 0.696 trên Djinni (424 cặp), κ 0.965 trên bộ ours (120 cặp chấm mù). Giá trị tuyệt đối
trên dữ liệu thật thấp hơn dữ liệu tổng hợp (0.64 so với 0.85): điều đáng tin là thứ tự giữa các hệ thống, không phải
con số tuyệt đối.

### Slide 18 · Đóng góp của agent · Hoa · 1:15

Thí nghiệm có kiểm soát trên phiên bản mơ hồ của 18 JD: không hỏi thì nDCG giảm từ 0.851 xuống 0.626; hỏi và HR trả
lời thì lên 0.756, tức +0.129 (p < 0.001), khôi phục 57% mức giảm; Kendall τ tăng 0.056 (p = 0.012).

Hành vi được kiểm tra tự động bằng code: 40/42 check trên các tác vụ HR (E15), 81/83 trên mục tiêu phức tạp chạy 3 lần
mỗi mục tiêu (E26). Ví dụ: "bắt buộc Docker" thì mọi CV xếp hạng đều có Docker; "loại CV hạng 1" thì CV rời bảng xếp
hạng; "dưới 30 tuổi, ưu tiên nam" thì hệ thống từ chối.

Một nguyên tắc đánh giá: bước xếp hạng bằng LLM chỉ được chấm bằng nhãn do chính họ mô hình đó gán thì con số không
được công bố, và tính năng đó để tắt.

### Slide 19 · Hai lượt chạy thật · Hoa · 0:45

Bên trái là P6 trên JD mơ hồ JD-003. Luật thấy không có công cụ bắt buộc nên hỏi HR, không cần LLM. Câu trả lời
"Selenium hoặc Playwright" được chuẩn hoá thành nhóm one-of. Verify bắt được 1/40 nhận định mâu thuẫn và regenerate.
Arbiter gặp cặp near-tie Δ 0.022 cho kết quả ngược khi đảo thứ tự, nên tự giữ thứ tự của luật.

Bên phải là P7 với mục tiêu không khả thi: 5 CV phải biết cả Cypress và Appium. Feasibility check trước khi xếp hạng
cho thấy chỉ 1/56 CV đáp ứng, agent hỏi lại HR; HR chấp nhận ít hơn, agent đề xuất đúng CV-026 và khai báo "1 of 5".
Mô hình chọn hành động; code quyết định điều gì trở thành dữ kiện. Mời bạn Vũ trình bày phần thảo luận.

## Phần 06 · Thảo luận

### Slide 20 · So sánh với nghiên cứu liên quan · Vũ · 1:15

- **Resume2Vec:** ATS tốt hơn embedding ở kiểm thử phần mềm. Nhóm kiểm chứng lại: BM25 0.794 so với dense 0.741 trên
  ours, 0.542 so với 0.333 trên coding.
- **ConFit v2:** bi-encoder contrastive đạt nDCG@10 82.20 trên AliYun. Với CV kiểm thử, dense đơn thuần yếu, nên nhóm
  dùng hybrid retrieval, cross-encoder và luật: +0.214 so với BM25 trên CV thực tế.
- **RankGPT:** LLM listwise ranking trên toàn bộ ứng viên. Nhóm chỉ dùng LLM làm arbiter cho near-tie; để LLM chấm điểm
  làm giảm 0.033; dùng LLM có chọn lọc tiết kiệm 47% token.
- **Clavié và Soulié:** LLM so khớp kỹ năng ESCO. Nhóm dùng taxonomy và canonicalize output tại boundary; LLM càng mạnh
  càng đọc dư yêu cầu JD (0.884 → 0.842), hybrid parser giữ 0.881.
- **Wilson và Caliskan:** retrieval bằng LM ưu tiên tên người da trắng ở 85.1% trường hợp. Nhóm xoá protected attribute
  trước khi lập chỉ mục và kiểm bằng counterfactual twins: không điểm nào thay đổi.

### Slide 21 · Giả thuyết và kết quả âm · Vũ · 1:15

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

### Slide 22 · Hạn chế và đạo đức · Vũ · 1:00

- Dữ liệu xếp hạng chủ yếu là tổng hợp; giá trị tuyệt đối không chuyển sang dữ liệu thật (0.85 so với 0.64).
- Test split 10 JD, khoảng tin cậy ±0.05–0.08; P6 so với rubric ranker chưa có ý nghĩa thống kê.
- Nhãn Djinni là trung bình của hai người, chưa adjudicate; một bộ được gán khi thấy điểm của mô hình; κ 0.696.
- HR trong E15 và E26 là kịch bản mô phỏng.
- LLM gọi qua Claude Code CLI: lượt gọi, token và hành vi tương đương API; độ trễ và chi phí thì không.

Về đạo đức: protected attribute bị xoá trước khi lập chỉ mục; hệ thống chỉ gợi ý, HR quyết định và mọi quyết định được
ghi log; yêu cầu lọc theo tuổi hay giới tính bị harness từ chối; tôn giáo và dân tộc đã được ẩn nhưng chưa được twin
test. Mời bạn Đán tổng kết các khoảng trống.

### Slide 23 · Mức độ giải quyết G1–G5 · Đán · 1:00

- **G2 đã giải quyết:** luật kết hợp hybrid retrieval vượt BM25 +0.214 trên CV thực tế có nhãn người.
- **G3 kiểm soát được:** 100% trích dẫn grounded, nhận định mâu thuẫn giảm còn 1.5% nhưng chưa về 0.
- **G1 một phần:** hiệu quả của trọng số theo doanh nghiệp chưa đo được vì nhãn không gắn bối cảnh doanh nghiệp (A2
  −0.004, HR learning −0.008).
- **G4 một phần:** token giảm 47–67%, chưa đạt 78%.
- **G5 một phần:** benchmark cố định bằng SHA-256, test báo cáo một lần, v1.1 cho τ 1.00 với nhãn người; holdout đã khoá
  nhưng chưa có nhãn nên chưa đo.

Mời bạn Sơn kết luận.

### Slide 24 · Kết luận và hướng phát triển · Sơn · 1:15

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

### Slide 25 · Hỏi đáp · Sơn · 0:15

Nhóm xin cảm ơn Hội đồng đã lắng nghe. Mã nguồn, benchmark và toàn bộ kết quả có trong repository; báo cáo đầy đủ ở
`docs/FINAL_REPORT.md`. Nhóm sẵn sàng nhận câu hỏi.

---

## Chuẩn bị câu hỏi phản biện

| Câu hỏi | Người trả lời | Câu trả lời ngắn |
|---|---|---|
| Hệ thống có thật sự agentic không, hay chỉ là pipeline? | Đán | P7: LLM chọn bước trong 17 tool theo mục tiêu HR, kiểm tra tính khả thi trước khi xếp hạng, hỏi lại khi cần, tự kiểm kết quả và khai phần chưa đạt (slide 19). P6 đi 8 path khác nhau trên 18 JD, fixed pipeline đi 1. |
| Vì sao không để LLM chấm điểm? | Thịnh | Đã đo: ở v2 LLM chấm 5/8 tiêu chí làm nDCG giảm 0.033. LLM cũng có position bias (arbiter tự lật khi đảo thứ tự). Luật chấm, LLM giải thích và phân xử near-tie. |
| Agent không hơn fixed pipeline về nDCG, vậy agent để làm gì? | Hoa | Đúng theo thiết kế, vì luật chấm cả hai. Giá trị nằm ở JD mơ hồ (+0.129), CV lỗi ký tự, kiểm soát hallucination và mục tiêu HR phức tạp. |
| Nhãn do LLM gán thì đánh giá có vòng lặp không? | Hoa | Có rủi ro, nên: 120 cặp chấm mù bởi người cho v1.1 (thứ tự giữ nguyên, τ 1.00); Djinni dùng nhãn người; con số nào chỉ có nhãn cùng họ mô hình chấm thì không công bố và tính năng để tắt. |
| Dữ liệu tổng hợp có đại diện không? | Vũ | Không cho giá trị tuyệt đối (0.85 so với 0.64). Kết luận dựa trên thứ tự giữa các hệ thống, đã được xác nhận trên 8 JD thật với nhãn người. |
| Vì sao gazetteer mà không phải NER học máy? | Thịnh | Tên công cụ là tập từ vựng đóng; gazetteer với boundary và evidence strength cho F1 0.978. NER học máy là hướng phát triển cho CV mô tả công cụ mà không gọi tên. |
| Có học từ phản hồi của HR không? | Đán | Có hiệu chỉnh trọng số trong phiên bằng pairwise logistic regression; học qua nhiều phiên đã đo (E24) và không cải thiện (−0.008), nên quyết định loại của HR được giữ như quyết định cứng. |
