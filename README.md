# CV Agent – Tìm CV Automation Test phù hợp JD (NLP501 · Nhóm 2)

Prototype hệ thống **Agentic AI** xếp hạng CV Automation Test theo JD (ví dụ JD *Automation Test Engineer – FPT Software*)
và một **Criteria Profile** do HR chọn/chỉnh. Bám đúng 7 stage, 3 vòng lặp và 9 cấu hình thực nghiệm trong tài liệu
*Kế hoạch & Requirements sản phẩm cuối môn*.

```
LangGraph StateGraph · shared state: criteria, candidate pool, evidence, budget, số vòng lặp
 1 Ingest + PII ─► 2 Criteria Profile ─► 3 Hybrid Retrieval ─► 4 Rerank + Scoring ─► 5 Selective Arbiter ─► 6 verify_evidence ─► 7 Report + HR
                   ↺ hỏi HR khi JD mơ hồ   ↺ query expansion (ESCO)                     LLM chỉ cho cặp Δ<0.05   ↺ sinh lại ≤ 3 lần
```

## Cài đặt & chạy (một lệnh mỗi việc)

```bash
py -3.11 -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt          # tuỳ chọn: pip install -r requirements-ml.txt (embedding/cross-encoder/NLI neural)
ollama pull qwen2.5:7b                   # LLM local mặc định (NFR-02: PII không rời máy)
```

| Việc | Lệnh |
|---|---|
| Sinh dữ liệu tổng hợp (200 CV + nhãn 2 annotator) | `python -m cv_agent generate-data` |
| Xếp hạng CV cho JD FPT Software (P6, Ollama) | `python -m cv_agent rank --jd fpt_automation_test --config P6` |
| Như trên nhưng offline, không cần Ollama | `python -m cv_agent rank --config P6 --llm heuristic` |
| Xếp hạng với JD/CV của bạn | `python -m cv_agent rank --jd path/jd.txt --cvs path/cv_folder --profile Outsourcing` |
| Xuất canonical CV JSON (đã che PII, không có raw text) | `python -m cv_agent export-cvs --cvs path/cv_folder --out runs/canonical_cvs` |
| Chạy 9 cấu hình + kiểm định H0–H6 | `python -m cv_agent experiments --split test` |
| Như trên với mô hình neural thật (FR-12/15/22) | `python -m cv_agent experiments --neural --llm ollama --model qwen2.5:3b` |
| Fairness audit đổi tên (H5) | `python -m cv_agent audit --jd fpt_automation_test --config P6` |
| Fine-tune dense encoder + cross-encoder (nên chạy trên Colab GPU) | [`notebooks/finetune_colab.ipynb`](notebooks/finetune_colab.ipynb) hoặc `python -m cv_agent finetune` |
| Fine-tune 3 seed, in trung bình ± độ lệch chuẩn | `python -m cv_agent finetune --seeds 42,7,13` (seed 42 giữ thư mục mặc định, seed khác thêm `-s<seed>`) |
| Kiểm định lại H0–H7 của một lượt đã chạy (không chạy lại mô hình) | `python -m cv_agent rescore runs/experiments/<timestamp>` |
| Huấn luyện tagger skill JobBERT trên SkillSpan (nên chạy trên Colab GPU) | `python -m cv_agent train-tagger` (ghi `models/jobbert-base-cased-skillspan-tagger/tagger_metrics.json`) |
| Chấm span SkillSpan test: từ điển, tagger, hai bộ hợp lại | `python -m cv_agent eval-spans --tagger models/jobbert-base-cased-skillspan-tagger --hybrid` |
| Phân tích lỗi theo JD của một lượt (post hoc) | `python -m cv_agent error-analysis runs/experiments/<timestamp> --data data/public` |
| Thực nghiệm với mô hình đã fine-tune | `python -m cv_agent experiments --neural --dense-model models/multilingual-e5-base-rjf-ft --rerank-model models/bge-reranker-base-rjf-ft` |
| Demo app cho HR | `streamlit run app/streamlit_app.py` |
| Unit test + coverage | `pytest --cov=cv_agent` |

Kết quả mỗi lần `rank` nằm ở `runs/<run_id>/result.json` (ranking, explanation card, trace, thống kê LLM) và
`runs/<run_id>/llm_calls.jsonl` khi có `run_dir`. Thực nghiệm nằm ở `runs/experiments/<timestamp>/{results.csv,summary.json}`
(kèm commit hash).

## Cấu trúc code (chia module theo stage – NFR-07)

| Thư mục | Stage / FR | Người phụ trách |
|---|---|---|
| `cv_agent/schema.py` | Canonical CV/JD/Criteria schema (chốt tuần 1), 8 trục | Cả nhóm |
| `cv_agent/ingest/` | FR-01 parse PDF/DOCX/TXT · FR-02 che PII (rule VN) · FR-03 entity + span · FR-04 depth 1–4 · FR-05 chuẩn hoá ESCO-lite, cờ parse kém · FR-06 trùng lặp | Sơn |
| `cv_agent/criteria/` | FR-07 parse JD → must-have/nice-to-have · FR-08 4 profile YAML (Σw=1) · FR-09 câu hỏi làm rõ · FR-10 HR duyệt | Sơn, Đán |
| `cv_agent/retrieval/` | FR-11 BM25 · FR-12 dense · FR-13 RRF top-k=200 · FR-14 query expansion | Vũ |
| `cv_agent/ranking/` | FR-15 rerank top 20 (mặc định chỉ phá hoà điểm trong top 20 theo Criteria, xem [Thay đổi so với FR-15](#thay-đổi-so-với-fr-15)) · FR-16 `Score = 1[H(c)] × Σ w·S` · FR-17 knock-out kèm lý do | Thịnh |
| `cv_agent/arbiter/` | FR-18 near-tie · FR-19 nhiều lượt đảo thứ tự, đa số · FR-20 LLM-all | Thịnh |
| `cv_agent/verify/` | FR-21 span có trong CV (exact/fuzzy) · FR-22 NLI · FR-23 sinh lại ≤ 3 vòng, gắn cờ "No evidence" | Thịnh |
| `cv_agent/orchestration/` | FR-24 StateGraph + shared state · FR-25 bật/tắt theo config · FR-26 budget, timeout, fallback | Đán |
| `cv_agent/report/` | FR-27 card L1–L6 (assert Σ L2 = score) · FR-28 contrastive + counterfactual | Thịnh |
| `cv_agent/hitl/` | FR-29 approve/reject/override có lý do + audit log · FR-30 pairwise logistic regression | Sơn |
| `cv_agent/llm/` | FR-31 log mọi lần gọi (tokens, latency, prompt version, kết quả) · chặn PII · cache | Đán |
| `cv_agent/eval/` | Metric, paired bootstrap CI 95%, Wilcoxon, κ, fairness, H0–H6 | Hoa, Vũ |
| `app/streamlit_app.py` | Demo HR (D3) | Sơn |

**Explanation card:** L1 must-have/knock-out · L2 phân rã điểm theo trục (tổng = score) · L3 claim + span đã verify +
id lần gọi LLM · L4 vì sao xếp trên người kế tiếp · L5 nếu có kỹ năng X thì điểm đổi bao nhiêu · L6 chất lượng dữ liệu
(parse confidence, cờ, depth suy luận, ghi chú arbiter, số trường PII đã che).

## Cấu hình thực nghiệm

`configs/base.yaml` là mặc định; `configs/experiments.yaml` ghi đè cho từng cấu hình:

| Cấu hình | Bật | Kiểm |
|---|---|---|
| B0 / B1 / B2 | BM25 / dense / RRF | H1 |
| B3 | B2 + Criteria scoring | H2 |
| B4 | B3 + rerank + LLM sinh claim top 10, pipeline cố định | control H0 |
| LLM-all | B4 + LLM xét **mọi** cặp kề nhau trong top 20 (cùng giao thức với arbiter; `listwise` là tuỳ chọn) | baseline H3 |
| P6 | Full agent: clarification + query expansion + Selective Arbiter + verify_evidence | H0, H3, H4 |
| A1 | P6 bỏ verify_evidence | H4 |
| A2 | P6 thay Criteria Profile bằng trọng số đều + prompt chung | H2 |
| P6-noclar | P6 tắt clarification (phụ trợ cho H6) | H6 |
| **AG** | Cùng thành phần P6 nhưng **LLM planner tự chọn công cụ**, critic kiểm mục tiêu | H7 |
| AG-rules | Cùng công cụ, mục tiêu, budget; planner quy tắc (đối chứng) | H7 |

Các backend thay được qua config:

| Thành phần | Mặc định (chạy được trên laptop) | Thay bằng |
|---|---|---|
| LLM | `ollama` · `qwen2.5:7b`, temperature 0, seed cố định, JSON | `openai_compat` (vLLM), `heuristic` (chỉ test) |
| Dense encoder | `hashing` (char n-gram + khái niệm taxonomy) | `ollama` (`ollama pull bge-m3`), `sentence_transformers` (multilingual-e5) |
| Reranker | `lexical` (độ phủ yêu cầu) | `cross_encoder` (bge-reranker) |
| NLI | `lexical` (thực thể/số trong claim phải có trong span) | `hf` (cross-encoder MNLI) |

## Agent (cấu hình AG): LLM tự quyết bước tiếp theo

P0–P6 là *agentic workflow*: thứ tự bước và mọi nhánh rẽ do quy tắc quyết định. Cấu hình **AG** biến hệ thống thành
agent có giới hạn (`cv_agent/agent/`):

```
criteria (+ bộ nhớ) ─► [ planner chọn 1 công cụ ─► guardrail kiểm tra ─► chạy ─► critic chấm lại mục tiêu ]* ─► report
```

| Thành phần | File | Vai trò |
|---|---|---|
| Công cụ | `agent/tools.py` | `ask_hr`, `search(extra_terms)`, `rank`, `inspect_cv(cv_id)`, `compare(a, b)`, `verify_claims`, `finish` — mỗi công cụ có guardrail riêng |
| Planner | `agent/planner.py` | LLM đọc tóm tắt state (chỉ id + điểm + trạng thái, không văn bản CV) và trả `{"action", "args", "reason"}`; planner quy tắc là đối chứng và lưới an toàn |
| Critic | `agent/critic.py` | Sau mỗi công cụ: đủ ứng viên đạt must-have? shortlist đã verify claim? còn cặp sát điểm chưa so? Mục tiêu không thể đạt được ghi `unreachable` (báo HR) chứ không chặn |
| Bộ nhớ | `agent/memory.py` | Câu trả lời làm rõ của HR (theo đúng JD) và trọng số HR đã hiệu chỉnh (theo profile), dùng lại ở lần sau |

**Guardrail (LLM không vượt qua được):** không `search` khi còn câu hỏi làm rõ chưa hỏi; HR duyệt trước lần tìm đầu tiên;
`search` chỉ nhận kỹ năng có trong taxonomy; `compare` chỉ cho cặp kề nhau, cùng đạt must-have, chênh < 0.05, trong
vùng shortlist, chưa so và chưa từng đổi chỗ (một ứng viên đi tối đa 1 bậc như arbiter); `finish` chỉ khi critic đạt;
tối đa `agent.max_steps` = 20 bước; quyết định LLM sai/không hợp lệ → quy tắc làm thay và ghi `rules(fallback: …)`;
LLM lỗi/hết budget → fallback, vẫn ra báo cáo. Mọi bước (ai quyết, công cụ, tham số, lý do, kết quả) nằm trong
`agent_log` của `result.json` và bảng "Agent" trên demo.

**Bộ nhớ** chỉ lưu câu trả lời từ HR thật (giao diện), không lưu câu trả lời mô phỏng; tắt trong AG/AG-rules để
thực nghiệm không mang thông tin giữa các JD. Trên demo, sau "Đề xuất trọng số mới" có nút lưu vào bộ nhớ agent.

**H7** (đặt trước khi chạy): AG ≥ AG-rules về nDCG@10 (không thấp hơn quá ε = 0.02), tỷ lệ claim không căn cứ không
cao hơn, và quy tắc phải thay LLM ở ≤ 20% quyết định. Bảng kết quả có thêm `agent_steps`, `planner_fallback_rate`,
`goal_met`.

**Kết quả H7** (28/10/2026, qwen2.5:3b, CPU, 6 JD tổng hợp, `runs/experiments/20261027-233950`):

| Cấu hình | nDCG@10 | Kendall τ | Claim không căn cứ | LLM calls/JD | Bước agent | Quy tắc phải thay LLM | Đạt mục tiêu |
|---|---|---|---|---|---|---|---|
| P6 (workflow cố định) | 0.852 | 0.428 | 0% | 25.0 | — | — | — |
| AG-rules | 0.852 | 0.428 | 0% | 27.3 | 10.5 | — | 6/6 |
| AG (LLM planner) | 0.851 | 0.424 | 0% | 41.5 | 12.2 | **24%** | 6/6 |

* **H7 bác bỏ**, chỉ vì tiêu chí độ tin cậy: quy tắc phải thay 24% quyết định của planner 3b (ngưỡng 20%); về chất
  lượng AG đạt (nDCG chênh −0.001, CI [−0.001, 0]; claim không căn cứ như nhau) và đạt mục tiêu ở cả 6 JD.
* Planner LLM tốn thêm ~14 lời gọi/JD (CI [10, 20]) mà không cải thiện chất lượng trên dữ liệu tổng hợp — ở đây các
  quyết định "đúng" gần như đã được quy tắc mã hoá sẵn. Lợi ích kỳ vọng của agent (tự mở rộng truy vấn, chọn cặp cần
  so, dừng sớm) cần kiểm trên Gold Set thật và với model mạnh hơn (7b/GPU).
* Thời gian/JD trong lô này không so được: P6/AG-rules dùng lại cache LLM của các lần chạy trước, AG thì không (prompt
  planner mới).
* Lần chạy thử đầu với 3b (trước các bản sửa) cho thấy đúng các rủi ro guardrail phải chặn: LLM đưa "thuật ngữ" như
  `depth >= 2` vào `search` (bị loại vì không có trong taxonomy), chọn cặp không hợp lệ, viết `"finish {}"`, và đẩy một
  ứng viên xuống nhiều bậc qua các lần `compare` liên tiếp (đã chặn: mỗi ứng viên chỉ đổi chỗ 1 lần).

**Kết quả H7 với qwen2.5:7b** (28/10/2026, CPU, 6 JD, `runs/experiments/20261028-072307`; mọi bước LLM — planner,
judge, claim — đều dùng 7b, nên số liệu không so trực tiếp với bảng 3b):

| Cấu hình | nDCG@10 | Kendall τ | Claim không căn cứ | LLM calls/JD | Bước agent | Quy tắc phải thay LLM | Đạt mục tiêu |
|---|---|---|---|---|---|---|---|
| AG-rules | 0.839 | 0.425 | 0% | 29.0 | 10.5 | — | 6/6 |
| AG (LLM planner) | 0.831 | 0.422 | 0% | 55.8 | 19.5 | **6%** | **3/6** |

* Ban đầu H7 được ghi là chấp nhận vì nDCG chênh −0.009 nằm trong ε = 0.02. Theo quy tắc CI (mục "Quy tắc kiểm
  định" bên dưới) thì **H7 bác bỏ**: cận dưới CI [−0.024, 0] vượt −ε. Claim không căn cứ như nhau, quy tắc chỉ phải
  thay 6% quyết định (3b: 24%). Planner 7b hiểu định dạng và công cụ tốt hơn hẳn.
* Nhưng planner 7b **làm thừa việc**: gần gấp đôi lời gọi LLM (+27/JD, CI [20, 32]), 5/6 JD chạm trần 20 bước nên chỉ
  3/6 kết thúc với critic đạt. Mẫu lặp lại: rank → dùng hết 6 lượt `compare` → `search` lại (dù đủ ứng viên đạt
  must-have) → rank → lại 6 lượt `compare` → hết bước. Nguyên nhân thiết kế: `search` đặt lại `compared_pairs`, nên
  ngân sách `max_compares` thực tế được nạp lại sau mỗi lần tìm. Ở bank_api_qa planner gọi `search` 4 lần liên tiếp
  với cùng bộ thuật ngữ.
* Kết luận: với 7b, agent đáng tin về mặt *hợp lệ* nhưng chưa *tiết kiệm*, và chưa chứng minh được không kém AG-rules.

**Sau khi sửa ngân sách** (28/10/2026, 7b, `runs/experiments/20261028-211316`): `max_compares` tính trên cả lượt chạy
(bộ đếm `compares` không bị `search` đặt lại); `search` bị chặn khi pool đã đủ ứng viên đạt must-have hoặc không còn
thuật ngữ mở rộng mới, và lần tìm lặp lại đúng bộ thuật ngữ cũ bị từ chối; critic không gợi ý một lần tìm bị chặn.

| Cấu hình | nDCG@10 | Kendall τ | Claim không căn cứ | LLM calls/JD | Bước agent | Quy tắc phải thay LLM | Đạt mục tiêu |
|---|---|---|---|---|---|---|---|
| AG-rules | 0.839 | 0.425 | 0% | 28.3 | 8.8 | — | 6/6 |
| AG (LLM planner) | **0.840** | 0.426 | 0% | 42.5 | 13.8 | 8% | **6/6** |

* H7 chấp nhận, kể cả theo quy tắc CI: nDCG chênh +0.001 (CI [−0.001, +0.003]), không còn JD nào chạm trần bước; so với lần trước, AG giảm
  19.5 → 13.8 bước và 55.8 → 42.5 lời gọi LLM/JD, đạt mục tiêu 3/6 → 6/6.
* Phần thừa còn lại: sau khi xong việc, planner 7b gần như luôn dùng hết 5 lượt `inspect_cv` (chỉ đọc, nhưng mỗi
  bước là một lời gọi planner) trước khi `finish`, nên vẫn tốn thêm ~14 lời gọi/JD so với AG-rules.

```bash
python -m cv_agent rank --config AG --model qwen2.5:7b          # một JD, in từng bước của agent
python -m cv_agent experiments --configs AG,AG-rules,P6 --llm ollama --model qwen2.5:7b
```

## Hiệu năng trên laptop CPU (NFR-04)

Đo bằng `python -m cv_agent bench` (P6, 200 CV, 6 JD, **tắt cache LLM**, Ollama trên CPU không GPU, ngày 26/10/2026).
p95 với 6 JD gần như là giá trị lớn nhất — cần đo lại trên Gold Set thật.

| Phiên bản | Model | Thời gian/JD p50 | p95 | Pairwise p50 | Claim p50 | Claim bị verify bác ở lần đầu |
|---|---|---|---|---|---|---|
| Ban đầu (prompt đầy đủ, ngưỡng 0.05) | 7b | > 15 phút (timeout 900s → fallback) | — | ~37s | — | — |
| Hiện tại | **qwen2.5:3b** | **197s** | **241s** | 6.8s | 10.1s | 39% (92/238) |
| Hiện tại | qwen2.5:7b | 406s | 478s | 14.3s | 20.0s | 12% (28/240) |

Những thay đổi tạo ra khác biệt:
1. **Ngưỡng near-tie chọn trên dev** (`python -m cv_agent tune`): trên dev mọi ngưỡng từ "chỉ hoà tuyệt đối" tới 0.05 cho
   cùng nDCG@10 và cùng số cặp, nên quy tắc đặt trước (rẻ nhất, rồi nhỏ nhất) chọn `1e-9`. Ghi chú: điểm Criteria hay
   bão hoà ở top nên phần lớn near-tie là hoà tuyệt đối.
2. **`judge.max_rank: 10`** — arbiter chỉ xét các cặp có thể làm đổi shortlist top 10 (từ ~16 cặp xuống 0–7 cặp/JD).
3. **Prompt gọn**: CV → các dòng chứa bằng chứng (nguyên văn), JD → danh sách must/nice; output JSON ngắn và có
   `num_predict` (pairwise ~810 → ~500 token vào, ~90 → ~30 token ra).
4. **Claim trích dẫn theo số dòng** + vòng sửa chỉ hỏi lại claim bị bác: hết lỗi "span không có trong CV" do diễn giải.

Khuyến nghị: demo trên laptop CPU dùng `--model qwen2.5:3b` (khoảng 3–4 phút/JD); chạy thực nghiệm báo cáo dùng 7b
(chậm gấp đôi nhưng tỷ lệ claim sai ít hơn ~3 lần) hoặc GPU. Arbiter đổi chỗ 0–4 cặp/JD nhưng không làm đổi nDCG@10
trên dữ liệu tổng hợp (các cặp hoà thường cùng mức nhãn) — đây là finding cần kiểm lại trên Gold Set thật.

```bash
python -m cv_agent tune --llm ollama --model qwen2.5:3b --split dev
python -m cv_agent bench --models qwen2.5:3b,qwen2.5:7b --split all
python -m cv_agent rank --jd fpt_automation_test --config P6 --model qwen2.5:3b
```

## Kết quả với mô hình neural thật (FR-12/15/22)

Chạy ngày 27/10/2026: `python -m cv_agent experiments --neural --llm ollama --model qwen2.5:3b`, CPU, dữ liệu
**tổng hợp** (200 CV, 6 JD). Thành phần: `intfloat/multilingual-e5-base`, `BAAI/bge-reranker-base`,
`cross-encoder/nli-deberta-v3-base` (tên mô hình thực sự đã nạp ghi trong `summary.json` → `components_loaded`).

| Cấu hình | nDCG@10 | Claim không có căn cứ | LLM calls/JD | Thời gian/JD |
|---|---|---|---|---|
| B0 BM25 | 0.567 | — | 0 | <0.1s |
| B1 e5 dense | 0.412 | — | 0 | 5s |
| B2 RRF | 0.533 | — | 0 | 5s |
| B3 + Criteria | **0.827** | — | 0 | 7s |
| B4 + cross-encoder + claim | 0.631 | 44.5% | 10 | 112s |
| A1 = P6 bỏ verify | 0.631 | 45.0% | 14 | 45s |
| P6 | 0.631 | **17.4%** | 23 | 125s |

* **H4 chấp nhận** với NLI thật: verify_evidence giảm claim không có căn cứ từ 45% xuống 17%
  (CI 95% [-0.33, -0.22], Wilcoxon p = 0.031). H0 chấp nhận nhờ tiêu chí này (nDCG P6 ≈ B4), không có fallback.
* **Cross-encoder làm B4 kém hơn B3** (0.631 vs 0.827): B4 cắt cứng top 20 theo reranker *trước* khi chấm
  tiêu chí. Ở JD FPT, top 20 của bge-reranker-base chỉ chứa 5 CV phù hợp (nhãn ≥ 2), lexical 17, Criteria 18.
  `bge-reranker-v2-m3` (đa ngôn ngữ) cũng chỉ 5/20 và chậm ~2 phút/JD, nên không phải lỗi ngôn ngữ.
  Đầu vào đã đổi sang danh sách yêu cầu JD + dòng bằng chứng CV (cửa sổ 512 token cắt mất CV) — cải thiện ít.
* **Tuỳ chọn `rerank.mode: within_criteria`** (`--rerank-mode within_criteria`, nay là mặc định): Criteria xếp toàn bộ pool trước,
  cross-encoder chỉ chấm top 20 theo Criteria và chỉ dùng để phá hoà điểm — không loại ai. So với `filter` (FR-15,
  trước đây là mặc định), cùng mô hình, qwen2.5:3b, 6 JD:

  | | nDCG@10 `filter` | nDCG@10 `within_criteria` | MRR@10 `within_criteria` | Claim không căn cứ `within_criteria` | Thời gian/JD |
  |---|---|---|---|---|---|
  | B3 (không rerank) | 0.827 | 0.827 | 0.917 | — | 5s |
  | B4 | 0.631 | **0.835** | 1.000 | 47.6% | 71s |
  | P6 | 0.631 | **0.833** | 1.000 | **10.5%** | 122s |

  Với `within_criteria`, cross-encoder không còn làm hỏng ranking và nhỉnh hơn B3 một chút (+0.008 nDCG, MRR 0.917 →
  1.0) — mức chênh này nằm trong nhiễu với 6 JD. JD FPT: B3 0.951, B4 0.874, P6 0.921 (arbiter kéo lại một phần).
  Từ 27/10/2026 `within_criteria` là mặc định — lý do ở mục [Thay đổi so với FR-15](#thay-đổi-so-với-fr-15).
* B3 tăng từ 0.719 lên ~0.83 so với lần chạy 26/10 do sửa tie-break theo `potential_score` (chạy lại không bật
  neural cũng ra 0.829), không phải do mô hình neural — bank_api_qa từ thứ tự gần ngẫu nhiên lên 0.861.
* B1 (e5) kém B0 (BM25): CV tổng hợp sinh từ template nên độ tương đồng ngữ nghĩa gần như bằng nhau (~0.84–0.86).
* **Thiên lệch của dữ liệu tổng hợp:** nhãn gold sinh từ cùng bộ quy tắc kỹ năng mà Criteria scoring dùng, nên
  B3/lexical được lợi. Kết luận về FR-12/15 phải kiểm lại trên Gold Set thật; đề xuất cho nhóm: dùng cross-encoder
  để *xếp lại trong* top-k theo tiêu chí thay vì *lọc* top 20 (thay đổi thiết kế thực nghiệm, cần nhóm quyết).
* H5 vẫn bác bỏ khi xét từng JD: bank_api_qa 0.72, ambiguous_senior 0.62, ambiguous_junior 0.40 < 0.8.
* Mặc định `configs/base.yaml` vẫn là hashing/lexical (chạy được không cần GPU); bật neural bằng `--neural`.

### Lượt chạy đầy đủ 28/10/2026 (dùng cho slide 28–37)

`experiments --neural --llm ollama --model qwen2.5:3b`. Gold Set tổng hợp ở `runs/experiments/20261028-220112`,
dữ liệu công khai ở `runs/experiments/20261028-232304`.

| nDCG@10 | B0 | B1 | B2 | B3 | B4 | LLM-all | P6 | A1 | A2 | P6-noclar |
|---|---|---|---|---|---|---|---|---|---|---|
| Gold Set tổng hợp (6 JD, 202 CV) | 0.567 | 0.412 | 0.533 | 0.827 | 0.835 | 0.845 | 0.833 | 0.833 | 0.867 | 0.845 |
| HF resume-job-description-fit (6 JD, 82 CV) | 0.578 | **0.631** | 0.627 | 0.595 | 0.588 | — | 0.566 | — | — | — |

* Hypothesis trên Gold Set (quy tắc CI): H0 và H4 được chấp nhận; H1, H2, H3, H5, H6 bị bác bỏ.
  * H3: P6 so với LLM-all chênh −0.012 nDCG@10, với 8.0 so với 20.7 judge call/JD (CI của số call [−19.7, −5.7]).
    Trung bình nằm trong ε = 0.02 nhưng CI [−0.045, +0.010] chưa loại trừ mức kém hơn ε, nên chưa chứng minh được
    "không kém hơn". Bản đầu tiên của README và slide xét trung bình nên ghi nhầm là chấp nhận.
  * H4: claim không căn cứ 10.5% so với 41.2%, p = 0.03.
* A2 (prompt chung) có nDCG@10 cao nhất (0.867), nhưng Kendall τ của nó không thấp hơn P6. Vì vậy H2 không được ủng hộ.
* **Dữ liệu công khai không lặp lại được lợi thế ở trung bình.** Dense B1 tốt nhất; P6 (0.566) thấp hơn BM25.
  Phân tích lỗi theo JD (mục "Phân tích lỗi" bên dưới) tìm ra nguyên nhân chính.
* SkillSpan test: từ điển taxonomy có precision khớp chính xác 0.706, recall 0.074 (trên các span thuộc taxonomy
  recall là 0.866). Tagger JobBERT học trên SkillSpan train đạt span F1 0.592 (mục "Tagger skill" bên dưới).
* Slide: `scripts/build_results_slides.py` tạo `Slide_Group2_NLP501-3.pptx` từ các thư mục run này. Bản gốc `-2` giữ nguyên.

## Fine-tune trên Colab (thí nghiệm tuỳ chọn kiểu ConFit)

Phần huấn luyện nằm ở `cv_agent/training/finetune.py`. Notebook [`notebooks/finetune_colab.ipynb`](notebooks/finetune_colab.ipynb)
được **sinh từ chính module đó** (`python scripts/build_colab_notebook.py`). Notebook ghi nguyên văn mã nguồn vào Colab, nên
không cần repo hay dữ liệu trên máy. Test `tests/test_finetune.py` báo lỗi khi notebook lệch với code.

1. Mở notebook trên Colab, chọn *Runtime → Change runtime type → T4 GPU*, rồi chọn *Run all*.
2. Tải `rjf_finetuned_models.zip` về và giải nén vào `models/`.
3. Chạy lại thực nghiệm với `--dense-model` và `--rerank-model` như trong bảng lệnh ở trên.

| Mô hình | Gốc | Huấn luyện |
|---|---|---|
| Dense (B1, B2, retrieval của P6) | `intfloat/multilingual-e5-base` | Triplet (JD, Good Fit, No Fit) với CachedMultipleNegativesRankingLoss: 16 in-batch negative không trùng, forward theo mini-batch 4 cho đỡ tốn bộ nhớ |
| Cross-encoder (B4, P6) | `BAAI/bge-reranker-base` | BCE, nhãn No Fit / Potential / Good = 0 / 0.5 / 1 |

**Dữ liệu và chống rò rỉ.** Nguồn là `cnamuangtoun/resume-job-description-fit`, tải thẳng từ Hugging Face (NFR-02).
- Bỏ mọi JD QA / Automation Test, vì đó chính là các JD đánh giá trong `data/public`.
- Bỏ mọi CV từng được gán cho các JD đó.
- Phần còn lại của `train.csv` chia train / dev theo JD.
- `test.csv` đã lọc, bỏ cả JD đã gặp, chỉ dùng để kiểm tra cuối.
- Gold Set tổng hợp không bao giờ được dùng để huấn luyện.

Mỗi thư mục mô hình có `finetune_metrics.json`, ghi cấu hình, số cặp và nDCG@10 trước / sau trên dev / test.

### Kết quả fine-tune 29/10/2026 (1 epoch, max_len 256, batch 16)

Dữ liệu sau khi lọc: train 3966 cặp / 248 JD, dev 445 cặp / 28 JD, test 1149 cặp / 69 JD.

| Mô hình | Ví dụ train | nDCG@10 dev (trước → sau) | nDCG@10 test (trước → sau) |
|---|---|---|---|
| Dense e5-base | 956 triplet | 0.807 → 0.827 | 0.749 → 0.797 |
| Cross-encoder bge | 3966 cặp | 0.682 → 0.809 | 0.629 → 0.707 |

Chạy lại thực nghiệm với hai mô hình này: Gold Set ở `runs/experiments/20261029-105306`, dữ liệu công khai ở `runs/experiments/20261029-111458`.

| nDCG@10 | B0 | B1 | B2 | B3 | B4 | P6 |
|---|---|---|---|---|---|---|
| Gold Set: mô hình gốc | 0.567 | 0.412 | 0.533 | 0.827 | 0.835 | 0.833 |
| Gold Set: fine-tune | 0.567 | 0.480 | 0.607 | 0.820 | 0.838 | 0.835 |
| Công khai: mô hình gốc | 0.578 | 0.631 | 0.627 | 0.595 | 0.588 | 0.566 |
| Công khai: fine-tune | 0.578 | **0.699** | 0.675 | 0.595 | 0.588 | 0.577 |

* Fine-tune giúp rõ nhất ở dense:
  * Gold Set: B1 tăng +0.068, B2 tăng +0.074.
  * Dữ liệu công khai: B1 tăng +0.068 và vượt BM25. Điều này cùng hướng với ConFit.
* Trên Gold Set, dense fine-tune vẫn thua BM25 và Criteria Profile.
* B3 / B4 / P6 gần như không đổi, vì thứ hạng ở các cấu hình này do Criteria Profile quyết định.
* Trên dữ liệu công khai, B4 không đổi. Cross-encoder tốt lên trên test của nó, nhưng trên 6 JD QA vẫn chưa đổi được top 10 của B4.
* Chỉ có 6 JD, nên đây là quan sát chứ không phải kết luận thống kê.
* Ở hai lượt này chỉ chạy B0–B4 và P6, nên các hypothesis cần A1 / A2 / LLM-all hiện "BÁC BỎ" vì thiếu cấu hình. Kiểm định hypothesis lấy từ lượt 28/10.


## Tagger skill học trên SkillSpan (FR-03)

Từ điển taxonomy chỉ biết thuật ngữ Automation Test nên recall trên SkillSpan thấp là đúng thiết kế. Để có một mốc
so sánh với tài liệu, `cv_agent/training/skill_tagger.py` fine-tune `jjzha/jobbert-base-cased` thành bộ gán nhãn BIO:

* Dữ liệu: `jjzha/skillspan` (CC-BY-4.0), train 4,800 / dev 3,174 / test 3,569 câu. Hai lớp skill và knowledge gộp
  thành một tập nhãn (`B/I-SKILL`, `B/I-KNOWLEDGE`, `O`); khi chồng nhau thì skill thắng.
* Chọn mô hình: epoch có span F1 khớp chính xác cao nhất trên dev; test chỉ chấm một lần sau khi huấn luyện.
* Cùng cách chấm span với từ điển (`cv_agent/eval/span_scoring.py`).

Kết quả 29/10/2026 (3 epoch, batch 16, max_len 128, seed 42, CPU ~1h50):

| SkillSpan test | Precision | Recall | F1 khớp chính xác | F1 chồng lấn |
|---|---|---|---|---|
| Từ điển taxonomy | 0.706 | 0.074 | 0.134 | 0.155 |
| Tagger JobBERT | 0.559 | 0.628 | **0.592** | **0.873** |
| Hai bộ hợp lại | 0.549 | 0.634 | 0.589 | 0.870 |

* Dev F1 theo epoch: 0.494 → 0.568 → 0.578.
* Bài báo SkillSpan (Zhang et al. 2022, Bảng 5) báo cáo JobBERT 59.7 ± 0.4 (trung bình 5 seed). Kết quả 1 seed của
  nhóm nằm trong khoảng đó; nhóm nói "gần với bài báo", không nói "vượt".
* Gộp hai lớp thành một nhãn mỗi token nên span knowledge nằm trong span skill bị mất. Đây là trần recall của cách
  gộp; bài báo dùng hai đầu ra riêng.
* Hợp tagger với từ điển không tăng F1: từ điển chỉ thêm ít span đúng (recall +0.006) nhưng làm precision giảm.
* Hệ thống vẫn dùng từ điển để gắn concept ID làm bằng chứng. Tagger dùng để báo skill ngoài từ điển, không thay
  bước matching.
* Notebook Colab có ô `TRAIN_TAGGER` chạy cùng bước này trên GPU.

## Thay đổi so với FR-15

FR-15 viết: cross-encoder rerank top-k → **top 20**, sau đó mới chấm Criteria (FR-16). Từ 27/10/2026 mặc định là
`rerank.mode: within_criteria`: Criteria chấm **toàn bộ** pool, cross-encoder chỉ chấm 20 người đứng đầu theo
Criteria và chỉ dùng để phân thứ tự những người bằng điểm. Muốn chạy đúng FR-15 như văn bản:
`--rerank-mode filter` (hoặc `rerank.mode: filter` trong config).

Lý do:
1. **Lọc cứng làm mất ứng viên tốt mà Criteria không cứu được.** Ở JD FPT, top 20 của `bge-reranker-base` chỉ có
   5 CV phù hợp (nhãn ≥ 2), so với 18 của Criteria; `bge-reranker-v2-m3` (đa ngôn ngữ) cũng chỉ 5. Cross-encoder
   pretrained đo độ giống nội dung JD–CV, không đo mức đáp ứng must-have/số năm/độ sâu automation.
2. **Kết quả đo được** (neural, qwen2.5:3b, 6 JD): B4 nDCG@10 0.631 → 0.835, P6 0.631 → 0.833; MRR@10 1.0;
   claim không căn cứ ở P6 10.5%. `filter` làm B4/P6 kém cả B3 không rerank (0.827).
3. **Giữ đúng tinh thần FR-15/FR-16:** reranker vẫn chạy trên top 20 và vẫn ảnh hưởng thứ tự; điểm cuối vẫn là
   `Score = 1[H(c)] × Σ w·S` (L2 của card vẫn cộng đúng bằng score); không ứng viên nào bị loại khỏi danh sách.
4. **Rẻ hơn:** cross-encoder chỉ chấm 20 cặp/JD thay vì cả pool (~200), B4 từ 112s xuống 71s/JD.

Giới hạn: số liệu trên là dữ liệu **tổng hợp** với nhãn sinh từ cùng quy tắc kỹ năng mà Criteria dùng, nên có lợi
cho cách xếp theo Criteria. Phải chạy lại cả hai chế độ trên Gold Set thật trước khi chốt trong báo cáo (D4);
mức hơn B3 (+0.008 nDCG) nằm trong nhiễu với 6 JD. Báo cáo cần ghi rõ B4/P6 dùng `within_criteria`.

## Quy tắc kiểm định

* Mọi hypothesis được quyết định theo **CI 95% paired bootstrap theo JD**, không theo trung bình đơn lẻ
  (`summary.json` → `hypothesis_rule: ci95`, mỗi hypothesis có trường `rule`).
  * Hơn / kém có ý nghĩa: cả CI nằm trên / dưới 0 (H0, H2, H4, H6, H1).
  * Không kém hơn (H3, H7): cận dưới CI của Δ nDCG@10 ≥ −ε (ε = 0.02).
  * H7 còn yêu cầu cận trên CI của mức tăng claim không căn cứ ≤ +0.02 và tỷ lệ luật phải thay planner đo được, ≤ 20%.
  * Cần ít nhất 5 JD ghép cặp; ít hơn thì hypothesis không được chấp nhận (H6 chỉ có JD mơ hồ nên rơi vào trường hợp này).
* Với n = 6 JD, Wilcoxon hai phía không thể cho p nhỏ hơn 2/2⁶ ≈ 0.031 (`wilcoxon.p_floor`). p = 0.03 chỉ có nghĩa là
  cả 6 JD cùng chiều; vì vậy kết luận dựa vào CI, còn p chỉ để tham khảo.
* `python -m cv_agent rescore <run>` tính lại hypothesis từ `results.csv` khi quy tắc thay đổi; bản đầu tiên được giữ ở `summary.pre-rescore.json`. Các lượt 27–29/10 đã
  được tính lại: H3 (Gold Set 28/10) và H7 (7b, `20261028-072307`) đổi từ chấp nhận sang bác bỏ, các kết luận khác giữ nguyên.

## Phân tích lỗi trên dữ liệu công khai (post hoc)

`python -m cv_agent error-analysis runs/experiments/20261028-232304` ghi `error_analysis.json`: nDCG@10 theo từng
JD, các JD trùng lặp (Jaccard 3-gram từ ≥ 0.8) và trung bình khi mỗi nhóm trùng chỉ tính một lần. Nhóm được xác định
từ văn bản JD, không từ kết quả, nhưng phân tích vẫn làm **sau khi thấy kết quả** nên chỉ là gợi ý.

* 3/6 JD (`…2d30eda156`, `…4c8a607c3c`, `…5412e21ef1`) là **cùng một tin tuyển "Software Developer (Playbook,
  Ansible)"** đăng ở ba thành phố. Tin này lọt bộ lọc QA nhờ hai cụm lỏng ("automation testing", "test cases"), nhưng
  không phải vai trò Automation Test, nên Criteria Profile áp tiêu chí sai: B3 0.301 so với BM25 0.409.
* Trên 3 JD còn lại (đúng vai trò QA): B3 0.889, B1 0.827, BM25 0.747, P6 0.845.
* Khi mỗi tin trùng chỉ tính một lần (n hiệu dụng = 4): B3 0.742, B4 0.739, B1 0.729, P6 0.705, BM25 0.662.
* P6 có tỷ lệ claim không căn cứ 0.29–0.50 trên CV công khai, so với 0.04–0.18 trên CV tổng hợp: verify_evidence kém
  hơn trên CV thật (văn bản dài, định dạng tự do).
* Nhóm **không đổi tập đánh giá** sau khi thấy kết quả. Việc cần làm: lọc JD theo tiêu đề, bỏ tin trùng, rồi chạy lại.

## Threats to validity

* **Gold Set tổng hợp và Criteria Profile cùng do nhóm viết.** Bước nhảy B2 → B3 (0.533 → 0.827) có thể một phần đến từ
  việc profile "biết" cách gán nhãn. Trên dữ liệu công khai (nhãn do người khác gán), B3 thấp hơn B1 ở trung bình
  6 JD nhưng cao nhất trên 3 JD đúng vai trò QA (0.889 so với 0.827): profile có giúp khi đúng vai trò, nhưng đây là
  phân tích post hoc trên 3 JD.
  Cần Gold Set thật do người gán nhãn độc lập với người viết profile, kèm Cohen κ giữa 2 người gán.
* **n = 6 JD mỗi lượt**: CI rộng; một JD có thể đổi kết luận. Trên dữ liệu công khai, 3 JD là cùng một tin nên n hiệu
  dụng chỉ là 4.
* **Bộ lọc JD công khai dựa trên từ khoá** để lọt một tin không phải QA (xem "Phân tích lỗi").
* **Fine-tune 1 seed**: chưa có độ lệch chuẩn. Notebook mặc định chạy 3 seed (`SEEDS`), CLI có `--seeds`.
* **LLM nhỏ (Qwen 3B / 7B trên CPU)**: kết luận về LLM-all, arbiter và planner có thể khác với LLM lớn hơn.
* **Impact ratio** đo trên CV tổng hợp; tên đã bị che nên lệch đến từ thuộc tính khác tương quan với nhóm.

## Lưu ý trung thực về kết quả

* `data/synthetic/` là **CV tổng hợp** với nhãn mô phỏng 2 annotator — dùng để chạy end-to-end và kiểm tra harness
  trước khi có Gold Set thật (D2). Báo cáo kết quả trên dữ liệu này **tách riêng** khỏi Gold Set thật.
* Backend `heuristic` **mô phỏng** LLM (kể cả tỷ lệ "bịa" claim để thử verify_evidence). Kết quả H3/H4 chạy với nó
  **không phải** kết quả LLM — kết quả báo cáo phải chạy với `--llm ollama` (hoặc vLLM).
* `hashing` encoder và `lexical` reranker/NLI là fallback không phụ thuộc mô hình; khi báo cáo B1/B4/H4 với chúng phải ghi rõ.
* Ngưỡng near-tie, `top_k` và ε (H3) phải chọn trên split `dev` rồi cố định trước khi chạy `--split test`.
* ID khái niệm dùng namespace `esco-lite:`; nạp ESCO v1.1 thật bằng `cv_agent.ingest.taxonomy.load_esco_csv`
  (file `skills_en.csv`, cần cột `conceptUri`, `preferredLabel`, `altLabels`).
* H5 chỉ được chấp nhận khi **mọi** JD đạt quy tắc 4/5 (`summary.json` → `failing_jds`), không xét trung bình.
* Khi không ứng viên nào qua must-have, shortlist xếp theo `potential_score` (điểm nếu bỏ knock-out) để HR xem ai gần đạt nhất.
* Mô hình neural (`--neural`) tải từ Hugging Face lần đầu (~3 GB); đặt `HF_HOME` sang ổ còn trống nếu ổ C đầy.

## Định dạng Gold Set thật

```
data/<gold_set>/cvs/*.pdf|docx|txt
data/<gold_set>/gold.json   {jd_id: {"relevance": {cv_id: 0..3}, "annotator_a": {...}, "annotator_b": {...},
                                     "hr_ranking": [cv_id, ...], "ambiguous": bool, "split": "dev"|"test"}}
data/<gold_set>/meta.json   {"rationale": {cv_id: [span, ...]}, "groups": {cv_id: {"gender": "M"|"F"}}, "names": {cv_id: name}}
```
JD đặt trong `data/jds/jds.yaml` (`hr_answers` = câu trả lời của HR cho câu hỏi làm rõ, dùng cho H6). Chạy:
`python -m cv_agent experiments --data data/<gold_set> --split test`.
