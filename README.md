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

* **H7 chấp nhận** theo tiêu chí đặt trước: nDCG chênh −0.009 (CI [−0.024, 0], trong ε = 0.02), claim không căn cứ
  như nhau, quy tắc chỉ phải thay 6% quyết định (3b: 24%). Planner 7b hiểu định dạng và công cụ tốt hơn hẳn.
* Nhưng planner 7b **làm thừa việc**: gần gấp đôi lời gọi LLM (+27/JD, CI [20, 32]), 5/6 JD chạm trần 20 bước nên chỉ
  3/6 kết thúc với critic đạt. Mẫu lặp lại: rank → dùng hết 6 lượt `compare` → `search` lại (dù đủ ứng viên đạt
  must-have) → rank → lại 6 lượt `compare` → hết bước. Nguyên nhân thiết kế: `search` đặt lại `compared_pairs`, nên
  ngân sách `max_compares` thực tế được nạp lại sau mỗi lần tìm. Ở bank_api_qa planner gọi `search` 4 lần liên tiếp
  với cùng bộ thuật ngữ.
* Kết luận: với 7b, agent đáng tin về mặt *hợp lệ* nhưng chưa *tiết kiệm*; H7 đạt không có nghĩa AG tốt hơn AG-rules.

**Sau khi sửa ngân sách** (28/10/2026, 7b, `runs/experiments/20261028-211316`): `max_compares` tính trên cả lượt chạy
(bộ đếm `compares` không bị `search` đặt lại); `search` bị chặn khi pool đã đủ ứng viên đạt must-have hoặc không còn
thuật ngữ mở rộng mới, và lần tìm lặp lại đúng bộ thuật ngữ cũ bị từ chối; critic không gợi ý một lần tìm bị chặn.

| Cấu hình | nDCG@10 | Kendall τ | Claim không căn cứ | LLM calls/JD | Bước agent | Quy tắc phải thay LLM | Đạt mục tiêu |
|---|---|---|---|---|---|---|---|
| AG-rules | 0.839 | 0.425 | 0% | 28.3 | 8.8 | — | 6/6 |
| AG (LLM planner) | **0.840** | 0.426 | 0% | 42.5 | 13.8 | 8% | **6/6** |

* H7 chấp nhận: nDCG chênh +0.001 (CI [−0.001, +0.003]), không còn JD nào chạm trần bước; so với lần trước, AG giảm
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

### Lượt chạy đầy đủ 28/10/2026 (dùng cho slide 28–35)

`experiments --neural --llm ollama --model qwen2.5:3b`. Gold Set tổng hợp ở `runs/experiments/20261028-220112`,
dữ liệu công khai ở `runs/experiments/20261028-232304`.

| nDCG@10 | B0 | B1 | B2 | B3 | B4 | LLM-all | P6 | A1 | A2 | P6-noclar |
|---|---|---|---|---|---|---|---|---|---|---|
| Gold Set tổng hợp (6 JD, 202 CV) | 0.567 | 0.412 | 0.533 | 0.827 | 0.835 | 0.845 | 0.833 | 0.833 | 0.867 | 0.845 |
| HF resume-job-description-fit (6 JD, 82 CV) | 0.578 | **0.631** | 0.627 | 0.595 | 0.588 | — | 0.566 | — | — | — |

* Hypothesis trên Gold Set: H0, H3, H4 được chấp nhận; H1, H2, H5, H6 bị bác bỏ.
  * H3: P6 so với LLM-all chênh −0.012 nDCG@10, nằm trong ε = 0.02, với 8.0 so với 20.7 judge call/JD.
  * H4: claim không căn cứ 10.5% so với 41.2%, p = 0.03.
* A2 (prompt chung) có nDCG@10 cao nhất (0.867), nhưng Kendall τ của nó không thấp hơn P6. Vì vậy H2 không được ủng hộ.
* **Dữ liệu công khai không lặp lại được lợi thế.** Dense B1 tốt nhất; P6 (0.566) thấp hơn BM25.
  * Criteria Profile được viết cho Gold Set, còn nhãn công khai thô (3 mức) và chỉ gán một phần.
  * Cần viết profile riêng cho từng JD công khai trước khi rút ra kết luận.
* SkillSpan test: precision khớp chính xác 0.706, recall 0.074. Trên các span thuộc taxonomy, recall là 0.866.
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
| Dense (B1, B2, retrieval của P6) | `intfloat/multilingual-e5-base` | Triplet (JD, Good Fit, No Fit) với MultipleNegativesRankingLoss, in-batch negative không trùng |
| Cross-encoder (B4, P6) | `BAAI/bge-reranker-base` | BCE, nhãn No Fit / Potential / Good = 0 / 0.5 / 1 |

**Dữ liệu và chống rò rỉ.** Nguồn là `cnamuangtoun/resume-job-description-fit`, tải thẳng từ Hugging Face (NFR-02).
- Bỏ mọi JD QA / Automation Test, vì đó chính là các JD đánh giá trong `data/public`.
- Bỏ mọi CV từng được gán cho các JD đó.
- Phần còn lại của `train.csv` chia train / dev theo JD.
- `test.csv` đã lọc, bỏ cả JD đã gặp, chỉ dùng để kiểm tra cuối.
- Gold Set tổng hợp không bao giờ được dùng để huấn luyện.

Mỗi thư mục mô hình có `finetune_metrics.json`, ghi cấu hình, số cặp và nDCG@10 trước / sau trên dev / test.


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
