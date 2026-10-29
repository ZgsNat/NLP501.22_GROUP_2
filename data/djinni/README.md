# Gold Set thật từ Djinni Recruitment Dataset

Nguồn: [Djinni Recruitment Dataset](https://aclanthology.org/2024.unlp-1.2) (Drushchak & Romanyshyn, UNLP 2024),
giấy phép **MIT**. Hồ sơ ứng viên trên Djinni ẩn danh từ khi đăng ký; nhóm tác giả còn loại mọi CV còn sót số điện
thoại, email, địa chỉ, liên kết mạng xã hội. File gốc (`raw/`, ~383 MB) không đưa lên git.

## Nội dung thư mục

| File | Nội dung |
|---|---|
| `jds.yaml` | 8 JD "QA Automation" tiếng Anh, mỗi công ty một JD, trải đều 4 stack (Java, Python, JS/TS, C#) và các mức kinh nghiệm |
| `cvs/` | 250 hồ sơ dùng chung cho mọi JD: 150 QA Automation, 70 QA (chủ yếu thủ công), 30 lập trình viên (mẫu nhiễu khó) |
| `meta.json` | Nguồn, giấy phép, trích dẫn, tham số chọn mẫu (seed 42) |
| `labeling/labels_A.xlsx`, `labels_B.xlsx` | Bảng gán nhãn cho hai người gán nhãn (424 cặp JD–CV) |
| `labeling/pool.json` | Danh sách cặp cần gán nhãn và cách chọn |

Tạo lại toàn bộ (kết quả giống hệt với cùng seed):

```bash
python -m cv_agent prepare-djinni
python -m cv_agent label-pool --neural --dense-model models/multilingual-e5-base-rjf-ft --rerank-model models/bge-reranker-base-rjf-ft
```

## Cách chọn cặp cần gán nhãn (pooling)

Gán nhãn 8 × 250 = 2,000 cặp là quá nhiều. Theo cách làm của TREC, mỗi JD chỉ gán nhãn:

* hợp của top 20 từ bốn cấu hình không dùng LLM (B0 BM25, B1 dense, B2 lai ghép, B3 Criteria Profile);
* thêm 10 hồ sơ ngẫu nhiên để có cả những CV không hệ thống nào xếp cao.

Mỗi JD còn 49–59 cặp, tổng 424 cặp. Thứ tự trong bảng đã xáo trộn và khác nhau giữa hai người, nên người gán nhãn
không biết hệ thống nào đề xuất CV nào. Các CV không được gán nhãn bị bỏ qua khi tính độ đo (`judged_only`).

## Quy trình cho nhóm

1. **Chọn hai người gán nhãn** không tham gia viết Criteria Profile (để tránh vòng lặp "tự viết, tự chấm").
2. **Đọc sheet "Hướng dẫn"** trong file và thống nhất cách hiểu thang 0–3 trên 5–10 cặp mẫu *không có* trong bảng
   (ví dụ lấy từ `data/synthetic`) trước khi bắt đầu.
3. **Gán nhãn độc lập**: mỗi người một file, không trao đổi, không xem kết quả của hệ thống. Điền cột "Nhãn (0–3)"
   bằng ô chọn; ghi chú ngắn khi chọn 1 hoặc 2.
4. **Nhập nhãn và đo độ đồng thuận**:

   ```bash
   python -m cv_agent import-labels data/djinni/labeling/labels_A.xlsx data/djinni/labeling/labels_B.xlsx
   ```

   Lệnh in Cohen κ có trọng số bậc hai (cho thang thứ bậc 0–3) và ghi `agreement.json`. Hai nhãn bằng nhau thì giữ
   nguyên; lệch 1 mức thì lấy mức thấp hơn; lệch từ 2 mức trở lên được ghi vào `conflicts.csv`.
5. **Thống nhất các cặp lệch xa**: hai người cùng xem từng cặp trong `conflicts.csv`, chọn nhãn chung, điền vào
   một bản sao của bảng (chỉ cần các dòng đó), rồi chạy lại lệnh ở bước 4 với `--adjudicated <file>`. Khi không còn
   cặp lệch xa, lệnh ghi `gold.json`.
6. **Chạy thực nghiệm** trên Gold Set thật:

   ```bash
   python -m cv_agent experiments --data data/djinni --neural --llm ollama --model qwen2.5:3b --dense-model models/multilingual-e5-base-rjf-ft --rerank-model models/bge-reranker-base-rjf-ft
   ```

## Lưu ý khi báo cáo

* Ghi κ trước và sau khi thống nhất; κ dưới 0.6 nghĩa là hướng dẫn chưa đủ rõ, nên sửa hướng dẫn và gán lại.
* Gold Set này chưa có câu trả lời làm rõ của HR (H6) và chưa có thông tin nhóm nhân khẩu học (H5), nên hai giả
  thuyết này vẫn chỉ kiểm định trên dữ liệu tổng hợp.
* Bốn cấu hình dùng để chọn cặp là B0–B3; các cấu hình dùng LLM chỉ sắp xếp lại top 20 của B3/B4, nên phần lớn CV
  mà chúng đưa lên đầu đã có trong tập được gán nhãn.
