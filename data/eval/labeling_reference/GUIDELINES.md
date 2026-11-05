# Hướng dẫn đánh giá 120 cặp JD–CV (Human Labeling Reference)

Tài liệu này được tạo nhằm hỗ trợ quá trình gán nhãn của người thật (Human Annotation) cho 120 cặp mẫu trong `docs/labeling/index.html`.

---

## 1. Thang điểm chuẩn (0 – 3)

| Điểm | Ý nghĩa | Tiêu chuẩn đánh giá thực tế |
|:---:|---|---|
| **0** | **Không phù hợp** (Not relevant) | Thiếu công cụ bắt buộc (must-have), hoặc kỹ năng chỉ được liệt kê trong danh sách ("skills list") mà không có dự án thực tế chứng minh, hoặc bị rào đón (*hedged* - ví dụ: "đang tự học", "được đào tạo 2 tuần", "không dùng trong thực tế"). |
| **1** | **Yếu / Liền kề** (Weak / Adjacent) | Có một số kỹ năng liên quan hoặc công cụ thay thế nhưng không khớp trọng tâm JD, kinh nghiệm quá non so với yêu cầu, hoặc chỉ làm các tác vụ cơ bản (chạy script có sẵn, L1). |
| **2** | **Phù hợp một phần** (Would phone-screen) | Đạt hầu hết các yêu cầu cốt lõi (must-have), có bằng chứng thực tế trong dự án (L2/L3), xứng đáng được gọi phỏng vấn sơ loại (phone-screen). |
| **3** | **Rất phù hợp** (Would interview) | Khớp chặt chẽ cả công cụ lẫn chiều sâu (L3/L4 - tự thiết kế framework/kiến trúc test), số năm kinh nghiệm vững vàng, có dẫn chứng rõ ràng bằng số liệu/dự án cụ thể. |

---

## 2. Các bẫy cần lưu ý khi chấm (Trap Detection)

1. **Bẫy nhồi từ khóa (Keyword Stuffing - CV-004, CV-005, CV-006):**
   - Ứng viên liệt kê hàng loạt công cụ (Selenium, Playwright, Cypress, Appium, JMeter...) ở phần tóm tắt kỹ năng, nhưng phần kinh nghiệm thực tế lại chỉ làm kiểm thử thủ công (manual testing) hoặc chỉ viết test case, chạy test có sẵn.
   - **Xử lý:** Cho điểm **0** hoặc **1** tùy theo mức độ thiếu hụt bằng chứng thực tế.

2. **Bẫy rào đón (Hedging traps):**
   - CV có chứa cụm từ rào đón: *"currently learning"*, *"familiar with"*, *"used in 2-week course"*, *"not used in production"*.
   - **Xử lý:** Không tính đây là kỹ năng thành thạo; nếu đó là công cụ must-have của JD thì đánh giá là knockout hoặc chấm điểm **0**.

3. **Bẫy thổi phồng chức danh (Title inflation):**
   - Chức danh ghi "Automation Test Lead / Architect" nhưng các gạch đầu dòng mô tả công việc chỉ dừng lại ở mức L1 (chạy script kiểm thử, báo cáo bug).
   - **Xử lý:** Đánh giá dựa trên hành động thực tế (action verbs) thay vì chức danh.

4. **CV tiếng Việt hoặc không dấu (CV-031 trở đi):**
   - Kiểm tra kỹ các mục kinh nghiệm viết bằng tiếng Việt ("xây dựng framework", "viết kịch bản tự động hóa") để ghi nhận đúng độ sâu thực tế của ứng viên.

---

## 3. Quy trình thực hiện

1. Nhấp đúp chuột mở file `docs/labeling/index.html` trên trình duyệt (Chrome, Safari, Edge...).
2. Nhập tên của bạn vào ô `Your name` ở góc trên cùng.
3. Duyệt lần lượt 120 cặp:
   - Đọc yêu cầu cốt lõi của JD ở cột trái.
   - Đối chiếu bằng chứng thực tế trong CV ở cột phải.
   - Nhấn phím số `0`, `1`, `2`, hoặc `3` để gán điểm.
   - Có thể thêm ghi chú ngắn ở ô `note` nếu gặp cặp phân vân (ví dụ: *"thiếu Playwright, chỉ có Cypress"*).
4. Khi hoàn thành đủ 120 cặp, nhấn nút **"Download labels"** (file tải về dạng `human_<tên_bạn>.json`).
5. Copy file vừa tải vào thư mục `data/eval/` và chạy lệnh tính điểm:
   ```bash
   python -m app.eval.human score
   ```
   Hệ thống sẽ tự động cập nhật kết quả vào `docs/HUMAN_LABELS.md`.
