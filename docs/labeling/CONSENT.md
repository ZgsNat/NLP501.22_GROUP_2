# Collecting consented real CVs

**Goal.** 5–10 CVs of real automation testers, used with written consent, to check that the ranking holds on real
text. Both benchmarks are synthetic, and this is the only real test-automation data we can use cleanly: public
datasets either have no clear license or contain no QA CVs (docs/REALTEXT_RESULTS.md).

**Who.** Any teammate who knows automation testers (colleagues, classmates, friends) and can ask them directly.

## Rules

1. **Written consent first.** No CV is copied before the person has agreed to the text below, by message or e-mail.
   Keep that message.
2. **Raw files never enter git.** They go only into `.runtime/consented/raw/`, which is git-ignored, on one machine.
3. **Only redacted text is used.** `intake` writes the redacted copy. A person reads it to confirm no name, contact,
   photo, birth date, address or employer-identifying detail survives, then **deletes the raw file**.
4. **Only aggregate numbers are published.** Nothing that could identify a person goes into the report, slides or repo.
5. **Withdrawal.** A person can withdraw at any time; delete both their files and rerun the evaluation.
6. **Deletion.** Delete everything in `.runtime/consented/` when the course ends.

## Consent text to send (English)

> We are a group of master's students (NLP501, FSB) evaluating a research prototype that ranks CVs for automation
> testing roles. May we use a copy of your CV for this evaluation only? Your name, contact details, birth date,
> address and photo are removed by software and checked by a person before anyone reads it. The anonymised text
> stays on one student's computer, is never published or shared, is used only to compute aggregate accuracy
> numbers, and is deleted when the course ends. You can withdraw at any time by telling us. Reply "I agree" if
> that is OK.

## Consent text to send (Tiếng Việt)

> Chúng tôi là nhóm học viên cao học (môn NLP501, FSB) đang đánh giá một hệ thống nghiên cứu xếp hạng CV cho vị trí
> kiểm thử tự động. Bạn có đồng ý cho chúng tôi dùng một bản CV của bạn chỉ cho việc đánh giá này không? Họ tên,
> thông tin liên lạc, ngày sinh, địa chỉ và ảnh sẽ được phần mềm xoá và một người kiểm tra lại trước khi đọc. Bản đã
> ẩn danh chỉ lưu trên máy của một thành viên, không công bố, không chia sẻ, chỉ dùng để tính các chỉ số tổng hợp, và
> sẽ bị xoá khi kết thúc môn học. Bạn có thể rút lại sự đồng ý bất cứ lúc nào. Nếu đồng ý, vui lòng trả lời
> "Tôi đồng ý".

## Steps (Thịnh's machine)

```bash
mkdir -p .runtime/consented/raw            # put the consented CVs here (.pdf / .docx / .txt / .md)
cd backend && uv run python -m app.eval.realtext intake
# → .runtime/consented/redacted/<file>.txt, plus a line per CV: what was masked, what still looks like PII
```

Then:
1. Read every redacted file and fix or remove anything identifying by hand.
2. Delete the raw files.
3. Grade each consented CV against the 12 JDs (0–3, the same guideline as the labelling page), two people
   independently.

With 5–10 CVs, the result is a sanity check: do strong real testers rank above weak ones? It is not a benchmark, and
the report says so.
