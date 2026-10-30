# Human labelling: validating the silver labels

**Who:** Hoa and one more teammate, each working alone. **Time:** about 2 hours each (120 pairs, roughly 1 minute per pair).

Every ranking number in this project is scored against *silver* labels written by two Claude models (docs/EVAL.md §2).
Models from one family make correlated mistakes, so their agreement with each other (κ 0.919) overstates their
reliability. Your labels tell us how far to trust every result in the report. This is the single most important input
to the final evaluation.

## Steps

1. Open `docs/labeling/index.html` in a browser (double-click it; no server or internet needed).
2. Type your name in the top bar and read the guideline (click "Guideline").
3. For each pair, read the JD on the left and the CV on the right, then grade 0–3 (buttons, or keys `0`–`3`).
   Add a note when you hesitate. Your progress is saved in this browser, so you can stop and come back.
4. When all 120 are done, click **Download labels**. You get `human_<name>.json`.
5. Put that file in `data/eval/` and commit it, or send it to Thịnh.
6. Thịnh runs `cd backend && uv run python -m app.eval.human score`, which writes `docs/HUMAN_LABELS.md`.

## Rules

- **Work alone.** Don't discuss pairs with the other labeller until you have both downloaded your files; independent
  labels are what make κ meaningful.
- **Judge the evidence, not the keywords.** A tool in a skills list is a claim; a project that used it is evidence.
  "Familiar with" or "currently learning" is not the skill.
- **Ignore anything not job-related.** The CVs are synthetic and anonymised.
- Grades: 0 = a core must-have is absent (or only claimed or hedged); 1 = weak or adjacent; 2 = would phone-screen;
  3 = strong fit with demonstrated work, would interview.

## What the sample is

The page holds 10 pairs per JD across all 12 JDs. In each JD, 3 pairs are ones where the two silver annotators
disagreed, and the rest spread across every silver grade (seed 13, `app/eval/human.py › sample`). You never see the silver
grades or any system's ranking. Because disagreements are over-represented, κ on this sample is a conservative
(lower) estimate of agreement over all 600 pairs.
