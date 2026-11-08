# Noisy coding CVs (E33)

Synthetic CVs only: the `coding` branch's generator wrote them from sampled facts (fake names, fake contacts); no real
person is in this folder. Each condition applies one kind of noise at one rate to those CVs with a fixed seed, so the
same command always produces the same files.

- `latent.json`: the gold facts per CV (tools, years, depth, ISTQB, languages, name, gender), sampled by the generator
  **before** the CV text was written. Noise never changes it.
- `gold.json`, `jds.yaml`: the generator's ranking labels and the 6 JDs (for nDCG).
- `<kind>_<level>/cvs/*.txt`: the noisy CV text (the 193 clean CVs; `accents_*` only the 72 Vietnamese ones;
  `original_0` = all 200 as rendered, including the generator's 7 noisy ones).
- `<kind>_<level>/pdf/*.pdf`: the same text as a text-layer A4 PDF (fpdf2, Arial 10.5 pt), what the app's upload path reads.
- `<kind>_<level>/pdf_pool/*.pdf`: all 200 CVs as PDFs for the ranking runs (noise applied to the CVs it covers).
- `manifest.json`: SHA-256 of every file.

Noise kinds: `char` (characters replaced by `#@%~`), `ocr` (simulated OCR confusions: I→l, l→1, O→0, rn→m, dropped
diacritics), `accents` (Vietnamese typed without diacritics), `layout` (line breaks lost), `truncate` (end of the CV cut).

| condition | txt | pdf | pdf_pool |
|---|---|---|---|
| `accents_0.5` | 72 | 72 | 0 |
| `accents_1` | 72 | 72 | 0 |
| `char_0.02` | 193 | 193 | 0 |
| `char_0.05` | 193 | 193 | 200 |
| `char_0.1` | 193 | 193 | 0 |
| `char_0.25` | 193 | 193 | 0 |
| `layout_0.25` | 193 | 193 | 0 |
| `layout_0.5` | 193 | 193 | 0 |
| `layout_1` | 193 | 193 | 0 |
| `ocr_0.05` | 193 | 193 | 0 |
| `ocr_0.15` | 193 | 193 | 0 |
| `ocr_0.3` | 193 | 193 | 0 |
| `original_0` | 200 | 200 | 200 |
| `truncate_0.25` | 193 | 193 | 0 |
| `truncate_0.5` | 193 | 193 | 0 |

Regenerate or score (results in `docs/NOISE_ROBUSTNESS_RESULTS.md`):

```bash
cd backend && uv run python -m app.eval.noise_robustness            # full run: extraction (text and PDF) + ranking
cd backend && uv run python -m app.eval.noise_robustness --sample 60 --no-rank   # quick
```
