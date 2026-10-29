# MedLift-3D: FYDP-III final report (LaTeX)

UIU FYDP template layout, six chapters, all diagrams in TikZ (`figures/*.tex`).

## Build
* Overleaf: upload this folder (or `medlift3d_report.zip`), set the compiler to pdfLaTeX, main file `main.tex`.
* Local: `latexmk -pdf main.tex`

## What you still have to fill
Every placeholder prints as red **[FILL: ...]** in the PDF. Search the sources for `\fillme`.
When all are filled, set `\showfillfalse` in `main.tex` to hide any leftovers.

| Where | What |
|---|---|
| Abstract, 4.3.4-4.3.7, 4.4, 6.1 | numbers from the Kaggle tables: `outputs/10_evaluation/table_accuracy.md`, `table_intervals.md`, `outputs/16_uncertainty/table_uq_*.md`, `outputs/11_growth/table_growth.md`, `outputs/17_explain/table_explain.md`, `table_error_drivers.md` |
| 4.2 issues (4), (5) | resolution of the step-10 interval bound; the three failed checks in `report.html` |
| Fig. 3.7, Table 3.7 | your real week numbers and task allocation |
| 5.2.3, 5.2.8, 5.3 | AI-assistance sentence; current Bangladesh DGDA / data-protection rules; local RTX 4090 price |
| `references.bib` | entries marked `VERIFY` |

## Result figures
Copy these files from the Kaggle `outputs/` folder into `figures/results/` with the name in the right column
(the report shows a red placeholder box until the file is there):

| Kaggle file | name in `figures/results/` |
|---|---|
| `outputs/15_fit/fit_curves.png` | `fit_curves.png` |
| `outputs/14_detector/froc_test.png` | `froc_test.png` |
| `outputs/10_evaluation/volume_errors.png` | `volume_errors.png` |
| `outputs/10_evaluation/bland_altman_volume_<best>.png` | `bland_altman_volume.png` |
| `outputs/16_uncertainty/volume_uq_overview.png` | `volume_uq_overview.png` |
| `outputs/16_uncertainty/reliability.png` | `reliability.png` |
| `outputs/11_growth/doubling_time_<best>.png` | `doubling_time.png` |
| `outputs/17_explain/<case>_n<j>_seen_unseen.png` (pick one) | `seen_unseen_example.png` |
| `outputs/17_explain/saliency_examples.png` | `saliency_examples.png` |
| demo screenshot (LIDC-IDRI-0018 report) | `demo_report.png` |
