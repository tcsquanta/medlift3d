# MedLift-3D: final report

UIU FYDP template layout, six chapters, all diagrams in TikZ (`figures/*.tex`), result figures in `figures/results/`.

| File | What it is |
|---|---|
| `MedLift3D_Report.pdf` | compiled report |
| `MedLift3D_Report.docx` | Word version (same content, figures, numbering and references) |
| `main.tex`, `chapters/`, `frontmatter/`, `figures/`, `references.bib` | LaTeX source |

## Build
* PDF on Overleaf: upload this folder (or `medlift3d_report.zip`), compiler pdfLaTeX, main file `main.tex`.
* PDF locally: `latexmk -pdf main.tex`
* Word: after building the PDF, `python tools/make_docx.py` (needs pandoc and python-docx). TikZ diagrams are
  rendered to 300 dpi images; figure, table, equation, section and citation numbers are taken from the LaTeX build.
  When Word opens the file it asks to update fields: answer Yes to fill the table of contents.
