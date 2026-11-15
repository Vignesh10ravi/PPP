# Final PPP report

The source follows the official PPP guideline and the common structure of the
three professor-approved reference reports supplied by the author.

## Build

From this directory:

```bash
latexmk -pdf main.tex
```

or:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Overleaf

Upload the complete `final_report_overleaf_updated_20261115.zip` project
archive, not only `main.tex`. The older `final_report_overleaf.zip` predates
the latest shear-holdout and macro-FE report updates. The updated archive keeps
`main.tex`, `metadata.tex`, `references.bib`,
and the `frontmatter/`, `chapters/`, and `appendices/` directories together at
the project root, together with the referenced vector figures. In Overleaf,
select `main.tex` as the main document if it is not selected automatically.
The reported `frontmatter/titlepage.tex` error means the project upload did
not include the complete folder tree (the file is present in this source and
archive). Replace the incomplete upload with the complete archive; do not
upload `main.tex` by itself.

## Draft status

The current source provides:

- title/front matter and centralized metadata;
- a concise CMS-framed abstract and introduction;
- detailed theory, numerical methods, and software provenance;
- an explicit two-stage proof-of-concept and multiscale workflow;
- an honest AI/tool-usage declaration;
- completed Phase 1 and RVE verification results;
- verification plots for tangent differentiation, Newton convergence, and
  matched FE error, plus pilot RVE loading coverage and residuals;
- a plotting script and the pilot data/manifest used by its RVE figure;
- manual and reproducibility sections; and
- evidence-backed surrogate metrics, independent shear evaluation of the
  selected shear-adapted checkpoint, a matched macro-FE integration result,
  and previously frozen artifact hashes.

Remaining placeholders include the final shear dataset, manifest, report, and
macro-FE comparison hashes, unavailable Git history, official examiner names,
and the submission date. Per-function and direct-RVE versus surrogate timings
are marked as unrecorded rather than estimated.

Do not remove a remaining placeholder by guessing. Replace it only from a
frozen result or official metadata.

The current 41-page preview is saved in the `Final Report prep` directory as
`final_report_draft_20261115.pdf`. It is a draft until the remaining artifact
hashes, Git history, examiner names, and submission date are filled.
