# Classical approach documents

Two documents describe the same classical baselines for two audiences.

## Technical report

`classical_approach.tex` is the editable LaTeX source;
`classical_approach.pdf` is the compiled report. It describes Models A–D,
preprocessing, noise mixing, validation, metrics, CUDA/CPU execution, and the
requirements for a matched quantum comparison. It includes citations for DDL,
ESC-50, UrbanSound8K, NASA Small UAS, and the Svanström/ITU ARIS datasets referenced
by the teammate protocol. It reports no new training results.

The report identifies the source snapshot it describes. The file fingerprints in
`classical_approach_sources.json` record that snapshot, including uncommitted work.
Concurrent protocol changes may require a later document update.

Compile from this directory with a standard TeX Live installation (or upload the
`.tex` file to Overleaf):

```bash
pdflatex -interaction=nonstopmode -halt-on-error classical_approach.tex
pdflatex -interaction=nonstopmode -halt-on-error classical_approach.tex
```

Two passes resolve references. No external images, bibliography database, model
artifacts, or training runs are required to build either PDF.

## Beginner paper

`classical_approach_beginner.tex` compiles to `classical_approach_beginner.pdf`,
*Can a Computer Hear a Drone?* It covers the same problem and Models A–D for a reader
with no machine-learning background, in research-paper form: problem statement,
research questions, background vocabulary, data design, one section per model
(how it decides, why it is in the comparison, where it may fail), experimental
design, evaluation with a worked example, threats to validity, and a glossary.
Each idea is introduced in an "In everyday language" paragraph before its equation.

Its worked numbers are labelled as instructional; it reports no training results
either. It describes the current workspace, including the matched selected-feature
protocol, and records those file fingerprints in
`classical_approach_beginner_sources.json`. Build it the same way:

```bash
pdflatex -interaction=nonstopmode -halt-on-error classical_approach_beginner.tex
pdflatex -interaction=nonstopmode -halt-on-error classical_approach_beginner.tex
```
