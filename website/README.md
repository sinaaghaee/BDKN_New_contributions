# Comparison report

Open `dist/index.html` offline, or use the published private site.

Rebuild from the repository root:

```bash
python build_website.py
```

Edit `experiments.json` to add a completed experiment. Dataset IDs map to a label and formula. Each experiment specifies:

- A unique lowercase `id`.
- Its `dataset` ID.
- Its nominal neural `learning_rate` (actual per-method rates remain visible in the tables).
- Its `repulsion` strength.
- A completed `run` directory relative to the repository root.

The `default` experiment appears at the site's root. Other profiles have their own directories; switching dataset, learning rate, or repulsion weight preserves the current page. Only registered, completed results are shown. One dataset ID must always refer to identical data, including across learning rates and strengths. Register one run per dataset/LR/strength combination.

For each run, the report shows the actual method settings and epoch budgets. The Results table uses green-to-red colors per metric column; it has no summary cards or delta rows. Compare weights places all registered strengths side by side at the selected dataset/LR. Prediction observation modes and independent loss-axis scales work offline.

Registered repulsion weights: 0.001, 0.01, 0.1, 1, and 10. Dedicated R-only plots appear on Losses and Compare weights. Loss axes default to log/log; nonpositive values are preserved with signed log. Each prediction panel shows its own full Y range, including the extreme high-strength functional-DE fits.

Loss definitions are organized method by method, with LaTeX equations and concise symbol definitions. All selectors use visible button groups; no dropdowns or collapsed details. KaTeX is bundled locally under `vendor/katex` with its license.
