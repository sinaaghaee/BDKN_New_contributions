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

For each run, the report shows the actual method settings and epoch budgets. The Results page has separate RMSE, NLL, and training-time tables, with datasets as rows and methods as columns. Colors rank methods from green to red within each dataset row; it has no summary cards or delta rows. Compare weights places all registered strengths side by side at the selected dataset/LR. Prediction observation modes and independent loss-axis scales work offline.

Registered repulsion weights: 0.001, 0.01, 0.1, 1, and 10. Dedicated R-only plots appear on Losses and Compare weights. Loss axes default to log/log; nonpositive values are preserved with signed log. Each prediction panel shows its own full Y range, including the extreme high-strength functional-DE fits.

Loss definitions are organized method by method, with LaTeX equations and concise symbol definitions. All selectors use visible button groups; no dropdowns or collapsed details. KaTeX is bundled locally under `vendor/katex` with its license.

`run_suite.py --register` adds completed dataset/weight profiles. Cross-dataset tables retain learning-rate and strength controls; the other pages use dataset buttons. Plot bounds come from each dataset grid. The combined CSV download includes all registered profiles.

Training setting buttons now compare the existing low/long setting with High LR 0.1 / 40 epochs. High applies to every method, including GP; Bayesian aggregation remains 100 steps. Each setting covers all ten datasets and five weights.

Aggregation weights reads the saved training posterior, showing Student-t marginals, intervals, correlations, and functions/contributions for each BDKN variant. Observation switches affect function panels, while the same training-derived coefficient posterior is retained. Coefficients are unconstrained; contribution shares are descriptive summaries rather than probabilities.
