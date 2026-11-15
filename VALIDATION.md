# Validation — regression comparison

Completed 2026-11-15 with NVIDIA GeForce RTX 3090 GPUs.

## Completed fits

- Common baseline: `results/timed_baseline`.
- DKL and LDBLL: 10,000 epochs. Other methods: 2,000 epochs. Bayesian aggregation: 100 additional optimization steps.
- LDBLL uses learning rate 0.001 and hidden widths [50, 50]. Existing source priors, derivative regularization, activation, and preprocessing are retained.
- Functional, weight, and first-order repulsion tested at 0.001, 0.01, 0.1, 1, and 10. Extra profiles reuse identical non-repulsive fits; `run.json` records their origins.
- Every fit used the same 100 training / 1,000 noisy test observations and dataset hash. Every completed run has an empty `failures.json`.
- Training times are measured separately from prediction and plotting, with CUDA synchronization. Reused DE retains the original BDKN stage-one training cost. BDKN totals include both fitting stages.

All requested strengths finished with finite metrics. Strong functional repulsion can severely degrade the unaggregated DE: strength 10 has RMSE about 127.58. These values remain visible. The weight-comparison page uses separate full-range panels for each strength; no overlaid curves or hidden clipping. Individual prediction figures use their own full range; the all-model grid shares axes.

## Automated checks

The numerical/source-parity suite checks data generation/noise, training-only normalization, source hashes, original ensemble outputs, repulsion objectives/gradients, zero-strength behavior, BDKN posterior/scoring, GP/LDBLL training, DKL/VBLL gradients, checkpoint reuse, both training stages, dedicated unweighted repulsion-loss curves, independent logarithmic axes, default log selections, heatmap colors, and experiment registry/navigation.

All five completed strength profiles are checked against saved configurations and scores. LDBLL/DKL epoch counts, nonzero fitting times, and the reused-DE cost are checked directly against artifacts. Static report checks cover every page, download, pre-rendered axis/observation variant, color-coded metric cell, and the absence of summary cards or delta rows. Executable JavaScript checks exercise loss scales, observation modes, and valid experiment switching.

Initial desktop/mobile Chrome checks passed for the earlier report. Later browser rechecks could not complete: the renderer stalled, and a single-process retry crashed. Current full browser-layout verification remains incomplete; source tests, static assets, and executable control checks are used instead.

## Reproduce

```bash
BDKN_SOURCE_REPO=../BDKN_Experiments python -m unittest -v test_experiment
python build_website.py
```

Experiment selection is configured in `website/experiments.json`. The offline report is `website/dist/index.html`. The private hosted copy uses the same generated static source.

Environment: Python 3.13; torch 2.11.0+cu126; NumPy 2.4.3; SciPy 1.17.1; matplotlib 3.10.8; scikit-learn 1.8.0; GPyTorch 1.15.2; VBLL 0.4.9. Source-specific settings and unequal budgets make this an illustrative comparison, not an equal-compute or repeated-seed benchmark.

Latest verification: all 23 numerical/source/UI-unit checks passed. Executable report-JavaScript checks passed all observation modes, all loss-axis combinations, each real strength switch, nested-to-default navigation preserving the page, and synthetic future dataset/LR combinations. Every profile has 36 colored main-table cells and 90 colored strength-comparison cells, with no delta rows or summary cards. There are 16 individual loss charts plus 15 side-by-side repulsion-loss charts; both axes default to log. All five strengths completed with finite scores and positive measured training costs.

Loss definitions are grouped by all nine methods with local KaTeX rendering and short symbol tables. All 52 LaTeX expressions rendered successfully. Button interactions passed every axis, observation, strength, and future dataset/LR navigation check. Full browser-layout rechecking remains unavailable as noted above.
