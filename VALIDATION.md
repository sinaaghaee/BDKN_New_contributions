# Validation — regression comparison

## Adaptive extension — 2026-11-18

- Added independent functional, weight, and input-gradient constrained-diversity scripts; existing fixed-weight model files and the source repository are unchanged.
- 33 tests pass across adaptive mathematics, reference sampling, existing method mathematics, dataset contracts, fixed-result matching, and report handling.
- Explicit pairwise calculations and finite differences verify full similarity gradients. Tests cover multiplier direction/nonnegativity, frozen initialization quantities, independent reference RNG, model-precision exclusions, and independence from test targets.
- All 14 two-epoch GPU 0 smoke fits completed with three Bayesian steps, covering all methods/locations and both learning rates. Both prediction stages have finite metrics and valid saved predictions.
- Re-running the smoke matrix starts no trainers. Modified saved artifacts are rejected by hash validation. A separate SIGTERM check confirms that the runner stops its own child and preserves the attempted output for inspection.
- Headless Chrome checks pass for direct `file://` operation across all ten datasets, both settings, reference locations, DE/BDKN, all five fixed baseline weights, observation modes, and loss scales. No network assets are requested; equations render locally, downloads are local, and selections survive reloads.
- The full 140-fit suite is launched on GPU 0 with automatic report updates. Following the user's concurrency change, `WORKERS='all'` admits as many fits as memory permits, reserving 450 MiB per fit and 2 GiB of headroom; capacity is checked every five seconds. The clean controller restart preserved 67 completed fits. A mocked-memory test verifies admission accounts for starting contexts and excludes other jobs' allocations. The live status and completed metrics are in `results/adaptive/manifest.json` and `website/dist/adaptive.html`; this section does not claim the full suite has finished.
- Constraint violations remain visible, including short high-rate runs that do not satisfy the target. No automatic target relaxation or convergence/calibration guarantee is introduced.

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

Experiment selection is configured in `website/experiments.json`. The offline report is `website/dist/index.html`. Publication status is recorded separately below; rebuilding locally alone does not update the private hosted copy.

Environment: Python 3.13; torch 2.11.0+cu126; NumPy 2.4.3; SciPy 1.17.1; matplotlib 3.10.8; scikit-learn 1.8.0; GPyTorch 1.15.2; VBLL 0.4.9. Source-specific settings and unequal budgets make this an illustrative comparison, not an equal-compute or repeated-seed benchmark.

Earlier five-profile verification: all 23 numerical/source/UI-unit checks passed. Executable report-JavaScript checks passed all observation modes, all loss-axis combinations, each real strength switch, nested-to-default navigation preserving the page, and synthetic future dataset/LR combinations. Every profile has 36 colored main-table cells and 90 colored strength-comparison cells, with no delta rows or summary cards. There are 16 individual loss charts plus 15 side-by-side repulsion-loss charts; both axes default to log. All five strengths completed with finite scores and positive measured training costs.

Loss definitions are grouped by all nine methods with local KaTeX rendering and short symbol tables. All 52 LaTeX expressions rendered successfully. Button interactions passed every axis, observation, strength, and future dataset/LR navigation check. Full browser-layout rechecking remains unavailable as noted above.

## Multi-dataset extension

The suite adds nine source-derived scalar tasks to the existing step dataset: cubic, Forrester, He, amplitude-varying sine, linear+sinc, Fourier square wave, shifted sinc, sine, and cosine. Source training counts/patterns and active noise settings are retained. New observations use seed 123 and independent streams; the sine/cosine tasks share input coordinates but use independent noise. Each has 1,000 held-out observations uniformly sampled across the full plotting domain and a 2,000-point noiseless function grid. Shifted sinc is noiseless.

All 81 two-epoch smoke executions passed on the two RTX 3090 GPUs. The numerical/source-parity and suite tests now contain 30 passing checks. Full runs retain 2,000 epochs, 10,000 for DKL/LDBLL, and 100 Bayesian aggregation steps. Non-repulsive fits are reused across five weight profiles, with original training costs preserved. All 45 added profiles completed successfully (189 successful standalone executions, including reused-DE evaluations). After the editor interruption, three incomplete cosine fits restarted from method boundaries; all other completed fits were preserved.

The new Results page has separate RMSE, NLL, and training-time tables with dataset rows and twelve method columns. Colors rank methods within each row. The other pages retain dataset buttons, observation controls, logarithmic loss controls, side-by-side weight panels, and method-wise LaTeX definitions. Full browser-layout verification remains unavailable; executable JavaScript and saved-artifact checks cover the controls, links, values, and formulas.

Final multi-dataset checks passed: 450 registered method artifacts with matching budgets/data/history/checkpoints; recomputed RMSE and NLL match every saved prediction; 400 HTML pages, 50 profiles, 600 metric rows, 6,220 unique plot assets, and 67 distinct LaTeX expressions. Executable JavaScript checks passed observation/axis/method controls, every real dataset/weight switch, and overview learning-rate selection across differing dataset availability. Publication was blocked by automatic approval review before the official Sites workflow started: explicit user authorization to send the report and derived project artifacts to the external hosting destination is required. The existing hosted site remains unchanged; the complete verified report is website/dist/index.html. No credential was sent to a terminal, and no source push or deployment occurred.

## High learning rate / 40-epoch setting

All 210 requested standalone executions completed successfully on the two RTX 3090 GPUs: ten datasets, five repulsion strengths, all nine methods, with common baselines reused across weights. Every method, including GP, DKL, and LDBLL, used learning rate 0.1 and 40 epochs. BDKN aggregation retained 100 steps at learning rate 0.1. Existing low-setting fits and shared data artifacts were preserved.

Verification passed for 900 registered method artifacts across both settings, including settings/budgets, complete histories/checkpoints, baseline reuse, and independently recomputed held-out RMSE/NLL. All 32 numerical/source/suite tests passed. The combined offline report has 100 profiles, 1,200 metric rows, 800 checked HTML pages, 12,430 unique plot assets, and 67 distinct rendered LaTeX expressions. Real low/high setting switches passed on Results, Predictions, and Losses, alongside dataset/weight/method/observation/axis controls. Concurrent status writers now use unique temporary files and atomic replacement.

The complete high-setting report is `website/dist/experiments/steps-lr0p1-r0p001/index.html`. Publication remains pending explicit authorization following the earlier automatic approval rejection; no new upload or deployment was attempted for this extension.

## Aggregation-weight diagnostics

Diagnostics were generated from 320 unique saved fits and reused across 400 BDKN-variant/profile combinations. Saved coefficient posteriors were read directly; CPU neural-function reconstructions were checked against saved test/grid predictive means (recorded reconstruction errors are downloadable). Exact Student-t marginal scales, standard deviations, 95% intervals, sign probabilities, and coefficient correlations passed closed-form and spectral/matrix equivalence checks. Negative coefficients remain unconstrained. The Gaussian prior reference is explicitly conditional on fitted posterior mean noise, not an unconditional pre-data marginal. Constant-function correlations are marked undefined.

All 34 numerical/source/suite/aggregation tests passed. The offline site now has 900 verified HTML pages, 1,200 unchanged benchmark metric rows, and 68 rendered LaTeX expressions. Dataset/LR/weight navigation, aggregation-variant filters, observation switches, and existing axis controls passed executable JavaScript checks. No method was retrained for this addition. Full browser-layout rechecking remains unavailable as previously recorded.

## Repulsion-location experiments

All 400 new standalone GPU fits completed on the two RTX 3090s: FRDE and FoRDE, random-only and training-plus-random repulsion, ten datasets, both existing learning-rate/budget settings, and all five strengths. Each fit produced its matched DE and BDKN results. The original 100 training-only profiles were preserved. Weight repulsion and the other baselines were reused with their original measured costs.

The random sampler uses an independent CPU generator, redraws each optimizer step, and excludes all training inputs and within-step duplicates in model precision. Tests cover deterministic matched streams, collision rejection, narrow-domain support, preprocessing, gap/extrapolation coverage, unchanged training-only objectives, and nonzero finite repulsion gradients. Training/random use B points and combined uses 2B; this deliberate count difference is documented in Settings and comparison pages. No prior or loss-formula changes were introduced.

All 44 numerical/source/suite/aggregation/location tests passed. Fit validation checked 2,700 registered method references, budgets, shared datasets, histories, checkpoints, baseline costs, and independently recomputed RMSE/NLL. Offline checks passed 3,000 HTML pages, 300 profiles, 3,600 metric rows, and 68 rendered LaTeX expressions. Executable controls passed all loss-axis/observation combinations, location navigation, location preservation across dataset/LR/weight changes, method filters, and aggregation controls. Aggregation diagnostics cover 720 unique saved fits and 1,200 profile/variant references. The site contains 100 location-comparison groups and 27,830 unique shared plot assets. No correlation plots remain; mean-weighted contribution panels retain the “Sum” legend. Every bundled file is below GitHub’s single-file limit.

The full offline report remains `website/dist/index.html` (also accessible through the repository-root `index.html`). This extension does not change the separately hosted private Site; browser-layout inspection remains unavailable as previously recorded.

## FoRDE raw-gradient ablation

Added `forde_raw.py` alongside the unchanged normalized implementation. The original fixed-weight KDE uses raw input-gradient fields; stop-gradient references, bandwidth calculation, self terms, sampling and Bayesian aggregation are retained. All 300 selected configurations match their saved normalized reference settings (ten datasets, both LR budgets, five weights and three input locations). Each fit produces DE and BDKN predictions. Dataset and source hashes, full budgets, finite artifacts and completion markers are checked before reuse.

Six raw-gradient tests passed, including explicit float64 loss/backward equivalence, unused normalization epsilon, zero-strength likelihood equivalence, sampling parity, all matched configurations and simultaneous allocation-failure retries. The preceding broader regression run passed 39 tests. Six CPU smoke fits and six full 40-epoch GPU-0 validation fits completed successfully; verified GPU fits were reused in the production selection. The 300-fit production run is in progress, with memory-aware concurrent admission on GPU 0 only; its manifest records live completion status.

The dedicated offline page compares original, normalized FoRDE and raw FoRDE, showing DE and BDKN together in every metric table. Real-browser checks passed dataset/LR/weight/location/observation/stage controls, signed logarithmic loss axes, local downloads, rendered equations and reload persistence, with no external asset requests or JavaScript errors. The report refreshes as production results complete.
