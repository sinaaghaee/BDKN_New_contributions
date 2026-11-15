# 1D regression experiments

Standalone implementations from `BDKN_Experiments`, evaluated on ten scalar 1D datasets. The existing smoothed-step dataset comes from [arXiv:2505.18526 §4.1](https://arxiv.org/html/2505.18526v1#S4.SS1). The paper supplies only the step-data definition; its DBK method is not included. Other data generators come from the local BDKN/BLLN toy scripts, with source references recorded in metadata.

[Existing private comparison website](https://bdkn-toy-regression-comparison.earlyyak24.chatgpt.site); publication of the new multi-dataset report awaits explicit authorization. Open `website/dist/index.html` for the offline report.

## Run

```bash
python -m pip install -r requirements.txt
python run_all.py --device cuda:0 --output results/my_run
```

On this machine, use `../BDKN_Experiments/.venv/bin/python` in place of `python`.

Each method has its own self-contained script and an editable configuration at the beginning:

| Script | Method | Default epochs |
| --- | --- | --- |
| `bdkn.py` | Original vectorized BDKN | 2,000 + 100 Bayesian steps |
| `deep_ensemble.py` | Original BDKN first stage | 2,000 |
| `frde.py` | Functional repulsion + BDKN | 2,000 + 100 Bayesian steps |
| `wrde.py` | Weight repulsion + BDKN | 2,000 + 100 Bayesian steps |
| `forde.py` | Input-gradient repulsion + BDKN | 2,000 + 100 Bayesian steps |
| `gp.py` | Source linear + RBF GP | 2,000 |
| `dkl.py` | Source Wilson-style RBF DKL | 10,000 |
| `ldbll.py` | Source latent-derivative Bayesian last layer | 10,000 |
| `vbll_regression.py` | Source VBLL | 2,000 |

Neural models use learning rate 0.001 and two 50-unit hidden layers. LDBLL retains its source activation, normalization, priors, and derivative regularization. GP retains learning rate 0.01. Repulsion defaults to 0.001. All method-specific settings remain editable. The scripts import installed libraries and load the shared data; they do not import one another or the old repository.

```bash
python generate_data.py
python ldbll.py --device cuda:0
python run_all.py --epochs 2 --device cpu  # Smoke test
python run_all.py --methods frde wrde forde --repulsion-strength 0.01 --device cuda:0 --output results/strength_001
python run_all.py --nn-learning-rate 0.01 --device cuda:0 --output results/nn_lr001
```

`--nn-learning-rate` changes neural learning rates and leaves GP's configured rate intact. Standalone scripts accept `--learning-rate`. Neither override changes the separate Bayesian-stage learning rate. The runner reuses BDKN's ensemble checkpoint for DE only when data and first-stage settings match. Direct method execution trains independently unless DE is explicitly given `--reuse-stage1`.

## Data and outputs

Defaults: 100 training points, 1,000 independent noisy test points, 2,000 plotting-grid points, seed 123. Inputs are uniform on `[-1,1]`. The paper's smoothed steps and small oscillation are observed with Gaussian noise of variance **0.01**, standard deviation **0.1**.

Edit `generate_data.py`, then use `--force` to intentionally regenerate existing data. A separate dataset can be saved with `--output data/another.npz`. Every method reads the same artifact. Fit preprocessing uses training data only. RMSE/NLL use the noisy held-out observations in original units. Ensemble NLL uses the original moment-matched Gaussian; BDKN stages use Student-t predictive densities. Plot intervals include observation noise.

Outputs contain resolved configurations, histories, checkpoints, prediction arrays, metrics, and logs. New output directories prevent accidental overwrites. CUDA-synchronized `training_seconds` measures fitting separately from prediction and plotting. BDKN totals include both fitting stages; DE reuse reports the original ensemble fitting cost, not near-zero loading time. Epoch budgets differ, so timing is a comparison of the configured runs rather than equal-compute performance.

The latest base run is `results/timed_baseline`. Additional strength runs are `results/timed_repulsion_001` (0.01), `results/timed_repulsion_01` (0.1), `results/timed_repulsion_1`, and `results/timed_repulsion_10`. These reuse the non-repulsive fits and replace the three repulsive fits, with source runs recorded in `run.json`.

## Website

```bash
python build_website.py
```

`website/experiments.json` registers completed experiments. Add a dataset label/formula and a run entry with its dataset ID, neural learning rate, and repulsion strength, then rebuild. The website automatically offers only valid combinations; no placeholder experiments are displayed. Runs grouped under a dataset ID must use identical data. `--registry` selects another registry and `--run` builds just the default profile with an explicit run.

The site opens on separate RMSE, NLL, and training-time tables: dataset rows, method columns, with green-to-red ranks within each row. Separate pages show predictions, DE/BDKN pairs, weight comparisons, losses, data, and settings. There are no summary statistic cards or delta rows. All prediction plots support training/test/both/no observations. Loss X/Y scales are independent; nonpositive losses use signed log. Repulsion weights can be selected globally or compared side by side with tables.

For a new dataset, produce the same NPZ array interface: `x_train`, `y_train`, `x_test`, `y_test`, `x_grid`, and `f_grid`, each shaped `(N,1)`. Optional `metadata` is a JSON string. The report formula comes from the registry, so other datasets are not mislabeled with the paper's function.

## Validation and provenance

```bash
python -m unittest -v test_experiment test_suite
BDKN_SOURCE_REPO=../BDKN_Experiments python -m unittest -v test_experiment test_suite
```

`provenance.json` records source hashes and adaptations. Tests cover data generation, preprocessing, model math/gradients against the source, checkpoint reuse, both-stage loss plots, log axes, heatmap colors, registry validation, and profile navigation. See `VALIDATION.md` for completed runs and verification limits. FoRDE's BSD license is retained under `licenses/`. Data/results/site assets are generated and ignored by Git.

The repulsion sweep includes 0.001, 0.01, 0.1, 1, and 10. The Losses page has a dedicated unweighted repulsion-loss R plot for each repulsive method. Compare weights shows a separate prediction and R-loss panel for each strength. Both loss axes default to log; signed log preserves zero/nonpositive values.

## All ten datasets on two GPUs

```bash
python run_suite.py --register
python build_website.py
```

The default suite adds Cubic, Forrester, He, amplitude-varying sine, linear+sinc, Fourier square wave, shifted sinc, sine, and cosine. Existing completed smoothed-step profiles remain registered and unchanged. `run_suite.py` starts with editable dataset/weight/device settings; CLI overrides include `--datasets`, `--weights`, `--devices`, `--learning-rate`, and `--output`. Use `--epochs 2 --weights 0.001 --output results/suite_smoke` for a short check; omit `--register` for smoke runs.

Each added dataset preserves its source training count, sampling pattern, and active noise setting. All use seed 123, 1,000 independent uniformly sampled test observations over the full plotting domain, and a noiseless 2,000-point plotting grid. Source seeds are standardized for this comparison; source realizations are not claimed to match. Cubic uses the active script noise standard deviation 6.0. Sine and cosine share X but have independent noise. Shifted sinc is noiseless. The optimization/active-learning examples use fixed initial samples and do not acquire more data.

The suite runs one training subprocess per GPU and skips only completed fits whose saved settings, data hash, history, and predictions validate. Incomplete attempts are retained under `attempts/`; failed jobs are reported in `suite_status.json`. Re-run the same command to resume. Non-repulsive fits are trained once per dataset; additional weight profiles reuse their original fitting times. Model settings stay unchanged: 2,000 epochs, DKL/LDBLL 10,000, and BDKN stage two 100 steps.

On the Results page, learning-rate and weight buttons select the cross-dataset tables. On other pages, dataset buttons select figures, data formulas, losses, and settings. Metric cells link to the matching prediction. The combined results download contains every registered dataset/weight/method row.

For a presentation-only website update, use `python build_website.py --ui-only`. It reuses existing figures, checks dataset identity and available fit/history fingerprints, then refreshes all tables and controls. `python verify_website.py --fixture /tmp/report-fixture.json` checks generated pages against saved metrics; `node verify_report_controls.mjs /tmp/report-fixture.json` exercises buttons and LaTeX rendering.

## High learning rate, short budget

```bash
python run_suite.py --preset high --register
python build_website.py
```

The high preset runs all ten datasets and all five repulsion weights with **learning rate 0.1 and 40 epochs for every method, including GP, DKL, and LDBLL**. BDKN Bayesian aggregation retains 100 steps at learning rate 0.1. Results go to `results/suite_high`; data artifacts and the original low-rate results are reused without overwriting fitted models. The report’s Training setting buttons switch between low/long and high/40-epoch comparisons.

Aggregation weights now has posterior means and marginal 95% Student-t intervals, exact coefficient densities, coefficient/function correlations, individual-network test scores, and weighted function contributions. All four BDKN variants are available across both training settings, all datasets, and all repulsion weights. These diagnostics read saved posteriors and checkpoints; they do not retrain models. Conditional prior density references use the fitted posterior mean noise variance.
