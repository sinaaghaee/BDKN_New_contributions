# 1D regression experiments

Standalone implementations from `BDKN_Experiments`, evaluated on the synthetic dataset from [arXiv:2505.18526 §4.1](https://arxiv.org/html/2505.18526v1#S4.SS1). The paper supplies only the data definition; its DBK method is not included.

[Private comparison website](https://bdkn-toy-regression-comparison.earlyyak24.chatgpt.site). Open `website/dist/index.html` for the offline report.

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

The site opens on the color-coded results table, including training time. Separate pages show predictions, DE/BDKN pairs, weight comparisons, losses, data, and settings. There are no summary statistic cards or delta rows. All prediction plots support training/test/both/no observations. Loss X/Y scales are independent; nonpositive losses use signed log. Repulsion weights can be selected globally or compared side by side with tables.

For a new dataset, produce the same NPZ array interface: `x_train`, `y_train`, `x_test`, `y_test`, `x_grid`, and `f_grid`, each shaped `(N,1)`. Optional `metadata` is a JSON string. The report formula comes from the registry, so other datasets are not mislabeled with the paper's function.

## Validation and provenance

```bash
python -m unittest -v test_experiment
BDKN_SOURCE_REPO=../BDKN_Experiments python -m unittest -v test_experiment
```

`provenance.json` records source hashes and adaptations. Tests cover data generation, preprocessing, model math/gradients against the source, checkpoint reuse, both-stage loss plots, log axes, heatmap colors, registry validation, and profile navigation. See `VALIDATION.md` for completed runs and verification limits. FoRDE's BSD license is retained under `licenses/`. Data/results/site assets are generated and ignored by Git.

The repulsion sweep includes 0.001, 0.01, 0.1, 1, and 10. The Losses page has a dedicated unweighted repulsion-loss R plot for each repulsive method. Compare weights shows a separate prediction and R-loss panel for each strength. Both loss axes default to log; signed log preserves zero/nonpositive values.
