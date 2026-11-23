# Author-code regression comparison

`run_author.py` submits all selected experiments on physical GPU 0 and updates
the existing offline website as fits finish. Change the dataset, method and
preset lists at the beginning of that script. The default selection contains
181 fits. It reuses existing original-method results on the ten old datasets.

The page is `website/dist/author.html`, linked as **Author code** throughout the
existing site. It works through `file://`; no results server or CDN is required.
Refreshing the page loads newly exported results.

## Source and compatibility

Source: <https://github.com/dngfra/repulsive_ensembles/tree/6dcb4c068082d18b7e86e90f4459d63d7ba8a750>.
The README command is the source of the upstream preset; it is not presented as
an exact reproduction of every setting in the paper's appendix.

`author_source.lock.json` pins individual Git blob hashes. `author_code.py`
downloads missing files into an ignored cache and verifies every byte before
loading them. Original upstream files are not edited or redistributed here.
Keep network access available for the first run on a new machine.

The selected implementations are upstream `SGD`, `kde_WGD` and `kde_f_WGD`.
All use the actual upstream Adam optimizer, posterior and training function.
The method name `SGD` identifies an independent posterior-gradient ensemble,
not PyTorch's SGD optimizer. This repository is not the FoRDE author repository.

Four compatibility repairs are applied in memory and recorded in every fit:

- Replace removed `np.int` with built-in `int` in batch-index construction.
- Resolve the upstream factory's `f_SVGD` import to `methods/f_SVGD.py`.
- Replace removed `torch.eig` on the symmetric SSGE Gram matrix with
  `torch.linalg.eigh`, retaining all eigenpairs and adding no clipping/jitter.
- Remove the second, broken `SGD.step` definition, which overwrites the working
  first definition in the published file. Retain the first definition verbatim.

Preserved details include unit repulsion, the self-inclusive interpolated
median bandwidth, no bandwidth floor, SSGE regularization 0.01, H prior samples
per functional update, likelihood scaling by training count, and consumption
of a shuffled test batch each optimizer step. `prior_variance=1` is passed as
a Normal standard deviation, following the upstream code. Fixed likelihood
standard deviation is 1 even though the original training-data noise is 0.25.

Logging/export hooks do not sample random numbers or change optimizer updates.
Tests compare their resulting trajectories exactly with direct upstream calls.
Compatibility does not imply bitwise equivalence across hardware or library
versions. No claim is made that the broken checkout runs without these repairs.

## Presets and data

| Preset | Networks | Hidden layers | LR | Budget | Input normalization |
| --- | ---: | --- | ---: | --- | --- |
| Upstream README | 100 | 10, 10 ReLU | 0.01 | 5,000 optimizer steps | None |
| Our low LR | 5 | 50, 50 ReLU | 0.001 | 2,000 × ceil(N/32) updates | Training mean/std |
| Our high LR | 5 | 50, 50 ReLU | 0.1 | 40 × ceil(N/32) updates | Training mean/std |

Our-settings experiments retain the upstream shared, wrapping minibatch stream;
matching the number of updates does not turn this into our independent
per-network minibatch procedure. They also retain a scalar output and fixed
observation variance, unlike our original heteroscedastic ensembles.

`author-xsine` is distinct from `he`: upstream task 4 has 45 random observations
in [1.5, 2.5], 45 in [4.5, 6], noise std 0.25 and NumPy seed 42. Its native test
is the exact 100-point noiseless grid over [0, 7]. Our additional noisy test has
1,000 observations: those 100 coordinates plus 900 independent uniform inputs,
all with separately generated noise (std 0.25, default_rng seed 123). The native
targets remain separate. No test targets are used for fitting or tuning. The
author trainer sees its original 100-point test stream, not the new noisy stream.

On the new dataset, the original fixed methods run both LR settings, all five
repulsion weights and three input choices for functional/FoRDE repulsion.
Weight repulsion runs once per weight because its update has no input choice.
The original low-LR DKL and LDBLL budgets remain 10,000 epochs.

## Predictions and BDKN

Native plots preserve the sample standard deviation across upstream networks
and show mean ± 1.96 std, without observation noise. Comparative predictive
plots and Gaussian NLL use population ensemble variance plus the upstream fixed
likelihood variance, following this project's moment-matched Gaussian scoring.
These are two explicitly different uncertainty summaries.

BDKN extensions freeze upstream network output functions, use them as basis
columns and call this project's existing float64 Bayesian aggregation code.
They optimize the prior variance for 100 Adam steps at LR 0.1 with the existing
alpha0=beta0=0 settings. Their Student-t predictions and stage-two histories
are saved separately. They do not alter upstream network training.

Each fit saves configuration, source revision, dataset hash, initialization,
checkpoint, members, predictions, histories, metrics and timings. Completion
markers hash all scientific artifacts; incomplete/mismatched attempts are
retained separately. CUDA allocation failures return to the memory-aware queue;
numerical failures remain visible and never trigger changes to the recipe.

Run validation with `python -m unittest test_author -v` in the project environment.

## D'Angelo setup with author FoRDE repulsion

`run_official_forde.py` adds 66 fits (11 datasets × low/high/upstream × FoRDE/control)
to the **same Author code page**. Following the clarified experiment, these
use D'Angelo's regression foundation: actual initial parameter arrays exported
from their model, their shared train/test batch stream, fixed likelihood std 1,
N(0,I) parameter prior, and Adam (0.9, 0.999, epsilon 1e-8). Only the diversity
term changes to the literal author FoRDE JAX function. Low/high budgets match
the corresponding D'Angelo comparisons; upstream uses the D'Angelo README
configuration. The author FoRDE classification/Nesterov training recipe is not
used in these common-foundation comparisons.

The scalar network is expressed in Haiku and initialized by copying every
upstream PyTorch weight/bias (with the necessary dense-weight transpose).
Adam and Gaussian/prior gradients are evaluated in JAX. Small floating-point
differences across these frameworks are expected; this is not a claim of
bitwise identical complete trajectories.

Repulsion itself is extracted directly from the pinned FoRDE source AST.
Normalization epsilon 1e-24, per-reference medians, bandwidth floor 1e-12,
stop-gradients, log-sum-exp reductions and unit repulsion are preserved.
Training-only PCA uses the author's second-moment/SVD calculation. The matched
control differs only by setting repulsion to zero. In 1D the force can vanish
because normalized derivatives approach signs; its parameter-gradient norm
is saved even when zero. BDKN again fits only the frozen mean functions.

The isolated `.venv-forde` environment uses `requirements-forde.txt`; its full
resolved versions are in `requirements-forde.lock.txt`. JAX GPU preallocation
is disabled. First-use JIT time is included and identified in timings.


D’Angelo paper training recipe
-----------------------------
`run_paper.py` adds a separate `results/paper` bank for all eleven saved 1D
datasets. Select **D’Angelo paper** on the existing Author code comparison page.
All three D’Angelo methods and the FoRDE combination/control produce both DE
and BDKN predictions. The published recipe is 50 particles, two hidden layers
of 50 ReLU units, Adam learning rate 0.01, 15,000 optimizer steps (not epochs),
batch size 64, seed 42, standard normal initialization/prior and a standard
Gaussian likelihood. Input/target arrays are unchanged and inputs are raw.

The pinned upstream trainer, median bandwidth rule and compatibility fixes
are preserved. `paper_code` isolates the new adapters from current runs; its
only training changes are the preset and batch-size selection. Code is frozen
per output bank, and completed artifacts are validated before reuse.
`paper_code/test_paper.py` verifies exact short-run parameter trajectories
against the author trainer for all three D’Angelo methods.

This is the paper’s training recipe on our existing datasets, not a claim that
the paper evaluated every dataset. The released x sin x generator uses noise
**standard deviation** 0.25, while the paper writes N(0, 0.25); its native
100-point noiseless test grid is available as the existing native evaluation.
We preserve those released arrays for comparison rather than silently changing
noise or test targets. BDKN and D’Angelo + FoRDE are our extensions.

The matching UCI runner in the source repository is
`src_FoRDE/bdkn_vectorized/launch_paper_code_comparison.py`. It uses the same
paper recipe on the existing five datasets, Standard/GAP splits, and stores
its own bank at `experiments_author_code_paper`. Both banks refresh the
existing offline comparison pages as fits complete. GPU 0 is the sole device;
all selections are submitted together with memory-based admission and OOM
retry, without altering previously submitted experiments.
