# Prob-Mamba

Probabilistic time-series forecasting with a Mamba feature encoder and a linear
Gaussian state-space head. The encoder produces causal features; the head uses
those features to parameterize latent dynamics and observation noise, then applies
Kalman filtering to produce predictive means and covariances.

The package includes chronological preprocessing, rolling forecasting baselines,
training, evaluation, and a complete CPU example with generated data. The
standalone state-space head runs on CPU; the Mamba encoder uses an optional GPU
backend.

## Quick start

Use Python 3.10 or newer; the CPU workflow has been tested with Python 3.12.
From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m prob_mamba.demo --output-dir runs/cpu-example
python -m pytest -q
```

The example generates a small latent-state time series, trains the state-space
head, selects a checkpoint using validation NLL, and forecasts 50 test observations
alongside an EWMA baseline. It needs no downloaded data, notebooks, Git executable,
or GPU. It exercises the forecasting pipeline; it does not train the Mamba encoder
or establish performance on financial data.

To include the rolling ARMA-GARCH baseline:

```bash
python -m pip install -e '.[baselines,dev]'
python -m prob_mamba.demo --output-dir runs/cpu-classical --with-classical
```

Run `python -m prob_mamba.demo --help` for seed, epoch, and output options.
Each run writes:

- `best.pt`: validation-selected model and optimizer state, with configuration.
- `predictions.csv`: forecast origins, target timestamps, outcomes, means, and variances.
- `scores.json`: forecast errors, likelihood scores, and uncertainty diagnostics.
- `training.json`: training and validation losses and selected epoch.
- `metadata.json`: seed, feature order, scaling, split cutoffs, dependency versions,
  data and source hashes, and source revision when available.

## Optional dependencies

| Extra | Purpose |
| --- | --- |
| `baselines` | ARMA, GARCH, and VIF feature selection |
| `data` | Feather/Arrow file loading |
| `dev` | Tests |
| `mamba` | Mamba encoder and convolution extension |

For example, `python -m pip install -e '.[data,baselines,dev]'` installs the CPU
tools and tests. `python -m pip install -r requirements.txt` installs the same set.
Tests for unavailable optional dependencies are skipped.

### Mamba setup

The Mamba-1 encoder uses a compiled selective-scan extension. On Linux with a
supported NVIDIA GPU, first install a CUDA-enabled PyTorch build suitable for your
machine, then install the optional backend:

```bash
python -m pip install setuptools wheel packaging ninja
MAMBA_KEEP_CUDA_BUILD=TRUE python -m pip install --no-build-isolation -e '.[mamba]'
```

The environment variable opts into the selective-scan extension in current Mamba
builds. See the [upstream installation instructions](https://github.com/state-spaces/mamba#installation)
for platform and CUDA requirements. The full encoder has not been validated in the
CPU environment used for this repository's tests.

## Model API

`ProbMambaHead` accepts features shaped `(batch, steps, features)` and optional
observations shaped `(batch, steps, outputs)`. For financial returns, set
`target_scale` to the standard deviation of the training targets, in the units
passed to the model.

```python
import torch
from prob_mamba import ProbMambaHead

training_target_std = 0.004  # substitute your training-only estimate
head = ProbMambaHead(d_feat=3, d_y=1, n_state=4,
                     target_scale=training_target_std)
features = torch.randn(2, 12, 3)
targets = torch.randn(2, 12, 1) * training_target_std
observed = torch.ones(2, 12, dtype=torch.bool)
observed[:, -1] = False
targets[:, -1] = float('nan')

prediction = head(features, targets, observed_mask=observed)
mean = prediction['y_mean'][:, -1]
variance = prediction['y_var_diag'][:, -1]
```

Predictions are computed before assimilating the corresponding observation.
Missing observations skip the update. The variance floor scales with
`target_scale**2`; the default `target_scale=1` is suitable for unit-scale targets.

To add the Mamba encoder after installing its backend:

```python
from prob_mamba import ProbabilisticMamba

model = ProbabilisticMamba(
    d_in=3, d_feat=32, d_y=1, n_state=4,
    head_cfg={'target_scale': training_target_std},
).to('cuda')
```

The head also exposes `predict_step`, `update_step`, `final_state_mean`, and
`final_state_covariance`. These support continuation over precomputed features.
Continuing the complete encoder requires its recurrent and convolution states too.

## Using your own data

Start with a pandas DataFrame containing unique timestamps, finite positive prices,
and any numeric features available at each forecast origin. Call `preprocess_frame`
with the date and price column names and explicit training/validation cutoffs. It
constructs next-observation log returns, fits feature scaling on training data,
and returns chronological train, validation, and test partitions. Targets remain
in their original return units. Feature publication times must be checked by the
caller; preprocessing cannot infer when an external feature was available.

Use the [complete example](src/prob_mamba/demo.py) as a template for connecting
preprocessing, data loaders, training, and scoring:

1. Build windows from `split.chronological()` with `create_causal_windows`, selecting
   each partition's target timestamps. This retains earlier context for validation
   and test. Insufficient context raises an error unless explicitly allowed.
2. Train with `train_probabilistic_model`. Its default `loss_scope='last_step'`
   scores window endpoints while historical observations condition the filter.
   The best validation checkpoint is restored automatically.
3. Forecast with `predict_causal_windows`, which withholds endpoint labels and
   returns timestamped scalar predictions. Each window starts from the head prior.
4. Use `assert_common_scoring_support` before comparing prediction tables, then
   `score_prediction_frame` to compute scores on matching observations.

`rolling_arma_forecast`, `rolling_arma_garch_forecast`,
`rolling_zero_mean_garch_forecast`, and `rolling_ewma_forecast` update from newly
observed test outcomes. Supply the complete chronological test sequence, then
select any common scoring subset. ARMA-based helpers select orders on training
data and fit parameters on train/validation observations available at the first
test origin. Match parameter-fitting and checkpoint-selection schedules when
designing a controlled comparison.

## Interpretation

The stochastic component is the state-space head; the Mamba encoder is
deterministic. The head uses exact zero-order-hold discretization and Joseph
covariance updates. Exact conditional Gaussian filtering requires a Gaussian
prior, fresh independent innovations, and coefficients determined by available
information. Learned parameters are treated as fixed during inference.

Evaluation includes RMSE, Gaussian NLL, residual QLIKE, interval coverage/width,
and summaries of probability integral transforms and standardized innovations.
For the same scalar forecasts, `NLL = (QLIKE + log(2*pi)) / 2`, so these two scores
provide equivalent rankings. Coverage and PIT summaries are diagnostics and do
not by themselves establish calibration. Zero-mean-proxy QLIKE assumes a zero
conditional mean.

## Layout

- `src/prob_mamba/data.py`, `datasets.py`: preprocessing and forecast windows.
- `src/prob_mamba/models.py`, `numerics.py`: models and filtering operations.
- `src/prob_mamba/training.py`: training and checkpoint selection.
- `src/prob_mamba/evaluation.py`, `metrics.py`: prediction tables and scoring.
- `src/prob_mamba/baselines.py`, `features.py`: baselines and feature helpers.
- `src/prob_mamba/demo.py`: complete CPU workflow.
- `tests/`: numerical, forecasting, and integration checks.

## License

MIT; see [LICENSE](LICENSE).
