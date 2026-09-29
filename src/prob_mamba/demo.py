"""Run a small CPU-only training and forecasting example on generated data.

This validates the pipeline; its scores are not financial benchmark results.
Run: python -m prob_mamba.demo --output-dir runs/cpu-example
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .baselines import rolling_arma_garch_forecast, rolling_ewma_forecast
from .data import preprocess_frame
from .datasets import create_causal_windows
from .evaluation import assert_common_scoring_support, predict_causal_windows, score_prediction_frame
from .models import ProbMambaHead
from .training import train_probabilistic_model


def _source_revision(source_dir: Path) -> str | None:
    """Record the source checkout revision when available; Git is optional."""
    repository = source_dir.resolve().parent.parent
    if not (repository / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository,
            capture_output=True, text=True, check=False, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_example(output_dir: Path, epochs: int = 5, seed: int = 7, with_classical: bool = False):
    """Generate a series, fit a small LGSSM head, and save auditable predictions."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    n_rows, window_length = 240, 12
    latent = np.zeros(n_rows)
    for step in range(1, n_rows):
        latent[step] = 0.85 * latent[step - 1] + rng.normal(scale=0.002)
    returns = latent + rng.normal(scale=0.003, size=n_rows)
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="D")
    raw = pd.DataFrame({"Date": dates, "Price": 100.0 * np.exp(np.cumsum(returns))})
    train_end, validation_end = dates[139].isoformat(), dates[189].isoformat()
    split, scaler = preprocess_frame(raw, train_end, validation_end, "Date", "Price")
    full = split.chronological()
    feature_columns = ["ret_t"]
    windows = {
        name: create_causal_windows(
            full, window_length, feature_columns,
            score_target_timestamps=frame["target_timestamp"], require_all=name != "train",
        )
        for name, frame in (("train", split.train), ("validation", split.validation), ("test", split.test))
    }
    loaders = {
        name: DataLoader(
            TensorDataset(torch.from_numpy(data.inputs), torch.from_numpy(data.target_sequences)),
            batch_size=32, shuffle=name == "train",
        )
        for name, data in windows.items() if name != "test"
    }
    head_config = {
        "d_feat": len(feature_columns), "d_y": 1, "n_state": 4,
        "target_scale": float(split.train["y_next"].std(ddof=1)),
        "variance_floor_ratio": 1e-8,
    }
    model = ProbMambaHead(**head_config)
    # Start the input forcing near the target scale, avoiding an arbitrary
    # unit-scale initial mean for small return targets.
    with torch.no_grad():
        model.map_b.weight.mul_(head_config["target_scale"])
    source_dir = Path(__file__).parent
    source_hash = hashlib.sha256()
    for path in sorted(source_dir.glob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    metadata = {
        "purpose": "synthetic CPU forecasting example",
        "seed": seed, "epochs": epochs, "sequence_length": window_length,
        "with_classical": with_classical,
        "feature_columns": feature_columns, "head_config": head_config,
        "train_target_cutoff": train_end, "validation_target_cutoff": validation_end,
        "loss_scope": "last_step", "git_revision": _source_revision(source_dir),
        "package_source_sha256": source_hash.hexdigest(),
        "data_sha256": hashlib.sha256(raw.to_csv(index=False).encode()).hexdigest(),
        "scaler": {"columns": list(scaler.feature_names_in_), "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "versions": {"python": platform.python_version(), **{name: version(name) for name in ("torch", "numpy", "pandas", "scikit-learn")}},
    }
    if with_classical:
        metadata["versions"].update({name: version(name) for name in ("arch", "statsmodels")})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = train_probabilistic_model(
        model, loaders["train"], loaders["validation"], epochs,
        device="cpu", checkpoint_path=output_dir / "best.pt", checkpoint_metadata=metadata,
    )
    learned = predict_causal_windows(
        model, windows["test"], dataset="synthetic", model_name="LGSSM-head", seed=seed,
    )
    historical = pd.concat([split.train, split.validation], ignore_index=True)
    tables = [learned, rolling_ewma_forecast(historical, split.test, dataset="synthetic")]
    if with_classical:
        _, classical, _, _ = rolling_arma_garch_forecast(
            split.train, split.validation, split.test, max_ar=1, max_ma=0,
            scale=1.0 / head_config["target_scale"], dataset="synthetic",
        )
        tables.append(classical)
    assert_common_scoring_support(tables)
    scores = {table["model"].iloc[0]: score_prediction_frame(table) for table in tables}
    pd.concat(tables, ignore_index=True).to_csv(output_dir / "predictions.csv", index=False)
    for name, content in (("metadata", metadata), ("training", history), ("scores", scores)):
        (output_dir / f"{name}.json").write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    return {"output_dir": str(output_dir.resolve()), "test_origins": len(learned), "best_epoch": history["best_epoch"], "scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cpu-example"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--with-classical", action="store_true", help="also exercise optional ARMA-GARCH")
    args = parser.parse_args()
    print(json.dumps(run_example(args.output_dir, args.epochs, args.seed, args.with_classical), indent=2))


if __name__ == "__main__":
    main()
