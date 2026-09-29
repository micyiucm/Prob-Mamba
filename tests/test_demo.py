import json

import numpy as np
import pandas as pd
import torch

from prob_mamba.demo import run_example
from prob_mamba.evaluation import assert_common_scoring_support
from prob_mamba.models import ProbMambaHead


def test_cpu_example_runs_without_git_and_saves_reloadable_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path / "no-executables"))
    result = run_example(tmp_path / "output", epochs=1)
    output = tmp_path / "output"

    assert {path.name for path in output.iterdir()} == {
        "best.pt", "predictions.csv", "scores.json", "training.json", "metadata.json",
    }
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["git_revision"] is None
    assert metadata["with_classical"] is False
    assert len(metadata["package_source_sha256"]) == 64
    assert result["test_origins"] == 50

    predictions = pd.read_csv(output / "predictions.csv")
    tables = [frame for _, frame in predictions.groupby("model")]
    assert len(tables) == 2
    assert all(len(frame) == 50 for frame in tables)
    assert_common_scoring_support(tables)
    assert np.isfinite(predictions[["y_true", "mean", "variance"]]).all().all()
    assert (predictions["variance"] > 0).all()

    checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    model = ProbMambaHead(**metadata["head_config"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
