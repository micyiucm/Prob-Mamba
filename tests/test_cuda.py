import pytest
import torch

from prob_mamba.models import ProbMambaHead


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable on this host")
def test_probabilistic_head_executes_on_cuda():
    device = torch.device("cuda")
    head = ProbMambaHead(d_feat=3, d_y=1, n_state=2).to(device)
    features = torch.randn(2, 4, 3, device=device)
    targets = torch.randn(2, 4, 1, device=device)
    output = head(features, targets)
    output["nll"].backward()
    torch.cuda.synchronize()
    assert output["y_mean"].is_cuda
    assert torch.isfinite(output["nll"])
