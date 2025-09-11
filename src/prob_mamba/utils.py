from typing import Tuple
import time
import torch
import torch.nn.functional as F

def softplus_pos(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return F.softplus(x) + eps

def gamma(z: torch.Tensor, tol: float = 1e-4) -> torch.Tensor:
    out = torch.empty_like(z)
    small = z.abs() < tol
    out[~small] = torch.expm1(z[~small]) / z[~small]
    zb = z[small]
    out[small] = 1.0 + 0.5 * zb + (zb * zb) / 6.0
    return out

def rho(z: torch.Tensor, tol: float = 1e-4) -> torch.Tensor:
    out = torch.empty_like(z)
    small = z.abs() < tol
    out[~small] = (torch.exp(2.0 * z[~small]) - 1.0) / (2.0 * z[~small])
    zb = z[small]
    out[small] = 1.0 + zb + (2.0/3.0) * (zb * zb)
    return out

def batch_diag_embed(v: torch.Tensor) -> torch.Tensor:
    return torch.diag_embed(v)

def safe_cholesky(M: torch.Tensor, jitter: float = 1e-6) -> Tuple[torch.Tensor, float]:
    try:
        return torch.linalg.cholesky(M), 0.0
    except Exception:
        I = torch.eye(M.size(-1), device=M.device, dtype=M.dtype)
        j = jitter
        for _ in range(8):
            try:
                return torch.linalg.cholesky(M + j * I), j
            except Exception:
                j *= 10.0
        return torch.linalg.cholesky(M + j * I), j

def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

