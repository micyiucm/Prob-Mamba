from typing import Optional, Tuple, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from prob_mamba.utils import softplus_pos, gamma, rho, safe_cholesky, batch_diag_embed

class SimpleRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, batch_first=True) # tanh activation
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.rnn(x, h0)  # out: (B, T, hidden_dim)
        out = self.fc(out[:, -1, :])  # Take last time step
        return out
    
def require_mamba_block():
    try:
        from mamba_ssm import Mamba  # official block
    except Exception as e:
        raise ImportError(
            "This model requires the official `mamba-ssm` package. "
            "Please install it (GPU build if you have CUDA)."
        ) from e
    return Mamba

class MambaModel(nn.Module):
    """
    Input proj: (input_dim -> d_model)
    n_layers of Mamba blocks (residual)
    Head: (d_model -> output_dim)
    """
    def __init__(self, input_dim, d_model=128, d_state=16, d_conv=4, expand=2,
                 output_dim=1, n_layers=1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model, bias=True)
        self.blocks = nn.ModuleList([
            Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)              # (B, L, d_model)
        for blk in self.blocks:
            h = h + blk(h)               # residual stack
        h = self.norm(h)
        h_last = h[:, -1, :]             # last-step regression
        return self.fc(h_last)           # (B, 1)
    
# Deterministic feature net
class FeatureNet(nn.Module):
    """
    Uses the official mamba_ssm.Mamba.
    (B, T, d_in) -> (B, T, d_feat)
    """
    def __init__(self, d_in: int, d_feat: int, n_layers: int = 1, mamba_cfg: Optional[Dict] = None):
        super().__init__()
        Mamba = require_mamba_block()
        cfg = mamba_cfg or {}

        self.proj_in = nn.Linear(d_in, d_feat)
        self.blocks = nn.ModuleList([
            Mamba(
                d_model=d_feat,
                d_state=cfg.get("d_state", 16),
                d_conv=cfg.get("d_conv", 4),
                expand=cfg.get("expand", 2),
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj_in(x)
        for blk in self.blocks:
            z = z + blk(z)  # residual
        return self.norm(z)
    
# Probabilistic Mamba Head (LGSSM)
class ProbMambaHead(nn.Module):
    """
    LGSSM with time-varying (from features) B_k, C_k, Q_k (diag), R_k (diag).
    Vectorized over time for speed; numerically stabilized for reliable NLL.
    """
    def __init__(self, d_feat: int, d_y: int, n: int, eps_pos: float = 1e-6):
        super().__init__()
        self.d_feat, self.d_y, self.n = d_feat, d_y, n
        self.eps_pos = eps_pos

        # continuous-time negative rates (for Abar = exp(Δ * a))
        self.a_raw = nn.Parameter(torch.randn(n) * 0.02)

        # selective maps (applied to each feature x_t)
        self.gate    = nn.Linear(d_feat, 1)          # Δ_t
        self.map_B   = nn.Linear(d_feat, n * d_feat) # (n, d_feat)
        self.map_C   = nn.Linear(d_feat, d_y * n)    # (d_y, n)
        self.map_sig = nn.Linear(d_feat, n)          # σ_t (proc noise diag)
        self.map_R   = nn.Linear(d_feat, d_y)        # R_t (meas noise diag)

        for lin in (self.gate, self.map_B, self.map_C, self.map_sig, self.map_R):
            nn.init.zeros_(lin.bias)

        # prior covariance diagonal
        self.p0 = nn.Parameter(torch.full((n,), 1e-2))

        # --- numeric stabilizers (tune if needed) ---
        self.delta_min = 1e-3
        self.delta_max = 1.0
        self.z_clip    = 20.0
        self.sigma_floor = 1e-3    # process noise floor
        self.R_floor     = 1e-4    # measurement noise floor

    @staticmethod
    def _reshape_time(x2d, B, T, *shape_tail):
        return x2d.view(B, T, *shape_tail)

    def forward(self, x_feat: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        x_feat: (B,T,d_feat)
        y:      (B,T,d_y)  (optional during inference)
        returns dict with keys: y_mean, y_var_diag, nll (if y), avg_ll_per_step, used_jitter
        """
        Bsz, T, Df = x_feat.shape
        n, d_y = self.n, self.d_y
        device = x_feat.device
        dtype  = x_feat.dtype

        # ---- 1) Precompute all per-time maps in one go (fast) ----
        X2 = x_feat.reshape(Bsz * T, Df)

        Delta = softplus_pos(self.gate(X2), self.eps_pos)                    # (BT,1)
        Delta = Delta.clamp(min=self.delta_min, max=self.delta_max)
        B_flat = self.map_B(X2)                                              # (BT, n*d_feat)
        C_flat = self.map_C(X2)                                              # (BT, d_y*n)
        sig    = softplus_pos(self.map_sig(X2), self.eps_pos) + self.sigma_floor  # (BT,n)
        Rdiag  = softplus_pos(self.map_R(X2),   self.eps_pos) + self.R_floor      # (BT,d_y)

        Δ = self._reshape_time(Delta, Bsz, T, 1)                             # (B,T,1)
        Bk = self._reshape_time(B_flat, Bsz, T, n, Df)                       # (B,T,n,d_feat)
        Ck = self._reshape_time(C_flat, Bsz, T, d_y, n)                      # (B,T,d_y,n)
        σ  = self._reshape_time(sig,   Bsz, T, n)                            # (B,T,n)
        Rk = self._reshape_time(Rdiag, Bsz, T, d_y)                          # (B,T,d_y)

        # ---- 2) Discretisation terms ----
        a = -softplus_pos(self.a_raw)                                        # (n,)  negative
        z = Δ * a.view(1, 1, n)                                              # (B,T,n)
        z = z.clamp(min=-self.z_clip, max=self.z_clip)
        exp_z = torch.exp(z)                                                 # (B,T,n)
        gamma_z = gamma(z)                                                   # (B,T,n)
        rho_z   = rho(z)                                                     # (B,T,n)

        # ---- 3) Kalman recursion ----
        h = torch.zeros(Bsz, n, device=device, dtype=dtype)                  # state mean
        P = torch.diag_embed(self.p0.abs().unsqueeze(0).expand(Bsz, -1))     # (B,n,n)
        eye_n = torch.eye(n, device=device, dtype=dtype)

        means_y, vars_y, ll_terms, jitters = [], [], [], []

        for t in range(T):
            xt   = x_feat[:, t, :]                 # (B,d_feat)
            Bt   = Bk[:, t, :, :]                  # (B,n,d_feat)
            Ct   = Ck[:, t, :, :]                  # (B,d_y,n)  <-- note orientation
            Δt   = Δ[:, t, :]                      # (B,1)
            exp_t = exp_z[:, t, :]                 # (B,n)
            γt    = gamma_z[:, t, :]               # (B,n)
            ρt    = rho_z[:, t, :]                 # (B,n)
            σt    = σ[:, t, :]                     # (B,n)
            Rt    = Rk[:, t, :]                    # (B,d_y)

            # Abar (diag) & Bbar
            Abar = batch_diag_embed(exp_t)         # (B,n,n)
            Bbar = (γt * Δt).unsqueeze(-1) * Bt    # (B,n,d_feat)

            # predict: h_pred = exp_t ⊙ h + Bbar @ x_t
            hx = torch.bmm(Bbar, xt.unsqueeze(-1)).squeeze(-1)               # (B,n)
            h_pred = exp_t * h + hx                                          # (B,n)

            # P_pred = Abar P Abar^T + Q_t (diag)
            P_pred = Abar @ P @ Abar.transpose(1, 2)
            q_diag = (σt ** 2) * ρt * Δt         # (B,n)
            P_pred = P_pred + torch.diag_embed(q_diag)                        # add diag without forming dense Q

            # measurement
            y_pred = torch.bmm(Ct, h_pred.unsqueeze(-1)).squeeze(-1)         # (B,d_y)
            means_y.append(y_pred)

            # S = C P_pred C^T + R
            CP = torch.bmm(Ct, P_pred)        
            
            if d_y == 1:
                cpT = torch.bmm(P_pred, Ct.transpose(1, 2))                 # (B,n,1)
                s = (torch.bmm(Ct, cpT).squeeze(-1).squeeze(-1) + Rt.squeeze(-1))  # (B,)
                s = s.clamp_min(self.R_floor)
                if y is not None:
                    e = (y[:, t, 0] - y_pred[:, 0])                          # (B,)
                else:
                    e = torch.zeros_like(s)

                K = (cpT.squeeze(-1) / s.unsqueeze(-1))                      # (B,n)
                h = h_pred + K * e.unsqueeze(-1)                             # (B,n)

                I = eye_n.unsqueeze(0)                                       # (1,n,n)
                KC = torch.bmm(K.unsqueeze(-1), Ct)                          # (B,n,n)
                I_KC = I - KC
                P = torch.bmm(I_KC, torch.bmm(P_pred, I_KC.transpose(1, 2))) + \
                    (Rt.squeeze(-1)).unsqueeze(-1).unsqueeze(-1) * torch.bmm(K.unsqueeze(-1), K.unsqueeze(1))

                if y is not None:
                    logdetS = torch.log(s)
                    ll = -0.5 * (logdetS + (e * e) / s + math.log(2.0 * math.pi))
                    ll_terms.append(ll)

                vars_y.append(s.unsqueeze(-1))                                # (B,1)
                jitters.append(torch.tensor(0.0, device=device, dtype=dtype)) # keep list aligned
            else:
                S  = torch.bmm(CP, Ct.transpose(1, 2)) + torch.diag_embed(Rt)    # (B,d_y,d_y)
                L, used_j = safe_cholesky(S, jitter=1e-6)
                jitters.append(torch.tensor(used_j, device=device, dtype=dtype))

                if y is not None:
                    e = y[:, t, :] - y_pred                                      # (B,d_y)
                    X = torch.cholesky_solve(CP, L)                               # (B,d_y,n)
                    K = X.transpose(1, 2)                                         # (B,n,d_y)
                    h = h_pred + torch.bmm(K, e.unsqueeze(-1)).squeeze(-1)       # (B,n)
                    I_KC = eye_n.unsqueeze(0) - torch.bmm(K, Ct)                 # (B,n,n)
                    P = I_KC @ P_pred @ I_KC.transpose(1, 2) + torch.bmm(K, torch.diag_embed(Rt)) @ K.transpose(1, 2)
                    logdetS = 2.0 * torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(-1)
                    v = torch.cholesky_solve(e.unsqueeze(-1), L).squeeze(-1)
                    ll = -0.5 * (logdetS + (e * v).sum(-1) + d_y * math.log(2.0 * math.pi))
                    ll_terms.append(ll)
                else:
                    h, P = h_pred, P_pred

                vars_y.append(torch.diagonal(S, dim1=-2, dim2=-1))            # (B,d_y)

        out: Dict[str, torch.Tensor] = {
            "y_mean":     torch.stack(means_y, dim=1),      # (B,T,d_y)
            "y_var_diag": torch.stack(vars_y,  dim=1),      # (B,T,d_y)
            "a_continuous": -softplus_pos(self.a_raw).detach(),
            "used_jitter":  (torch.stack(jitters).max() if len(jitters) > 0
                             else torch.tensor(0.0, device=device, dtype=dtype)),
        }
        if y is not None and ll_terms:
            ll_mat = torch.stack(ll_terms, dim=1)            # (B,T)
            out["nll"] = -ll_mat.sum(dim=1).mean()
            out["avg_ll_per_step"] = ll_mat.mean()
        return out


class ProbabilisticMamba(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_feat: int,
        d_y: int,
        n_state: int,
        n_mamba_layers: int = 1,
        mamba_cfg: Optional[Dict] = None,
    ):
        super().__init__()
        self.features = FeatureNet(
            d_in=d_in, d_feat=d_feat, n_layers=n_mamba_layers, mamba_cfg=mamba_cfg
        )
        self.lgssm = ProbMambaHead(d_feat=d_feat, d_y=d_y, n=n_state)

    def forward(self, x_raw: torch.Tensor, y: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        z = self.features(x_raw)
        return self.lgssm(z, y)

def softplus_pos(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Strictly-positive softplus used for Σ and R diagonals (and to make a_i ≤ 0 via -softplus).
    Adds a small ε to guarantee SPD behaviour in practice.  (ε ≈ 1e-6 in the thesis.)
    """
    return F.softplus(x) + eps
