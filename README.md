# Prob-Mamba: A Probabilistic Extension of Mamba for Time-Series Forecasting

**Prob-Mamba** augments the Mamba selective state-space model with **stochastic dynamics**, turning the latent ODE into an **input-dependent SDE**. Under **zero-order hold (ZOH)**, this yields a **time-varying Linear Gaussian State-Space Model (LGSSM)** with **exact Kalman inference** for predictive uncertainty.  
*(This project was part of my MSc Machine Learning dissertation at University College London.)*

---

## Project Highlights

- **What:** A probabilistic counterpart to Mamba that models both **means and variances** over time.  
- **How:** Inject Gaussian diffusion into Mamba’s state update; ZOH-discretise to a time-varying LGSSM; train with **Kalman-filter NLL**.  
- **Why:** Uncertainty-aware forecasts with a clear link to Bayesian filtering. *(Trade-off: slower than a deterministic Mamba model)*

---

## Key Contributions (Theory)

### 1) Stochastic selective dynamics
Replace Mamba's deterministic state equation with an **input-dependent SDE**:

$$
\text{d}h_t = (A h_t + B(x_t)x_t)\thinspace\text{d}t + \Sigma(x_t)\thinspace\text{d}W_t
$$

$$
y_t = C(x_t)h_t + \varepsilon_t
$$

with heteroskedastic, input-dependent observation noise  $\varepsilon_t \sim \mathcal{N}(0,R(x_t))$.  
The maps $B, C, \Sigma, R$ are learned and **input-dependent**.

### 2) Well-posedness
Under standard **global-Lipschitz** and **linear-growth** conditions on the learned maps, the SDE admits a **unique strong solution**, ensuring the discretised model faithfully represents the continuous dynamics.

### 3) ZOH ⇒ time-varying LGSSM + exact inference
With **ZOH**, the SDE discretises to:

$$
h_{k+1} = \bar A_k\,h_k + \bar B_k\,x_k + \eta_k,
\qquad
y_k = C(x_k)\,h_k + \varepsilon_k,
\qquad
\eta_k \sim \mathcal{N}(0,Q_k),\ \ \varepsilon_k \sim \mathcal{N}(0,R_k).
$$

The resulting model is conditionally linear-Gaussian, enabling **exact Kalman filtering** (and training via NLL).

> **Note:** The model outputs **predictive mean and variance**. Prediction intervals can be derived as
> \( \mu \pm z_\alpha \sigma \) under a Gaussian assumption.

---

## Computational Notes (Trade-offs)

- **Sequential & heavier compute:** Per-step Kalman predict–update with **time-varying** parameters makes training **significantly slower** than a deterministic head, especially on long or noisy series.  
