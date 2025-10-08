# Prob-Mamba: A Probabilistic Extension of Mamba for Time-Series Forecasting

**Prob-Mamba** augments the Mamba selective state-space model with **stochastic dynamics**, turning the latent ODE into an **input-dependent SDE**. Under **zero-order hold (ZOH)**, this yields a **time-varying Linear Gaussian State-Space Model (LGSSM)** with **exact Kalman inference** for principled predictive uncertainty. This project is part of my MSc Machine Learning dissertation at University College London (UCL).

---

## Project Highlighs

- **What:** A probabilistic counterpart to Mamba that models both **means and variances** over time.  
- **How:** Inject Gaussian diffusion into Mamba’s state update; ZOH-discretise to a time-varying LGSSM; train with **Kalman-filter NLL**.  
- **Why:** Get uncertainty-aware forecasts with a clear link to Bayesian filtering. *(Trade-off: it’s slower than deterministic Mamba.)*

---

## Key Contributions (Theory)

### 1) Stochastic selective dynamics
Replace Mamba’s deterministic state equation with an **input-dependent SDE**:
$$
\mathrm{d}h_t \;=\; \big(A h_t + B(x_t)\,x_t\big)\,\mathrm{d}t \;+\; \Sigma(x_t)\,\mathrm{d}W_t,
\qquad
y_t \;=\; C(x_t)\,h_t \;+\; \varepsilon_t,
$$
with heteroskedastic observation noise \( \varepsilon_t \sim \mathcal{N}\!\big(0,\,R(x_t)\big) \).  
The maps \( B, C, \Sigma, R \) are learned and **input-dependent**.

### 2) Well-posedness
Under standard **global-Lipschitz** and **linear-growth** conditions on the learned maps, the SDE admits a **unique strong solution**, ensuring the discretised model faithfully represents the continuous dynamics.

### 3) ZOH ⇒ time-varying LGSSM + exact inference
With **ZOH**, the SDE discretises to
$$
h_{k+1} \;=\; \bar{A}_k\,h_k \;+\; \bar{B}_k\,x_k \;+\; \eta_k, 
\qquad
y_k \;=\; C(x_k)\,h_k \;+\; \varepsilon_k,
$$
with input-dependent \( \bar{A}_k,\ \bar{B}_k,\ Q_k,\ R_k \), so we can train with **exact Kalman filtering**.