"""E2M-ATQ: Euclidean-to-Manifold Asymmetric Ternary Quantizer.

Faithful reimplementation of TWLA (arXiv:2606.13054v2) Algorithm 3. Replaces
the previous GPTQ-hybrid quantizer (blockwise Hessian error propagation,
order-2 "two ternary bases" groups, and iterative Stage-II refinement) with
the paper's literal two-stage pipeline:

  Stage I  -- Euclidean warm-start (coordinate descent on mu, alpha, T under
              the Frobenius weight-domain objective, Eq. 4-6).
  Stage II -- Manifold relocation: freeze T, then solve the exact per-row
              2x2 closed form (Eq. 8-9 / 46-49) under the calibration-induced
              metric S = sum_b X_b^T X_b (Eq. 7), with the moment
              regularization and degenerate-row fallback from Appendix B.2
              (Eq. 51).

There is a single ternary codebook T in {-1,0,+1} per row (Eq. 3) -- no
GPTQ-style structural salient masks, no column blocking, no multi-basis
("order-2") reconstruction.
"""

from typing import Tuple

import torch
import torch.nn as nn


# -----------------------------
# Core primitives (Eq. 1, 2 / 24-26)
# -----------------------------
@torch.no_grad()
def ternary_threshold(Wc: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Eq. 1 (TWN-style row threshold).

    Wc: [n, m] centered residual (W - mu). delta: [n] per-row threshold.
    """
    d = delta[:, None]
    T = torch.zeros_like(Wc)
    T = torch.where(Wc > d, torch.ones_like(T), T)
    T = torch.where(Wc < -d, -torch.ones_like(T), T)
    return T


@torch.no_grad()
def support_aware_ls(Wc: torch.Tensor, T: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Eq. 2 / Eq. 25: alpha_i = sum_j T_ij Wc_ij / sum_j |T_ij| (zeros excluded)."""
    num = torch.sum(T * Wc, dim=1)
    den = torch.sum(torch.abs(T), dim=1) + eps
    return num / den


def _delta_from(Wc: torch.Tensor) -> torch.Tensor:
    """Eq. 1: Delta_i ~= 0.75/m * sum_j |W_ij - mu_i|."""
    return 0.75 * Wc.abs().mean(dim=1)


# -----------------------------
# Stage I: Euclidean warm-start (Algorithm 3, lines 1-10)
# -----------------------------
@torch.no_grad()
def euclidean_warm_start(
    W: torch.Tensor, iters: int = 15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stabilizes a ternary pattern T^(0) and Euclidean init (mu^(0), alpha^(0)).

    Per-iteration update order matches the pseudocode exactly: mu-update
    (residual-mean correction, Eq. 24), then alpha-update using the *old*
    T (support-aware LS, Eq. 25), then T-update using the new mu (Eq. 1/26).
    """
    mu = W.mean(dim=1)
    Wc = W - mu[:, None]
    delta = _delta_from(Wc)
    T = ternary_threshold(Wc, delta)
    alpha = support_aware_ls(Wc, T)

    for _ in range(int(iters)):
        R = W - mu[:, None] - alpha[:, None] * T   # residual under (old mu, old alpha, current T)
        mu = mu + R.mean(dim=1)                     # mu-update, Eq. 24
        Wc = W - mu[:, None]
        alpha = support_aware_ls(Wc, T)              # alpha-update, uses the *old* T, Eq. 25
        delta = _delta_from(Wc)
        T = ternary_threshold(Wc, delta)              # T-update, uses the new mu, Eq. 1

    return T, mu, alpha


# -----------------------------
# Stage II: manifold relocation, exact closed form (Eq. 8-9 / 46-51)
# -----------------------------
@torch.no_grad()
def manifold_relocation(
    W: torch.Tensor,
    T0: torch.Tensor,
    S: torch.Tensor,
    mu0: torch.Tensor,
    alpha0: torch.Tensor,
    reg_lambda: float = 1e-4,
    degenerate_tau: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Freeze T = T0 and solve the per-row 2x2 normal equations (Eq. 46)

        [t_i S t_i^T   t_i S 1] [alpha_i]   [t_i S w_i^T]
        [1^T S t_i^T   1^T S 1] [mu_i   ] = [1^T S w_i^T]

    in closed form via Cramer's rule (Eq. 49), with the moment-regularized
    S (Eq. 51) and a fallback to the Euclidean warm-start values for rows
    whose 2x2 system is (near-)singular (Appendix B.2, "Practical numerical
    stabilization").

    W: [n, m] original weights. T0: [n, m] frozen ternary pattern.
    S: [m, m] calibration second moment, S = sum_b X_b^T X_b (Eq. 7).
    """
    m = W.shape[1]

    eps = reg_lambda * torch.diagonal(S).mean().clamp_min(1e-20)
    S = S + eps * torch.eye(m, device=S.device, dtype=S.dtype)

    St = S @ T0.t()                     # [m, n], column i = S t_i^T
    S1 = S.sum(dim=1)                   # [m],   S @ 1  (S is symmetric)
    Sw = S @ W.t()                      # [m, n], column i = S w_i^T

    a = (T0 * St.t()).sum(dim=1)        # t_i S t_i^T, per row
    b = (T0 * S1[None, :]).sum(dim=1)   # t_i S 1
    c = S1.sum()                        # 1^T S 1 (scalar, same for every row)
    d = (T0 * Sw.t()).sum(dim=1)        # t_i S w_i^T
    e = (W * S1[None, :]).sum(dim=1)    # 1^T S w_i^T

    D = a * c - b.pow(2)                # Gram determinant, Eq. 48

    # S is PSD, so [t_i, 1] form a Gram matrix under the S-inner-product and
    # 0 <= D <= a*c (Cauchy-Schwarz) -- D is only meaningfully "nonsingular"
    # relative to the system's own scale (a, c), not in absolute terms.
    # An absolute `D.abs() > degenerate_tau` check (the original form here)
    # passes for any row where a and c happen to be large -- e.g. a routed
    # MoE expert with few but high-magnitude calibration hits -- even when
    # D/(a*c) is tiny and the 2x2 solve is numerically unstable. Cramer's
    # rule then divides by a near-zero-relative-to-scale D and alpha*/mu*
    # blow up to 10-30x the row's true weight magnitude (confirmed
    # empirically: checkpoints/quantized_calibset232 layer 20's
    # mlp.experts.0.{gate_proj,down_proj} had dozens of rows at
    # 20-30x-inflated scale versus every other -- well-conditioned --
    # expert in the same layer, despite calibset232 already fixing the
    # separate 0-calibration-hit/dead-expert case this same check *does*
    # correctly catch when S=0 makes a=c=D=0). Comparing D against a*c
    # directly makes the check scale-invariant while still reducing to the
    # original dead-expert behavior when a=c=0.
    safe = D > degenerate_tau * (a * c).clamp_min(0)
    D_safe = torch.where(safe, D, torch.ones_like(D))

    alpha_cf = (d * c - b * e) / D_safe   # Eq. 49
    mu_cf = (a * e - b * d) / D_safe      # Eq. 49

    alpha_star = torch.where(safe, alpha_cf, alpha0)
    mu_star = torch.where(safe, mu_cf, mu0)

    return mu_star, alpha_star


# -----------------------------
# Full two-stage E2M-ATQ (Algorithm 3, end to end)
# -----------------------------
@torch.no_grad()
def e2m_atq_quantize(
    W: torch.Tensor,
    S: torch.Tensor,
    euclidean_iters: int = 15,
    reg_lambda: float = 1e-4,
    degenerate_tau: float = 1e-8,
) -> torch.Tensor:
    """Runs Stage I then Stage II and returns the reconstructed weights

        W_bar = mu* 1^T + diag(alpha*) T^(0)     (Eq. 10 / 52).
    """
    T0, mu0, alpha0 = euclidean_warm_start(W, iters=euclidean_iters)
    mu_star, alpha_star = manifold_relocation(
        W, T0, S, mu0, alpha0, reg_lambda=reg_lambda, degenerate_tau=degenerate_tau
    )
    return mu_star[:, None] + alpha_star[:, None] * T0


class E2MATQCalibrator:
    """Accumulates the calibration second moment S = sum_b X_b^T X_b (Eq. 7)
    for one nn.Linear layer via a forward-hook lifecycle, then runs E2M-ATQ
    (Algorithm 3) to replace its weight with the ternarized reconstruction.

    Mirrors the add_batch-then-quantize lifecycle the previous GPTQ-based
    quantizer used, so calibration orchestration code doesn't need to change
    shape -- but there is no Hessian, no damping, and no column-sequential
    error propagation, just the plain second-moment accumulation Eq. 7
    defines.
    """

    def __init__(self, layer: nn.Module):
        self.layer = layer
        self.dev = layer.weight.device
        self.columns = int(layer.weight.data.shape[1])
        self.S = torch.zeros((self.columns, self.columns), device=self.dev, dtype=torch.float32)
        self.nsamples = 0

    @torch.no_grad()
    def add_batch(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        if getattr(self.layer, "L", None) is not None:
            init_shape = inp.shape
            inp = inp.reshape(-1, self.layer.dim_l, self.layer.dim_r)
            inp = self.layer.L @ inp @ self.layer.R
            inp = inp.reshape(init_shape)

        if inp.dim() == 2:
            inp = inp.unsqueeze(0)
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.t().float()

        self.S += inp @ inp.t()
        self.nsamples += int(inp.shape[1])

    @torch.no_grad()
    def quantize(
        self,
        euclidean_iters: int = 15,
        reg_lambda: float = 1e-4,
        degenerate_tau: float = 1e-8,
    ) -> dict:
        W = self.layer.weight.data.float()
        W_bar = e2m_atq_quantize(
            W, self.S,
            euclidean_iters=euclidean_iters,
            reg_lambda=reg_lambda,
            degenerate_tau=degenerate_tau,
        )
        err = torch.sum((W - W_bar) ** 2).item()
        self.layer.weight.data = W_bar.contiguous().reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if torch.any(torch.isnan(self.layer.weight.data)):
            raise ValueError("NaN in weights after E2M-ATQ")
        return {"error": err}

    def free(self) -> None:
        self.S = None
        torch.cuda.empty_cache()
