"""Arbitrary-level E2M-ATQ helpers for bidirectional precision allocation.

This is an experimental companion to :mod:`E2M_ATQ` and
:mod:`E2M_ATQ_mixed_precision`.  Three levels deliberately continue to use
the repository's literal TWLA ternary implementation; this module handles
the 4/8/16-level candidates with the same Stage-I -> Stage-II structure.
"""

from __future__ import annotations

from typing import Tuple

import torch


@torch.no_grad()
def support_aware_ls(Wc: torch.Tensor, T: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    numerator = torch.sum(T * Wc, dim=1)
    denominator = torch.sum(T.square(), dim=1) + eps
    return numerator / denominator


@torch.no_grad()
def multilevel_codes(Wc: torch.Tensor, alpha: torch.Tensor, num_levels: int) -> torch.Tensor:
    center = (int(num_levels) - 1) / 2.0
    safe_alpha = alpha.abs().clamp_min(1e-8)
    indices = torch.round(Wc / safe_alpha[:, None] + center)
    return indices.clamp_(0, int(num_levels) - 1).sub_(center)


@torch.no_grad()
def euclidean_warm_start(
    W: torch.Tensor, *, num_levels: int, iters: int = 15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if int(num_levels) < 2:
        raise ValueError(f"num_levels must be >=2, got {num_levels}")
    mu = W.mean(dim=1)
    Wc = W - mu[:, None]
    max_code = (int(num_levels) - 1) / 2.0
    alpha = Wc.abs().amax(dim=1).div(max_code).clamp_min(1e-8)
    T = multilevel_codes(Wc, alpha, num_levels)
    alpha = support_aware_ls(Wc, T)
    for _ in range(int(iters)):
        residual = W - mu[:, None] - alpha[:, None] * T
        mu = mu + residual.mean(dim=1)
        Wc = W - mu[:, None]
        alpha = support_aware_ls(Wc, T)
        T = multilevel_codes(Wc, alpha, num_levels)
    return T, mu, alpha


@torch.no_grad()
def manifold_relocation(
    W: torch.Tensor,
    T0: torch.Tensor,
    S: torch.Tensor,
    mu0: torch.Tensor,
    alpha0: torch.Tensor,
    *,
    reg_lambda: float = 1e-4,
    degenerate_tau: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    width = W.shape[1]
    epsilon = reg_lambda * torch.diagonal(S).mean().clamp_min(1e-20)
    Sreg = S + epsilon * torch.eye(width, device=S.device, dtype=S.dtype)
    St = Sreg @ T0.t()
    S1 = Sreg.sum(dim=1)
    Sw = Sreg @ W.t()
    a = (T0 * St.t()).sum(dim=1)
    b = (T0 * S1[None, :]).sum(dim=1)
    c = S1.sum()
    d = (T0 * Sw.t()).sum(dim=1)
    e = (W * S1[None, :]).sum(dim=1)
    determinant = a * c - b.square()
    safe = determinant > degenerate_tau * (a * c).clamp_min(0)
    denominator = torch.where(safe, determinant, torch.ones_like(determinant))
    alpha_cf = (d * c - b * e) / denominator
    mu_cf = (a * e - b * d) / denominator
    return torch.where(safe, mu_cf, mu0), torch.where(safe, alpha_cf, alpha0)


@torch.no_grad()
def e2m_atq_quantize_levels(
    W: torch.Tensor,
    S: torch.Tensor,
    *,
    num_levels: int,
    euclidean_iters: int = 15,
    reg_lambda: float = 1e-4,
    degenerate_tau: float = 1e-8,
) -> torch.Tensor:
    """Return a dense fake-quantized reconstruction for ``num_levels``."""
    T0, mu0, alpha0 = euclidean_warm_start(
        W, num_levels=num_levels, iters=euclidean_iters
    )
    mu, alpha = manifold_relocation(
        W,
        T0,
        S,
        mu0,
        alpha0,
        reg_lambda=reg_lambda,
        degenerate_tau=degenerate_tau,
    )
    return mu[:, None] + alpha[:, None] * T0


@torch.no_grad()
def activation_weighted_row_error(
    W: torch.Tensor, W_bar: torch.Tensor, S: torch.Tensor
) -> torch.Tensor:
    error = W - W_bar
    return (error * (S @ error.t()).t()).sum(dim=1).clamp_min_(0)
