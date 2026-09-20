"""KOTMS: Kronecker Orthogonal Tri-Modal Shaping.

Faithful reimplementation of TWLA (arXiv:2606.13054v2) Algorithm 1
(Kronecker Dimension Factorization) and Algorithm 2 (KOTMS), replacing the
previous ad-hoc "budget" factor-search rule and asymmetric-sigma/global-
balance loss with the paper's literal factorization rule, per-row TriGMM
shaping objective (Eq. 12), and zero-peak-mass regularizer (Eq. 13, 72).
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch


# ------------------------- Algorithm 1: Kronecker Dimension Factorization -------------------------
def kronecker_factorize(d: int) -> Tuple[int, int]:
    """Algorithm 1. Search downward from floor(sqrt(d)) for the largest
    divisor of d; return (d1, d2) with d1*d2 == d and d1 the complementary
    (larger, or equal) factor.
    """
    assert d >= 1
    a = math.isqrt(d)
    while a > 1 and d % a != 0:
        a -= 1
    d2 = a
    d1 = d // d2
    return d1, d2


# -------------------- Kronecker-structured two-sided orthogonal rotation --------------------
class TwoSidedOrthoSimple(torch.nn.Module):
    """R = R1 (x) R2 via Cayley parameterization (Eq. 14, 16-17):
        A_k = S_k - S_k^T,   R_k = (I + A_k)^-1 (I - A_k),  k in {1,2}.

    Applied to a weight row reshaped to [n1, n2] via two compact
    matrix multiplications (Eq. 15) instead of a dense m x m matrix.
    """

    def __init__(self, n1: int, n2: int, device, dtype):
        super().__init__()
        self.n1, self.n2 = n1, n2
        self.S1 = torch.nn.Parameter(torch.zeros(n1, n1, device=device, dtype=dtype))
        self.S2 = torch.nn.Parameter(torch.zeros(n2, n2, device=device, dtype=dtype))

    @staticmethod
    def _cayley(S: torch.Tensor) -> torch.Tensor:
        A = S - S.transpose(-1, -2)
        I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
        return torch.linalg.solve(I + A, I - A)

    def forward(self, W2d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        oc, ic = W2d.shape
        assert ic == self.n1 * self.n2, f"ic({ic}) must equal n1*n2({self.n1 * self.n2})."

        R1 = self._cayley(self.S1)  # [n1,n1]
        R2 = self._cayley(self.S2)  # [n2,n2]

        W3 = W2d.view(oc, self.n1, self.n2)             # [oc,n1,n2]
        Xr = W3 @ R2                                     # rotate minor axis: [oc,n1,n2]

        Xr_perm = Xr.permute(1, 0, 2).contiguous()       # [n1,oc,n2]
        Xl = (R1 @ Xr_perm.view(self.n1, oc * self.n2)).view(self.n1, oc, self.n2)
        Xl = Xl.permute(1, 0, 2).contiguous()             # rotate major axis: [oc,n1,n2]

        X2d = Xl.view(oc, ic)
        return X2d, R1, R2


def train_kotms(
    W2d: torch.Tensor,
    iters: int = 100,
    eta: float = 1e-2,
    pi0: float = 0.5,
    target_rho: float = 0.5,
    beta: float = 0.1,
    sigma_min: float = 1e-6,
    n1: Optional[int] = None,
    n2: Optional[int] = None,
) -> Dict[str, Any]:
    """Algorithm 2: KOTMS. Optimizes Kronecker orthogonal factors (R1, R2)
    under the TriGMM shaping loss (Eq. 12) plus zero-peak-mass regularizer
    (Eq. 13, 72), via plain gradient descent on the free Cayley generators
    (S1, S2), exactly as specified in the pseudocode.
    """
    device, dtype = W2d.device, W2d.dtype
    oc, ic = W2d.shape

    if n1 is None or n2 is None:
        n1_auto, n2_auto = kronecker_factorize(ic)
        n1 = n1_auto if n1 is None else n1
        n2 = n2_auto if n2 is None else n2

    rot = TwoSidedOrthoSimple(n1=n1, n2=n2, device=device, dtype=dtype)

    pi_plus = (1.0 - pi0) / 2.0
    pi_minus = (1.0 - pi0) / 2.0
    log_pi = torch.log(
        torch.tensor([pi_plus, pi0, pi_minus], device=device, dtype=dtype).clamp_min(1e-12)
    )
    log2pi = math.log(2.0 * math.pi)

    for t in range(iters):
        for p in (rot.S1, rot.S2):
            if p.grad is not None:
                p.grad = None

        Z, R1, R2 = rot(W2d)

        c = Z.abs().mean(dim=1, keepdim=True)                              # [oc,1], Eq 12: c_i = mean_j|z_ij|
        sigma = Z.std(dim=1, keepdim=True).clamp_min(sigma_min)            # [oc,1], sigma_i = max(std(z_i), sigma_min)

        c_full = c.expand_as(Z)
        sigma_full = sigma.expand_as(Z)
        log_sigma2 = torch.log(sigma_full.pow(2))

        logp_plus = -0.5 * (log2pi + log_sigma2 + (Z - c_full).pow(2) / sigma_full.pow(2))
        logp_zero = -0.5 * (log2pi + log_sigma2 + Z.pow(2) / sigma_full.pow(2))
        logp_minus = -0.5 * (log2pi + log_sigma2 + (Z + c_full).pow(2) / sigma_full.pow(2))

        logps = torch.stack([logp_plus, logp_zero, logp_minus], dim=-1)    # [oc,ic,3]
        log_mix = torch.logsumexp(logps + log_pi, dim=-1)                  # log sum_k pi_k phi_k
        L_trigmm = -log_mix.mean()                                          # Eq 12

        resp = torch.softmax(logps + log_pi, dim=-1)                       # posterior responsibilities, Eq 70
        r_zero = resp[..., 1]                                               # [oc,ic]
        rbar_zero = r_zero.mean(dim=1)                                      # [oc], Eq 71
        L_zero = (rbar_zero - target_rho).pow(2).mean()                    # Eq 72 (mean over rows)

        L_shape = L_trigmm + beta * L_zero                                  # Eq 13
        L_shape.backward()

        with torch.no_grad():
            rot.S1 -= eta * rot.S1.grad
            rot.S2 -= eta * rot.S2.grad

    with torch.no_grad():
        X2d, R1, R2 = rot(W2d)

    return {
        "B": R2.detach(),
        "C": R1.detach(),
        "l": n1,
        "r": n2,
        "W_rot": X2d.detach(),
    }


# ------------------------- preprocessor -------------------------
@dataclass
class KroneckerSmoothConfig:
    gmm_iters: int = 100
    gmm_eta: float = 1e-2
    gmm_pi0: float = 0.5
    gmm_rho: float = 0.5
    gmm_beta: float = 0.1
    gmm_sigma_min: float = 1e-6
    l: Optional[int] = None
    r: Optional[int] = None


class KroneckerSmoothPreprocessor:
    def __init__(self, config: KroneckerSmoothConfig):
        self.config = KroneckerSmoothConfig(**config) if isinstance(config, dict) else config

    def _kronecker_process(self, weight: torch.Tensor) -> Dict[str, Any]:
        ori_shape, ori_dtype = weight.shape, weight.dtype
        x = weight.squeeze()

        if x.dim() > 2:
            oc = x.shape[0]
            ic = x.shape[1:].numel()
            W2d = x.reshape(oc, ic).to(ori_dtype)
        else:
            oc, ic = x.shape
            W2d = x.to(ori_dtype)

        out = train_kotms(
            W2d=W2d,
            iters=self.config.gmm_iters,
            eta=self.config.gmm_eta,
            pi0=self.config.gmm_pi0,
            target_rho=self.config.gmm_rho,
            beta=self.config.gmm_beta,
            sigma_min=self.config.gmm_sigma_min,
            n1=self.config.l,
            n2=self.config.r,
        )

        W_rot = out["W_rot"].reshape(ori_shape).to(ori_dtype).contiguous()

        return {
            "B": out["B"].to(ori_dtype),
            "C": out["C"].to(ori_dtype),
            "l": out["l"],
            "r": out["r"],
            "W_rot": W_rot,
        }
