"""Activation-aware asymmetric-codebook E2M-ATQ with codebook-aware KOTMS.

This module is deliberately opt-in.  It does not change the repository's
literal ternary E2M-ATQ path.  For one two-dimensional weight matrix it runs:

  1. row centering/scaling and an activation-weighted histogram;
  2. globally optimal weighted 1-D k-means on that histogram via DP;
  3. a Kronecker orthogonal rotation trained against the fixed DP codebook;
  4. hard code assignment in the rotated coordinates; and
  5. the E2M activation-aware closed-form relocation of row offset/scale.

The returned dense reconstruction is rotated back to the original basis.
It can therefore be written into an ordinary Hugging Face checkpoint for
accuracy evaluation without modifying the model forward.  A genuinely
packed deployment must instead store/use ``codes``, ``codebook``, ``mu``,
``alpha`` and the two rotation factors returned in the result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

try:
    from quantize.k_preprocessor_ternary import TwoSidedOrthoSimple, kronecker_factorize
    from quantize import E2M_ATQ as symmetric_ternary
except ImportError:  # Package import from the repository root.
    from .k_preprocessor_ternary import TwoSidedOrthoSimple, kronecker_factorize
    from . import E2M_ATQ as symmetric_ternary


@dataclass
class AsymmetricCodebookResult:
    reconstructed: torch.Tensor
    codes: torch.Tensor
    codebook: torch.Tensor
    codebook_priors: torch.Tensor
    mu: torch.Tensor
    alpha: torch.Tensor
    rotation_left: torch.Tensor
    rotation_right: torch.Tensor
    stats: dict[str, Any]


def _weighted_interval_sse(
    prefix_mass: np.ndarray,
    prefix_first: np.ndarray,
    prefix_second: np.ndarray,
    starts: np.ndarray,
    end: int,
) -> np.ndarray:
    """Weighted SSE of intervals ``[starts, end)`` in O(len(starts))."""
    mass = prefix_mass[end] - prefix_mass[starts]
    first = prefix_first[end] - prefix_first[starts]
    second = prefix_second[end] - prefix_second[starts]
    return np.maximum(0.0, second - first * first / np.maximum(mass, 1e-300))


def optimal_weighted_scalar_codebook(
    centers: torch.Tensor,
    masses: torch.Tensor,
    num_levels: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Exact weighted 1-D k-means for sorted histogram bins.

    The dynamic program uses the monotone-optimum divide-and-conquer
    optimization.  It is exact for the supplied histogram (the histogram is
    itself an approximation to the original, unbinned values).

    Returns normalized ordered centroids, their cluster priors, and the
    pre-normalization histogram SSE.
    """
    if centers.ndim != 1 or masses.ndim != 1 or centers.numel() != masses.numel():
        raise ValueError("centers and masses must be same-length 1-D tensors")
    if int(num_levels) < 2:
        raise ValueError("num_levels must be at least two")

    x = centers.detach().double().cpu().numpy()
    w = masses.detach().double().cpu().numpy()
    keep = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[keep], w[keep]
    if x.size < int(num_levels):
        raise ValueError(
            f"histogram has {x.size} non-empty bins, fewer than K={num_levels}"
        )
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    n = int(x.size)
    k_max = int(num_levels)

    p0 = np.concatenate(([0.0], np.cumsum(w, dtype=np.float64)))
    p1 = np.concatenate(([0.0], np.cumsum(w * x, dtype=np.float64)))
    p2 = np.concatenate(([0.0], np.cumsum(w * x * x, dtype=np.float64)))
    previous = np.full(n + 1, np.inf, dtype=np.float64)
    previous[0] = 0.0
    back = np.full((k_max + 1, n + 1), -1, dtype=np.int32)

    for k in range(1, k_max + 1):
        current = np.full(n + 1, np.inf, dtype=np.float64)

        def solve(left: int, right: int, opt_left: int, opt_right: int) -> None:
            if left > right:
                return
            middle = (left + right) // 2
            lo = max(k - 1, opt_left)
            hi = min(middle - 1, opt_right)
            candidates = np.arange(lo, hi + 1, dtype=np.int64)
            values = previous[candidates] + _weighted_interval_sse(
                p0, p1, p2, candidates, middle
            )
            best_offset = int(np.argmin(values))
            best = int(candidates[best_offset])
            current[middle] = float(values[best_offset])
            back[k, middle] = best
            solve(left, middle - 1, opt_left, best)
            solve(middle + 1, right, best, opt_right)

        solve(k, n, k - 1, n - 1)
        previous = current

    segments: list[tuple[int, int]] = []
    end = n
    for k in range(k_max, 0, -1):
        start = int(back[k, end])
        if start < 0:
            raise RuntimeError("failed to backtrack 1-D k-means DP")
        segments.append((start, end))
        end = start
    segments.reverse()

    codebook = []
    cluster_mass = []
    for start, end in segments:
        mass = p0[end] - p0[start]
        first = p1[end] - p1[start]
        codebook.append(first / mass)
        cluster_mass.append(mass)
    codebook_np = np.asarray(codebook, dtype=np.float64)
    prior_np = np.asarray(cluster_mass, dtype=np.float64)
    prior_np /= prior_np.sum()

    # Fix the affine gauge between the shared codebook and per-row mu/alpha.
    mean = float(np.sum(prior_np * codebook_np))
    variance = float(np.sum(prior_np * (codebook_np - mean) ** 2))
    if not np.isfinite(variance) or variance <= 1e-20:
        raise FloatingPointError("degenerate codebook variance")
    codebook_np = (codebook_np - mean) / np.sqrt(variance)

    out_device = centers.device
    out_dtype = centers.dtype if centers.is_floating_point() else torch.float32
    return (
        torch.as_tensor(codebook_np, device=out_device, dtype=out_dtype),
        torch.as_tensor(prior_np, device=out_device, dtype=out_dtype),
        float(previous[n]),
    )


@torch.no_grad()
def activation_weighted_histogram(
    weight: torch.Tensor,
    moment: torch.Tensor,
    *,
    bins: int = 1024,
    clip_quantile: float = 1e-4,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool row-normalized weights into a diagonal-moment-weighted histogram.

    A normalized error is multiplied by ``alpha_i`` when mapped back to the
    original weight domain, hence each point has mass
    ``alpha_i**2 * diag(moment)[j]`` rather than merely ``diag(moment)[j]``.
    """
    if weight.ndim != 2:
        raise ValueError("weight must be a 2-D [out_features, in_features] tensor")
    if moment.shape != (weight.shape[1], weight.shape[1]):
        raise ValueError("moment shape must be [in_features, in_features]")
    if int(bins) < 2:
        raise ValueError("bins must be at least two")
    if not 0.0 <= float(clip_quantile) < 0.5:
        raise ValueError("clip_quantile must be in [0, 0.5)")

    w = weight.float()
    mu = w.mean(dim=1)
    centered = w - mu[:, None]
    alpha = centered.square().mean(dim=1).sqrt().clamp_min(eps)
    normalized = centered / alpha[:, None]
    flat = normalized.reshape(-1)
    if clip_quantile > 0:
        bounds = torch.quantile(
            flat,
            torch.tensor(
                [clip_quantile, 1.0 - clip_quantile],
                device=flat.device,
                dtype=flat.dtype,
            ),
        )
        lower, upper = bounds[0], bounds[1]
    else:
        lower, upper = flat.min(), flat.max()
    if not bool(torch.isfinite(lower) & torch.isfinite(upper)):
        raise FloatingPointError("non-finite normalized-weight histogram range")
    if float((upper - lower).abs()) <= eps:
        lower, upper = lower - 1.0, upper + 1.0

    diagonal = torch.diagonal(moment.float()).clamp_min(0)
    if not bool((diagonal > 0).any()):
        diagonal = torch.ones_like(diagonal)
    point_mass = alpha.square()[:, None] * diagonal[None, :]
    # A common positive rescaling leaves the DP partition unchanged and keeps
    # its prefix sums in a numerically friendly range.
    point_mass = point_mass / point_mass.mean().clamp_min(eps)

    clamped = flat.clamp(lower, upper)
    width = (upper - lower) / float(bins)
    indices = torch.floor((clamped - lower) / width).long().clamp_(0, bins - 1)
    mass = torch.zeros(int(bins), device=w.device, dtype=torch.float64)
    mass.scatter_add_(0, indices, point_mass.reshape(-1).double())
    bin_centers = lower.double() + (torch.arange(bins, device=w.device, dtype=torch.float64) + 0.5) * width.double()
    keep = mass > 0
    return bin_centers[keep].float(), mass[keep].float(), mu, alpha


def _rotation_loss(
    rotated_weight: torch.Tensor,
    codebook: torch.Tensor,
    priors: torch.Tensor,
    peak_sigma: float,
    occupancy_beta: float,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    row_mu = rotated_weight.mean(dim=1, keepdim=True)
    row_scale = rotated_weight.std(dim=1, unbiased=False, keepdim=True).clamp_min(eps)
    normalized = (rotated_weight - row_mu) / row_scale
    squared_distance = (normalized.unsqueeze(-1) - codebook.view(1, 1, -1)).square()
    logits = priors.clamp_min(eps).log().view(1, 1, -1) - 0.5 * squared_distance / (
        float(peak_sigma) ** 2
    )
    nll = -torch.logsumexp(logits, dim=-1).mean()
    responsibilities = torch.softmax(logits, dim=-1)
    row_occupancy = responsibilities.mean(dim=1)
    occupancy = (
        (row_occupancy - priors.view(1, -1)).square()
        / priors.clamp_min(eps).view(1, -1)
    ).sum(dim=1).mean()
    return nll + float(occupancy_beta) * occupancy, occupancy


def train_fixed_codebook_rotation(
    weight: torch.Tensor,
    codebook: torch.Tensor,
    priors: torch.Tensor,
    *,
    iters: int = 30,
    learning_rate: float = 1e-2,
    peak_sigma: float = 0.25,
    occupancy_beta: float = 0.05,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Fit one Kronecker rotation to a fixed asymmetric codebook."""
    if weight.ndim != 2:
        raise ValueError("weight must be two-dimensional")
    if codebook.ndim != 1 or priors.shape != codebook.shape:
        raise ValueError("codebook and priors must be same-length 1-D tensors")
    if iters < 0:
        raise ValueError("iters must be non-negative")
    if peak_sigma <= 0:
        raise ValueError("peak_sigma must be positive")

    n1, n2 = kronecker_factorize(int(weight.shape[1]))
    rotation = TwoSidedOrthoSimple(
        n1=n1,
        n2=n2,
        device=weight.device,
        dtype=torch.float32,
    )
    source = weight.detach().float()
    codebook = codebook.detach().to(device=source.device, dtype=source.dtype)
    priors = priors.detach().to(device=source.device, dtype=source.dtype)
    optimizer = torch.optim.Adam(rotation.parameters(), lr=float(learning_rate))
    initial_loss = final_loss = final_occupancy = float("nan")

    for step in range(int(iters)):
        optimizer.zero_grad(set_to_none=True)
        rotated, _left, _right = rotation(source)
        loss, occupancy = _rotation_loss(
            rotated,
            codebook,
            priors,
            peak_sigma,
            occupancy_beta,
            eps,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite codebook-aware KOTMS loss at step {step}")
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        final_occupancy = float(occupancy.detach().cpu())

    with torch.no_grad():
        _rotated, left, right = rotation(source)
        if iters == 0:
            loss, occupancy = _rotation_loss(
                _rotated, codebook, priors, peak_sigma, occupancy_beta, eps
            )
            initial_loss = final_loss = float(loss.cpu())
            final_occupancy = float(occupancy.cpu())
    return left.detach(), right.detach(), {
        "initial_rotation_loss": initial_loss,
        "final_rotation_loss": final_loss,
        "final_occupancy_loss": final_occupancy,
        "rotation_iters": int(iters),
        "kronecker_left_dim": int(n1),
        "kronecker_right_dim": int(n2),
    }


def rotate_rows(weight: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    shape = weight.shape
    reshaped = weight.reshape(shape[0], left.shape[0], right.shape[0])
    return torch.matmul(left, torch.matmul(reshaped, right)).reshape(shape)


def inverse_rotate_rows(
    rotated: torch.Tensor, left: torch.Tensor, right: torch.Tensor
) -> torch.Tensor:
    shape = rotated.shape
    reshaped = rotated.reshape(shape[0], left.shape[0], right.shape[0])
    return torch.matmul(left.transpose(0, 1), torch.matmul(reshaped, right.transpose(0, 1))).reshape(shape)


@torch.no_grad()
def decode_asymmetric_representation(
    codes: torch.Tensor,
    codebook: torch.Tensor,
    mu: torch.Tensor,
    alpha: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Decode one expert or a batch of experts back to the native basis.

    Single-expert shapes are ``codes=[out,in]``, ``codebook=[K]``,
    ``mu/alpha=[out]`` and two 2-D rotation factors.  Batched bank slices add
    a leading expert dimension to every tensor.
    """
    single = codes.ndim == 2
    if single:
        codes = codes.unsqueeze(0)
        codebook = codebook.unsqueeze(0)
        mu = mu.unsqueeze(0)
        alpha = alpha.unsqueeze(0)
        left = left.unsqueeze(0)
        right = right.unsqueeze(0)
    if codes.ndim != 3:
        raise ValueError("codes must have shape [out,in] or [experts,out,in]")
    experts, rows, width = codes.shape
    if codebook.ndim != 2 or codebook.shape[0] != experts:
        raise ValueError("batched codebook shape must be [experts,K]")
    flat_codes = codes.long().reshape(experts, -1)
    levels = torch.gather(codebook, 1, flat_codes).reshape(experts, rows, width)
    rotated = mu[:, :, None] + alpha[:, :, None] * levels
    n1, n2 = int(left.shape[-1]), int(right.shape[-1])
    if n1 * n2 != width:
        raise ValueError("rotation factor dimensions do not match weight width")
    matrices = rotated.reshape(experts, rows, n1, n2)
    native = torch.matmul(left.transpose(-1, -2)[:, None], matrices)
    native = torch.matmul(native, right.transpose(-1, -2)[:, None])
    native = native.reshape(experts, rows, width)
    return native[0] if single else native


@torch.no_grad()
def euclidean_code_assignment(
    rotated_weight: torch.Tensor,
    codebook: torch.Tensor,
    *,
    iters: int = 5,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coordinate descent for hard codes and row affine parameters."""
    mu = rotated_weight.mean(dim=1)
    alpha = (rotated_weight - mu[:, None]).square().mean(dim=1).sqrt().clamp_min(eps)
    codebook = codebook.to(device=rotated_weight.device, dtype=rotated_weight.dtype)
    codes = torch.zeros_like(rotated_weight, dtype=torch.long)
    levels = torch.zeros_like(rotated_weight)
    for _ in range(max(1, int(iters))):
        normalized = (rotated_weight - mu[:, None]) / alpha[:, None]
        codes = (normalized.unsqueeze(-1) - codebook.view(1, 1, -1)).abs().argmin(dim=-1)
        levels = codebook[codes]
        level_mean = levels.mean(dim=1)
        weight_mean = rotated_weight.mean(dim=1)
        centered_level = levels - level_mean[:, None]
        denominator = centered_level.square().sum(dim=1).clamp_min(eps)
        alpha = (
            (centered_level * (rotated_weight - weight_mean[:, None])).sum(dim=1)
            / denominator
        ).clamp_min(eps)
        mu = weight_mean - alpha * level_mean
    normalized = (rotated_weight - mu[:, None]) / alpha[:, None]
    codes = (normalized.unsqueeze(-1) - codebook.view(1, 1, -1)).abs().argmin(dim=-1)
    levels = codebook[codes]
    return codes, levels, mu, alpha


@torch.no_grad()
def activation_aware_relocation_with_rotation(
    original_weight: torch.Tensor,
    rotated_levels: torch.Tensor,
    moment: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    mu0: torch.Tensor,
    alpha0: torch.Tensor,
    *,
    reg_lambda: float = 1e-4,
    degenerate_tau: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact 2x2 E2M relocation while evaluating error in original basis.

    A packed implementation would compute ``mu*1 + alpha*T`` in the rotated
    coordinates and rotate the activation at runtime.  For a normal dense
    HF checkpoint we inverse-rotate the two reconstruction bases here.  The
    resulting output is algebraically identical and needs no forward patch.
    """
    width = int(original_weight.shape[1])
    epsilon = reg_lambda * torch.diagonal(moment).mean().clamp_min(1e-20)
    regularized = moment + epsilon * torch.eye(
        width, device=moment.device, dtype=moment.dtype
    )
    level_basis = inverse_rotate_rows(rotated_levels, left, right)
    offset_basis = inverse_rotate_rows(
        torch.ones((1, width), device=original_weight.device, dtype=original_weight.dtype),
        left,
        right,
    )[0]

    s_level = regularized @ level_basis.t()
    s_offset = regularized @ offset_basis
    s_weight = regularized @ original_weight.t()
    a = (level_basis * s_level.t()).sum(dim=1)
    b = (level_basis * s_offset[None, :]).sum(dim=1)
    c = (offset_basis * s_offset).sum()
    d = (level_basis * s_weight.t()).sum(dim=1)
    e = (original_weight * s_offset[None, :]).sum(dim=1)
    determinant = a * c - b.square()
    safe = determinant > degenerate_tau * (a * c).clamp_min(0)
    denominator = torch.where(safe, determinant, torch.ones_like(determinant))
    alpha = torch.where(safe, (d * c - b * e) / denominator, alpha0)
    mu = torch.where(safe, (a * e - b * d) / denominator, mu0)
    reconstructed = mu[:, None] * offset_basis[None, :] + alpha[:, None] * level_basis
    return reconstructed, mu, alpha


def asymmetric_codebook_quantize(
    weight: torch.Tensor,
    moment: torch.Tensor,
    *,
    num_levels: int,
    histogram_bins: int = 1024,
    histogram_clip_quantile: float = 1e-4,
    rotation_iters: int = 30,
    rotation_learning_rate: float = 1e-2,
    rotation_peak_sigma: float = 0.25,
    rotation_occupancy_beta: float = 0.05,
    assignment_iters: int = 5,
    ternary_euclidean_iters: int = 15,
    reg_lambda: float = 1e-4,
    degenerate_tau: float = 1e-8,
) -> AsymmetricCodebookResult:
    """Run symmetric TWLA at K=3 and the asymmetric extension at K>=4."""
    source = weight.float()
    metric = moment.float()
    if int(num_levels) == 3:
        levels, mu0, alpha0 = symmetric_ternary.euclidean_warm_start(
            source, iters=ternary_euclidean_iters
        )
        mu, alpha = symmetric_ternary.manifold_relocation(
            source,
            levels,
            metric,
            mu0,
            alpha0,
            reg_lambda=reg_lambda,
            degenerate_tau=degenerate_tau,
        )
        reconstructed = mu[:, None] + alpha[:, None] * levels
        codes = (levels + 1.0).round().to(torch.uint8)
        codebook = torch.tensor(
            [-1.0, 0.0, 1.0], device=source.device, dtype=source.dtype
        )
        priors = torch.bincount(codes.reshape(-1).long(), minlength=3).to(source.dtype)
        priors = priors / priors.sum().clamp_min(1.0)
        n1, n2 = kronecker_factorize(int(source.shape[1]))
        left = torch.eye(n1, device=source.device, dtype=source.dtype)
        right = torch.eye(n2, device=source.device, dtype=source.dtype)
        error = source - reconstructed
        weighted_error = float(
            (error * (metric @ error.t()).t()).sum().double().cpu()
        )
        squared_error = float(error.square().sum().double().cpu())
        return AsymmetricCodebookResult(
            reconstructed=reconstructed,
            codes=codes,
            codebook=codebook,
            codebook_priors=priors,
            mu=mu,
            alpha=alpha,
            rotation_left=left,
            rotation_right=right,
            stats={
                "num_levels": 3,
                "quantization_scheme": "original_symmetric_twla_ternary",
                "ternary_euclidean_iters": int(ternary_euclidean_iters),
                "rotation_iters": 0,
                "kronecker_left_dim": int(n1),
                "kronecker_right_dim": int(n2),
                "weight_squared_error": squared_error,
                "activation_weighted_error": weighted_error,
            },
        )

    centers, masses, _hist_mu, _hist_alpha = activation_weighted_histogram(
        source,
        metric,
        bins=histogram_bins,
        clip_quantile=histogram_clip_quantile,
    )
    codebook, priors, histogram_sse = optimal_weighted_scalar_codebook(
        centers, masses, num_levels
    )
    left, right, rotation_stats = train_fixed_codebook_rotation(
        source,
        codebook,
        priors,
        iters=rotation_iters,
        learning_rate=rotation_learning_rate,
        peak_sigma=rotation_peak_sigma,
        occupancy_beta=rotation_occupancy_beta,
    )
    with torch.no_grad():
        rotated = rotate_rows(source, left, right)
        codes, levels, mu0, alpha0 = euclidean_code_assignment(
            rotated, codebook, iters=assignment_iters
        )
        reconstructed, mu, alpha = activation_aware_relocation_with_rotation(
            source,
            levels,
            metric,
            left,
            right,
            mu0,
            alpha0,
            reg_lambda=reg_lambda,
            degenerate_tau=degenerate_tau,
        )
        error = source - reconstructed
        weighted_error = float(
            (error * (metric @ error.t()).t()).sum().double().cpu()
        )
        squared_error = float(error.square().sum().double().cpu())
    return AsymmetricCodebookResult(
        reconstructed=reconstructed,
        codes=codes.to(torch.uint8),
        codebook=codebook,
        codebook_priors=priors,
        mu=mu,
        alpha=alpha,
        rotation_left=left,
        rotation_right=right,
        stats={
            "num_levels": int(num_levels),
            "histogram_bins": int(histogram_bins),
            "histogram_nonempty_bins": int(centers.numel()),
            "histogram_dp_sse": float(histogram_sse),
            "weight_squared_error": squared_error,
            "activation_weighted_error": weighted_error,
            **rotation_stats,
        },
    )
