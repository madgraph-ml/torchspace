"""Implement Rational Quadratic splines.
Based on the pytorch implementation of
https://github.com/bayesiains/nsf"""

import math

import torch
import torch.nn.functional as F


def searchsorted(
    bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs: torch.Tensor,
    theta: torch.Tensor,
    rev: bool,
    num_bins: int,
    left: float,
    right: float,
    bottom: float,
    top: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float,
    periodic: bool = False,
    sum_jacobian: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transform inputs using RQ splines defined by theta.

    Args:
        inputs: Input tensor
        theta: tensor with bin widths, heights and derivatives
        rev: If True, compute inverse transformation

    Returns:
        Transformed tensor and log of jacobian determinant
    """
    if not rev:
        inside_interval_mask = torch.all((inputs >= left) & (inputs <= right), dim=-1)
    else:
        inside_interval_mask = torch.all((inputs >= bottom) & (inputs <= top), dim=-1)
    outside_interval_mask = ~inside_interval_mask
    masked_outputs = torch.zeros_like(inputs)
    masked_logabsdet = torch.zeros(
        inputs.shape[: (1 if sum_jacobian else 2)],
        dtype=inputs.dtype,
        device=inputs.device,
    )
    masked_outputs[outside_interval_mask] = inputs[outside_interval_mask]
    masked_logabsdet[outside_interval_mask] = 0

    inputs = inputs[inside_interval_mask]
    theta = theta[inside_interval_mask, :]

    unnormalized_widths = theta[..., :num_bins]
    unnormalized_heights = theta[..., num_bins : num_bins * 2]
    unnormalized_derivatives = theta[..., num_bins * 2 :]

    # unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    # constant = np.log(np.exp(1 - min_derivative) - 1)
    # unnormalized_derivatives[..., 0] = constant
    # unnormalized_derivatives[..., -1] = constant

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = (min_derivative + F.softplus(unnormalized_derivatives)) / (
        min_derivative + math.log(2)
    )
    if periodic:
        derivatives[..., -1] = derivatives[..., 0]
        periodic_shift = (
            (right - left) / 2 * torch.tanh(unnormalized_derivatives[..., -1])
        )

        if not rev:
            inputs = (
                torch.remainder(inputs + periodic_shift - left, right - left) + left
            )
            infi = inputs[(inputs < left) | (inputs > right) | ~torch.isfinite(inputs)]
            if len(infi) > 0:
                print(infi)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if rev:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if rev:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (torch.isnan(discriminant) | (discriminant >= 0)).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths
        if periodic:
            outputs = (
                torch.remainder(outputs - periodic_shift - left, right - left) + left
            )

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = -torch.log(derivative_numerator) + 2 * torch.log(denominator)

    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

    if sum_jacobian:
        logabsdet = torch.sum(logabsdet, dim=1)

    masked_outputs[inside_interval_mask] = outputs
    masked_logabsdet[inside_interval_mask] = logabsdet

    return masked_outputs, masked_logabsdet
