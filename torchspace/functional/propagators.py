"""Implement propagator mappings.

Based on the mappings described in
[1] https://arxiv.org/abs/hep-ph/0206070v2

and described more precisely in
[2] https://arxiv.org/abs/hep-ph/0008033
[3] https://freidok.uni-freiburg.de/data/154629
"""

from typing import Tuple

import torch
from torch import Tensor, atan, log, tan

from .kinematics import EPS


def uniform_propagator(
    r_or_s: Tensor,
    s_min: Tensor,
    s_max: Tensor,
    inverse: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Implements uniform sampling of invariants as described
    in Eq.(2.2.7) in [2]

    """
    # define density
    gs = (s_max - s_min).clip(min=EPS)

    if inverse:
        # s needs to be within [s_min, s_max]
        r = (r_or_s - s_min) / gs
        return r, 1 / gs.reshape(-1)

    s = s_min + gs * r_or_s
    return s, gs.reshape(-1)


def breit_wigner_propagator(
    r_or_s: Tensor,
    mass: Tensor,
    width: Tensor,
    s_min: Tensor,
    s_max: Tensor,
    inverse: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Performs the Breit-Wigner mapping as described
    in (C.6) & (C.7) in [2]
    """
    # define common variables
    m2 = mass**2
    gm = mass * width
    y1 = atan((s_min - m2) / (gm))
    y2 = atan((s_max - m2) / (gm))
    dy21 = y2 - y1

    if inverse:
        r = (atan((r_or_s - m2) / (gm)) - y1) / dy21
        gs = gm / (dy21 * ((r_or_s - m2) ** 2 + gm**2))
        return r, gs.reshape(-1)

    s = gm * tan(y1 + dy21 * r_or_s) + m2
    gs = gm / (dy21 * ((s - m2) ** 2 + gm**2))
    return s, 1 / gs.reshape(-1)


def stable_propagator(
    r_or_s: Tensor,
    mass: Tensor,
    s_min: Tensor,
    s_max: Tensor,
    nu: float = 1.0,
    inverse: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Performs mapping for vanishing widths propagators
    \propto 1/(s-m)^2 described in (C.5) in [2].
    """
    # define common variables
    del nu
    m2 = mass**2
    q_max = s_max - m2
    q_min = s_min - m2

    if inverse:
        r = log((r_or_s - m2) / q_min) / log(q_max / q_min)
        gsm1 = (r_or_s - m2) * (log(q_max) - log(q_min))
        return r, 1 / gsm1.reshape(-1)

    s = q_max**r_or_s * q_min ** (1 - r_or_s) + m2
    gsm1 = (s - m2) * (log(q_max) - log(q_min))
    return s, gsm1.reshape(-1)


def stable_propagator_nu(
    r_or_s: Tensor,
    mass: Tensor,
    s_min: Tensor,
    s_max: Tensor,
    nu: float = 1.4,
    inverse: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Performs mapping for vanishing widths propagators
    \propto 1/(s-m)^2 described in (C.4) in [2].
    """
    # define common variables
    m2 = mass**2
    q_max = s_max - m2
    q_min = s_min - m2
    power = 1.0 - nu
    qmaxpow = q_max**power
    qminpow = q_min**power

    if inverse:
        spow = (r_or_s - m2) ** power
        r = (spow - qminpow) / (qmaxpow - qminpow)
        gs = power / ((qmaxpow - qminpow) * (r_or_s - m2) ** nu)
        return r, gs.reshape(-1)

    s = (r_or_s * qmaxpow + (1 - r_or_s) * qminpow) ** (1 / power) + m2
    gs = power / ((qmaxpow - qminpow) * (s - m2) ** nu)
    return s, 1 / gs.reshape(-1)


def massles_propagator(
    r_or_s: Tensor,
    s_min: Tensor,
    s_max: Tensor,
    nu: float = 1.0,
    inverse: bool = False,
) -> Tuple[Tensor, Tensor]:
    # define common variables
    r"""Performs mapping for massless propagators
    \propto 1/s^2 described in (C.5) in [2].
    Uses a negative m^2 = -a to avoid instabilities
    when s_min = 0. , with a = -1e-8 as mentioned in [3]
    """
    del nu
    m2 = torch.where(s_min == 0, -1e-8, 0.0)
    q_max = s_max - m2
    q_min = s_min - m2

    if inverse:
        r = log((r_or_s - m2) / q_min) / log(q_max / q_min)
        gsm1 = (r_or_s - m2) * (log(q_max) - log(q_min))
        return r, 1 / gsm1.reshape(-1)

    s = q_max**r_or_s * q_min ** (1 - r_or_s) + m2
    gsm1 = (s - m2) * (log(q_max) - log(q_min))
    return s, gsm1.reshape(-1)


def massless_propagator_nu(
    r_or_s: Tensor,
    s_min: Tensor,
    s_max: Tensor,
    nu: float = 1.4,
    inverse: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Performs mapping for massless propagators
    \propto 1/s^2 described in (C.4) in [2].
    Uses a negative m^2 = -a to avoid instabilities
    when s_min = 0. , with a = -1e-8 as mentioned in [3]
    """
    # define common variables
    m2 = torch.where(s_min == 0, -1e-8, 0.0)
    q_max = s_max - m2
    q_min = s_min - m2
    power = 1.0 - nu
    qmaxpow = q_max**power
    qminpow = q_min**power

    if inverse:
        spow = (r_or_s - m2) ** power
        r = (spow - qminpow) / (qmaxpow - qminpow)
        gs = power / ((qmaxpow - qminpow) * (r_or_s - m2) ** nu)
        return r, gs.reshape(-1)

    s = (r_or_s * qmaxpow + (1 - r_or_s) * qminpow) ** (1 / power) + m2
    gs = power / ((qmaxpow - qminpow) * (s - m2) ** nu)
    return s, 1 / gs.reshape(-1)
