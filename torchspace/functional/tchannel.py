"""Implement helper functions needed for the t-channel mappings defined in
https://arxiv.org/pdf/hep-ph/0008033
"""

from typing import Tuple

import torch
from torch import Tensor, sqrt

from .kinematics import EPS, kaellen, lsquare, rotxxx
from .ps_utils import tinv_two_particle_density


def tminmax(
    s: Tensor,
    p1_2: Tensor,
    p2_2: Tensor,
    m1: Tensor,
    m2: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Mandelstam invariant t=(p1-k1)^2 formula (C.21) in https://arxiv.org/pdf/hep-ph/0008033.pdf
    p=p1+p2 is at rest;
    p1, p2 are opposite along z-axis
    k1, k4 are opposite along the direction defined by theta
    theta is the angle in the COM frame between p1 & k1
    """
    num1 = (s + m1**2 - m2**2) * (s + p1_2 - p2_2)
    num2 = sqrt(kaellen(s, m1**2, m2**2)) * sqrt(kaellen(s, p1_2, p2_2))
    tmax = m1**2 + p1_2 - (num1 - num2) / torch.clip(2 * s, min=EPS)
    tmin = m1**2 + p1_2 - (num1 + num2) / torch.clip(2 * s, min=EPS)
    return tmin, tmax


def costheta_to_invt(
    s: Tensor,
    p1_2: Tensor,
    p2_2: Tensor,
    m1: Tensor,
    m2: Tensor,
    costheta: Tensor,
) -> Tensor:
    """
    Mandelstam invariant t=(p1-k1)^2 formula (C.21) in https://arxiv.org/pdf/hep-ph/0008033.pdf
    p=p1+p2 is at rest;
    p1, p2 are opposite along z-axis
    k1, k4 are opposite along the direction defined by theta
    theta is the angle in the COM frame between p1 & k1
    """
    num1 = (s + m1**2 - m2**2) * (s + p1_2 - p2_2)
    num2 = sqrt(kaellen(s, m1**2, m2**2)) * sqrt(kaellen(s, p1_2, p2_2)) * costheta
    num = num1 - num2
    t = m1**2 + p1_2 - num / torch.clip(2 * s, min=EPS)
    return torch.clamp_max_(t, -EPS)


def invt_to_costheta(
    s: Tensor,
    p1_2: Tensor,
    p2_2: Tensor,
    m1: Tensor,
    m2: Tensor,
    t: Tensor,
) -> Tensor:
    """
    https://arxiv.org/pdf/hep-ph/0008033.pdf Eq.(C.21)
    invert t=(p1-k1)^2 to cos_theta = ...
    """
    num1 = (t - m1**2 - p1_2) * 2 * s
    num2 = (s + m1**2 - m2**2) * (s + p1_2 - p2_2)
    num = num1 + num2
    denom = sqrt(kaellen(s, m1**2, m2**2)) * sqrt(kaellen(s, p1_2, p2_2))
    costheta = num / torch.clip(denom, min=EPS)
    return torch.clamp_(costheta, -1.0, 1.0)


def gen_mom_from_t_and_phi_com(
    p1: Tensor,
    p2: Tensor,
    t: Tensor,
    phi: Tensor,
    m1: Tensor,
    m2: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Generate 2->2 momenta in the COM frame of p1+p2, given
    the Mandelstam invariant t=(p1-k1)^2 and azimuthal angle phi
    for k1.

    Inputs:
        p1, p2 = incoming momenta (batch_size,4)
        t = Mandelstam invariant t=(p1-k1)^2
        phi = azimuthal angle of k3 (around p1 direction)
        m1, m2 = (virtual) masses of outgoing particles
    Outputs:
        k1, k2 = outgoing momenta (batch_size,4)
    """
    # get invariants and incoming virtualities
    p1_2 = lsquare(p1)
    p2_2 = lsquare(p2)
    ptot = p1 + p2
    s = lsquare(ptot)

    # Map from t to cos_theta (needs -t as it was sampled as |t|)
    costheta = invt_to_costheta(s, p1_2, p2_2, m1, m2, -t)
    sintheta = sqrt(1 - costheta**2)

    # Define the momenta (in COM frame of decaying particle)
    k1 = torch.zeros_like(ptot)
    k1mag = sqrt(kaellen(s, m1**2, m2**2)) / (2 * sqrt(s))
    k1[:, 0] = (s + m1**2 - m2**2) / (2 * sqrt(s))
    k1[:, 1] = k1mag * sintheta * torch.cos(phi)
    k1[:, 2] = k1mag * sintheta * torch.sin(phi)
    k1[:, 3] = k1mag * costheta

    # Then rotate into p1-plane)
    k1_com = rotxxx(k1, p1)

    # get the density and decay momenta
    det_two_particle_inv = tinv_two_particle_density(s, p1_2, p2_2)

    return k1_com, det_two_particle_inv
