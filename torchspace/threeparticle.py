"""Implement three-particle mappings.
Bases on the mappings described in
https://freidok.uni-freiburg.de/data/154629"""

from typing import Optional

import torch
from torch import Tensor, atan2, sqrt

from .base import PhaseSpaceMapping, TensorList
from .functional.kinematics import (
    boost,
    edot,
    inv_rotate_zy,
    lsquare,
    mass,
    pi,
    rotate_zy,
)
from .functional.ps_utils import three_particle_density
from .functional.tchannel import costheta_to_invt, gen_mom_from_t_and_phi_com, tminmax
from .invariants import (
    BreitWignerInvariantBlock,
    MasslessInvariantBlock,
    StableInvariantBlock,
    UniformInvariantBlock,
)


class ThreeBodyDecayCOM(PhaseSpaceMapping):
    """
    Implement isotropic 3-body phase-space, based on the mapping described in
        [1] https://freidok.uni-freiburg.de/data/154629

    This is expressed in the COM-frame and thus only requires the COM energy
    and the masses (or virtual ones) to construct the final decay momenta.
    Parametrizes the 3-body kinematics via two energies and three angles.
    """

    def __init__(self):
        dims_in = [(5,), (), (3,)]
        dims_out = [(3, 4)]
        super().__init__(dims_in, dims_out)

        self.e1_map = UniformInvariantBlock()
        self.e2_map = UniformInvariantBlock()

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs (TensorList): list of two tensors [r, s, m_out]
                r: random numbers with shape=(b,5)
                s: squared COM energy with shape=(b,)
                m_out: (virtual) masses of outgoing particles
                    with shape=(b,3)

        Returns:
            p_decay (Tensor): decay momenta (COM frame) with shape=(b,3,4)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r, s, m_out = inputs[0], inputs[1], inputs[2]
        p1 = torch.zeros(r.shape[0], 4, device=r.device)
        p2 = torch.zeros_like(p1)
        p3 = torch.empty_like(p1)

        # Maybe remove after debugging
        with torch.no_grad():
            if torch.any(s < 0):
                raise ValueError(f"s needs to be always positive")

        # Get virutalities/masses
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]
        m3 = m_out[:, 2]
        m1sq = m1**2
        m2sq = m2**2
        m3sq = m3**2

        # Define the energy p10
        E1a = sqrt(s) / 2 + (m1sq - (m2 + m3) ** 2) / (2 * sqrt(s))
        (p10,), p1_det = self.e1_map.map([r[:, 0]], [m1, E1a])

        # Define the energy p20
        Delta = 2 * sqrt(s) * (sqrt(s) / 2 - p10) + m1sq
        Delta_23 = m2sq - m3sq
        dE2 = (p10**2 - m1sq) * ((Delta + Delta_23) ** 2 - 4 * m2sq * Delta)
        E2a = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        E2b = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) + sqrt(dE2))
        (p20,), p2_det = self.e2_map.map([r[:, 1]], [E2a, E2b])

        # Define angles
        phi = 2 * pi * r[:, 2]
        costheta = 2 * r[:, 3] - 1
        beta = 2 * pi * r[:, 4]
        det_omega = 8 * pi**2

        # calculate cosalpha
        num_alpha_1 = 2 * sqrt(s) * (sqrt(s) / 2 - p10 - p20)
        num_alpha_2 = m1sq + m2sq + 2 * p10 * p20 - m3sq
        denom_alpha = 2 * sqrt(p10**2 - m1sq) * sqrt(p20**2 - m2sq)
        cosalpha = (num_alpha_1 + num_alpha_2) / denom_alpha

        # Fill momenta
        p1[:, 0] = p10
        p1[:, 3] = sqrt(p10**2 - m1sq)
        p2[:, 0] = p20
        p2[:, 3] = sqrt(p20**2 - m2sq)

        # Do rotations
        p1 = rotate_zy(p1, phi, costheta)
        # Double rotate p2
        p2 = rotate_zy(p2, beta, cosalpha)
        p2 = rotate_zy(p2, phi, costheta)

        # Get final momentum
        p3[:, 0] = sqrt(s) - p10 - p20
        p3[:, 1:] = -p1[:, 1:] - p2[:, 1:]

        # get the density and decay momenta
        # (C.10) in [2] == (C.8)/(4PI)
        gs = three_particle_density() * p1_det * p2_det * det_omega
        p_decay = torch.stack([p1, p2, p3], dim=1)

        return (p_decay,), gs

    def map_inverse(self, inputs: TensorList, condition=None):
        """Inverse map from decay momenta onto random numbers

        Args:
            inputs (TensorList): list with only one tensor [p_decay]
                p_decay: decay momenta (COM frame) with shape=(b,3,4)

        Returns:
            r (Tensor): random numbers with shape=(b,5)
            s (Tensor): squared COM energy with shape=(b,)
            m_out: (virtual) masses of outgoing particles
                    with shape=(b,3)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        p_decay = inputs[0]
        # Decaying particle in lab-frame
        p0 = p_decay.sum(dim=1)
        s = lsquare(p0)
        m_out = mass(p_decay)

        # particle features
        p1 = p_decay[:, 0]
        p2 = p_decay[:, 1]
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]
        m3 = m_out[:, 2]
        m1sq = m1**2
        m2sq = m2**2
        m3sq = m3**2

        # Define the energy p10
        p10 = p1[:, 0]
        E1a = sqrt(s) / 2 + (m1sq - (m2 + m3) ** 2) / (2 * sqrt(s))
        (r_p1,), p1_det_inv = self.e1_map.map([p10], [m1, E1a])

        # Define the energy p10
        p20 = p2[:, 0]
        Delta = 2 * sqrt(s) * (sqrt(s) / 2 - p10) + m1sq
        Delta_23 = m2sq - m3sq
        dE2 = (p10**2 - m1sq) * ((Delta + Delta_23) ** 2 - 4 * m2sq * Delta)
        E2a = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        E2b = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        (r_p2,), p2_det_inv = self.e2_map.map([p20], [E2a, E2b])

        # get p1/2 absolute momentum
        p1mag = sqrt(edot(p1[:, 1:], p1[:, 1:]))

        # Extract phi and theta
        costheta = p1[:, 3] / p1mag
        phi = atan2(p1[:, 2], p1[:, 1])

        # Get the random numbers
        r_phi = phi / (2 * pi) + (phi < 0)
        r_theta = (costheta + 1) / 2

        # Get last angle
        p2 = inv_rotate_zy(p2, phi, costheta)
        beta = atan2(p2[:, 2], p2[:, 1])
        r_beta = beta / (2 * pi) + (beta < 0)
        det_omega_inv = 1 / (8 * pi**2)

        # Pack all together and get full density
        gs = p1_det_inv * p2_det_inv * det_omega_inv / three_particle_density()
        r = torch.stack([r_p1, r_p2, r_phi, r_theta, r_beta], dim=1)

        return (r, s, m_out), gs

    def density(self, inputs: TensorList, condition=None, inverse=False):
        """Returns the density only of the mapping"""
        if inverse:
            _, density = self.map_inverse(inputs, condition)
            return density
        _, density = self.map(inputs, condition)
        return density


class ThreeBodyDecayLAB(PhaseSpaceMapping):
    """
    Implement isotropic 3-body phase-space, based on the mapping described in
        [1] https://freidok.uni-freiburg.de/data/154629

    This is expressed in the LAB-frame and thus requires the input momentum
    in the lab frame. It also requires the masses (or virtual ones) to construct the final decay momenta.
    Parametrizes the 3-body kinematics via two energies and three angles.
    """

    def __init__(self):
        dims_in = [(5,), (4,), (3,)]
        dims_out = [(3, 4), (2,)]
        super().__init__(dims_in, dims_out)

        self.e1_map = UniformInvariantBlock()
        self.e2_map = UniformInvariantBlock()

    def map(self, inputs: TensorList, condition: TensorList):
        """Map from random numbers to momenta

        Args:
            inputs (TensorList): list of two tensors [r, p0, m_out]
                r: random numbers with shape=(b,5)
                p0: incoming momentum in lab frame with shape=(b,4)
                m_out: (virtual) masses of outgoing particles
                    with shape=(b,3)

        Returns:
            p_lab (Tensor): decay momenta (lab frame) with shape=(b,3,4)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r, p0, m_out = inputs[0], inputs[1], inputs[2]
        p1 = torch.zeros(r.shape[0], 4, device=r.device)
        p2 = torch.zeros_like(p1)
        p3 = torch.empty_like(p1)
        s = lsquare(p0)

        # Maybe remove after debugging
        with torch.no_grad():
            if torch.any(s < 0):
                raise ValueError(f"s needs to be always positive")

        # Get virutalities/masses
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]
        m3 = m_out[:, 2]
        m1sq = m1**2
        m2sq = m2**2
        m3sq = m3**2

        # Define the energy p10
        E1a = sqrt(s) / 2 + (m1sq - (m2 + m3) ** 2) / (2 * sqrt(s))
        (p10,), p1_det = self.e1_map.map([r[:, 0]], [m1, E1a])

        # Define the energy p10
        Delta = 2 * sqrt(s) * (sqrt(s) / 2 - p10) + m1sq
        Delta_23 = m2sq - m3sq
        dE2 = (p10**2 - m1sq) * ((Delta + Delta_23) ** 2 - 4 * m2sq * Delta)
        E2a = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        E2b = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        (p20,), p2_det = self.e2_map.map([r[:, 1]], [E2a, E2b])

        # Define angles
        phi = 2 * pi * r[:, 2]
        costheta = 2 * r[:, 3] - 1
        beta = 2 * pi * r[:, 4]
        det_omega = 8 * pi**2

        # calculate cosalpha
        num_alpha_1 = 2 * sqrt(s) * (sqrt(s) / 2 - p10 - p20)
        num_alpha_2 = m1sq + m2sq + 2 * p10 * p20 - m3sq
        denom_alpha = 2 * sqrt(p10**2 - m1sq) * sqrt(p20**2 - m2sq)
        cosalpha = (num_alpha_1 + num_alpha_2) / denom_alpha

        # Fill momenta
        p1[:, 0] = p10
        p1[:, 3] = sqrt(p10**2 - m1sq)
        p2[:, 0] = p20
        p2[:, 3] = sqrt(p20**2 - m2sq)

        # Do rotations
        p1 = rotate_zy(p1, phi, costheta)
        # Double rotate p2
        p2 = rotate_zy(p2, beta, cosalpha)
        p2 = rotate_zy(p2, phi, costheta)

        # Get final momentum
        p3[:, 0] = sqrt(s) - p10 - p20
        p3[:, 1:] = -p1[:, 1:] - p2[:, 1:]

        # boost into lab-frame
        p_decay = torch.stack([p1, p2, p3], dim=1)
        p_lab = boost(p_decay, p0[:, None])

        # get the full density
        gs = three_particle_density() * p1_det * p2_det * det_omega

        return (p_lab,), gs

    def map_inverse(self, inputs: TensorList, condition: TensorList):
        """Inverse map from decay momenta onto random numbers

        Args:
            inputs (TensorList): list with only one tensor [p_decay]
                p_lab: decay momenta (lab frame) with shape=(b,3,4)

        Returns:
            r (Tensor): random numbers with shape=(b,5)
            p0 (Tensor): incoming momentum in lab frame with shape=(b,4)
            m_out: (virtual) masses of outgoing particles
                    with shape=(b,3)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        p_lab = inputs[0]

        # Decaying particle in lab-frame
        p0 = p_lab.sum(dim=1)
        s = lsquare(p0)
        m_out = mass(p_lab)

        # boost into COM-frame
        p_decay = boost(p_lab, p0[:, None], inverse=True)

        # particle features
        p1 = p_decay[:, 0]
        p2 = p_decay[:, 1]
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]
        m3 = m_out[:, 2]
        m1sq = m1**2
        m2sq = m2**2
        m3sq = m3**2

        # Define the energy p10
        p10 = p1[:, 0]
        E1a = sqrt(s) / 2 + (m1sq - (m2 + m3) ** 2) / (2 * sqrt(s))
        (r_p1,), p1_det_inv = self.e1_map.map([p10], [m1, E1a])

        # Define the energy p10
        p20 = p2[:, 0]
        Delta = 2 * sqrt(s) * (sqrt(s) / 2 - p10) + m1sq
        Delta_23 = m2sq - m3sq
        dE2 = (p10**2 - m1sq) * ((Delta + Delta_23) ** 2 - 4 * m2sq * Delta)
        E2a = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        E2b = 1 / (2 * Delta) * ((sqrt(s) - p10) * (Delta + Delta_23) - sqrt(dE2))
        (r_p2,), p2_det_inv = self.e2_map.map([p20], [E2a, E2b])

        # get p1/2 absolute momentum
        p1mag = sqrt(edot(p1[:, 1:], p1[:, 1:]))

        # Extract phi and theta
        costheta = p1[:, 3] / p1mag
        phi = atan2(p1[:, 2], p1[:, 1])

        # Get the random numbers
        r_phi = phi / (2 * pi) + (phi < 0)
        r_theta = (costheta + 1) / 2

        # Get last angle
        p2 = inv_rotate_zy(p2, phi, costheta)
        beta = atan2(p2[:, 2], p2[:, 1])
        r_beta = beta / (2 * pi) + (beta < 0)
        det_omega_inv = 1 / (8 * pi**2)

        # Pack all together and get full density
        gs = p1_det_inv * p2_det_inv * det_omega_inv / three_particle_density()
        r = torch.stack([r_p1, r_p2, r_phi, r_theta, r_beta], dim=1)

        return (r, s, m_out), gs

    def density(self, inputs: TensorList, condition=None, inverse=False):
        """Returns the density only of the mapping"""
        if inverse:
            _, density = self.map_inverse(inputs, condition)
            return density
        _, density = self.map(inputs, condition)
        return density


class TwoToThreeScatteringLAB(PhaseSpaceMapping):
    """
    Implement 2->3 scattering, based on the mapping described in
        [1] E.~Byckling and K.~Kajantie,
        ``Reductions of the phase-space integral in terms of simpler processes,''
        Phys. Rev. 187 (1969), doi:10.1103/PhysRev.187.2008.

    This is expressed in the LAB-frame and thus requires the input momenta
    in the lab frame and the masses (or virtual ones) to construct the final decay momenta.
    Parametrizes the 2->3 kinematics via the Mandelstam t and an azimuthal angle phi.
    """

    def __init__(
        self,
        mt: Optional[Tensor] = None,
        wt: Optional[Tensor] = None,
        nu_s: float = 1.4,
        nu_t: float = 1.4,
        flat: bool = False,
    ):
        dims_in = [(2,), (2,)]
        dims_out = [(2, 4)]
        dims_c = [(2, 4)]
        super().__init__(dims_in, dims_out, dims_c)

        # Define which t-mapping is used
        # MadGraph only uses flat mappings for t!
        if flat:
            self.t_map = UniformInvariantBlock()
        elif mt is None:
            self.t_map = MasslessInvariantBlock(nu=nu)
        elif wt is None:
            self.t_map = StableInvariantBlock(mass=mt, nu=nu)
        else:
            self.t_map = BreitWignerInvariantBlock(mass=mt, width=wt)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs: list of two tensors [r, m_out]
                r: random numbers with shape=(b,2)
                m_out: (virtual) masses of outgoing particles
                    with shape=(b,2)
            condition: list with single tensor [p_in]
                p_in: incoming momenta with shape=(b,2,4)

        Returns:
            p_decay (Tensor): decay momenta (lab frame) with shape=(b,2,4)
            det (Tensor): log det of mapping with shape=(b,)
        """
        r, m_out = inputs[0], inputs[1]
        p_in = condition[0]

        # Extract random numbers, input momenta and output masses
        r1, r2 = r[:, 0], r[:, 1]
        ptot = p_in.sum(dim=1)
        p1 = p_in[:, 0]
        p2 = p_in[:, 1]
        p1_com = boost(p1, ptot, inverse=True)
        p2_com = boost(p2, ptot, inverse=True)

        # get invariants and incoming virtualities
        p1_2 = lsquare(p1)
        p2_2 = lsquare(p2)
        s = lsquare(ptot)

        # Define outgoing momenta and extract their masses/virtualities
        k1 = torch.zeros(r.shape[0], 4, device=r.device)
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]

        # Get phi angle
        phi = 2 * r1 * pi - pi
        det_phi = 2 * pi

        # get t_min and max
        tmin, tmax = tminmax(s, p1_2, p2_2, m1, m2)
        (t,), det_t = self.t_map.map([r2], condition=[-tmax, -tmin])

        # get the density and decay momenta
        k1_com, det_two_particle_inv = gen_mom_from_t_and_phi_com(
            p1_com, p2_com, t, phi, m1, m2
        )
        k1 = boost(k1_com, ptot)
        k2 = ptot - k1

        # get the density and outputs
        det_two_particle_inv = tinv_two_particle_density(s, p1_2, p2_2)
        p_out = torch.stack([k1, k2], dim=1)

        return (p_out,), det_t * det_two_particle_inv * det_phi

    def map_inverse(self, inputs: TensorList, condition=None):
        p_out = inputs[0]
        p_in = condition[0]

        # Extraxt incoming momenta
        ptot = p_in.sum(dim=1)
        p1 = p_in[:, 0]
        p2 = p_in[:, 1]
        p1_2 = lsquare(p1)
        p2_2 = lsquare(p2)
        s = lsquare(ptot)
        p1_com = boost(p1, ptot, inverse=True)

        # Get angles of incoming momenta
        p1mag = pmag(p1_com)
        phi1 = atan2(p1_com[:, 2], p1_com[:, 1])
        costheta1 = p1_com[:, 3] / p1mag.clip(min=EPS)

        # Get outgoing momenta
        k1 = p_out[:, 0]
        k2 = p_out[:, 1]

        # get invariants and outgoing masses
        m1 = mass(k1)
        m2 = mass(k2)
        m_out = torch.stack([m1, m2], dim=1)

        # boost inverse, rotate back
        # and then extract phi and theta
        k1 = boost(k1, ptot, inverse=True)
        k1 = inv_rotate_zy(k1, phi1, costheta1)
        k1mag = pmag(k1)
        costheta = k1[:, 3] / k1mag.clip(min=EPS)
        phi = atan2(k1[:, 2], k1[:, 1])

        # Map from cos_theta to t
        tmin, tmax = tminmax(s, p1_2, p2_2, m1, m2)
        t = costheta_to_invt(s, p1_2, p2_2, m1, m2, costheta)

        # Get the random numbers
        r1 = phi / 2 / pi + 0.5
        det_phi_inv = 1 / (2 * pi)
        (r2,), det_t_inv = self.t_map.map_inverse([-t], condition=[-tmax, -tmin])
        r = torch.stack([r1, r2], dim=1)

        # get the density and output momenta
        det_two_particle_inv = tinv_two_particle_density(s, p1_2, p2_2)
        p_in = torch.stack([p1, p2], dim=1)

        return (r, m_out), det_t_inv / det_two_particle_inv * det_phi_inv

    def density(self, inputs: TensorList, condition=None, inverse=False):
        """Returns the density only of the mapping"""
        if inverse:
            _, density = self.map_inverse(inputs, condition)
            return density
        _, density = self.map(inputs, condition)
        return density
