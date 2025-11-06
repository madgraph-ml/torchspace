"""Implement two-particle mappings.
Bases on the mappings described in
[1] https://freidok.uni-freiburg.de/data/154629
[2] https://arxiv.org/abs/hep-ph/0008033
"""

from typing import Optional

import torch
from torch import Tensor, atan2, sqrt

from .base import PhaseSpaceMapping, TensorList
from .functional.kinematics import (
    EPS,
    boost,
    inv_rotate_zy,
    kaellen,
    lsquare,
    mass,
    pi,
    pmag,
    rotate_zy,
)
from .functional.ps_utils import tinv_two_particle_density, two_particle_density
from .functional.tchannel import costheta_to_invt, gen_mom_from_t_and_phi_com, tminmax
from .invariants import (
    BreitWignerInvariantBlock,
    MasslessInvariantBlock,
    StableInvariantBlock,
    UniformInvariantBlock,
)


class TwoBodyDecayCOM(PhaseSpaceMapping):
    """
    Implement isotropic 2-body phase-space, based on the mapping described in
        [2] https://arxiv.org/abs/hep-ph/0008033
        [3] https://freidok.uni-freiburg.de/data/154629

    This is expressed in the COM-frame and thus only requires the COM energy
    and the masses (or virtual ones) to construct the final decay momenta.
    """

    def __init__(self):
        dims_in = [(2,), (), (2,)]
        dims_out = [(2, 4)]
        super().__init__(dims_in, dims_out)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs (TensorList): list of two tensors [r, s, m_out]
                r: random numbers with shape=(b,2)
                s: squared COM energy with shape=(b,)
                m_out: (virtual) masses of outgoing particles
                    with shape=(b,2)

        Returns:
            p_decay (Tensor): decay momenta (lab frame) with shape=(b,2,4)
            det (Tensor): log det of mapping with shape=(b,)
        """
        del condition
        r, s, m_out = inputs[0], inputs[1], inputs[2]
        p1 = torch.zeros(r.shape[0], 4, device=r.device)
        p2 = torch.zeros(r.shape[0], 4, device=r.device)

        # with torch.no_grad():
        #    if torch.any(s < 0):
        #        raise ValueError(f"s needs to be always positive")

        r1, r2 = r[:, 0], r[:, 1]

        # Define the angles
        phi = 2 * pi * r1 - pi
        costheta = 2 * r2 - 1
        g_angular = 4 * pi

        # Define the momenta (in COM frame of decaying particle)
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]
        p1[:, 0] = (s + m1**2 - m2**2) / (2 * sqrt(s))
        p2[:, 0] = (s + m2**2 - m1**2) / (2 * sqrt(s))
        p1[:, 3] = sqrt(kaellen(s, m1**2, m2**2)) / (2 * sqrt(s))

        # Rotate and define p2 spatial components
        p1 = rotate_zy(p1, phi, costheta)
        p2[:, 1:] = -p1[:, 1:]

        # get the density and decay momenta
        # (C.10) in [2] == (C.8) * (4PI)
        gs = two_particle_density(s, m1**2, m2**2) * g_angular
        p_decay = torch.stack([p1, p2], dim=1)

        return (p_decay,), gs

    def map_inverse(self, inputs: TensorList, condition=None):
        """Inverse map from decay momenta onto random numbers

        Args:
            inputs (TensorList): list with only one tensor [p_decay]
                p_decay: decay momenta (lab frame) with shape=(b,2,4)

        Returns:
            r (Tensor): random numbers with shape=(b,2)
            det (Tensor): log det of mapping with shape=(b,)
        """
        del condition
        p_decay = inputs[0]
        m_out = mass(p_decay)
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]

        # Decaying particle in lab-frame
        p0 = p_decay.sum(dim=1)
        s = lsquare(p0)
        p1 = p_decay[:, 0]

        # get p1 absolute momentum
        p1mag = pmag(p1)

        # Extract phi and theta
        costheta = p1[:, 3] / p1mag.clip(min=EPS)
        phi = atan2(p1[:, 2], p1[:, 1])

        # Get the random numbers
        r1 = phi / 2 / pi + 0.5
        r2 = (costheta + 1) / 2
        g_angular = 4 * pi

        # get the density and random numbers
        # (C.10) in [2] == (C.8)/(4PI)
        gs = two_particle_density(s, m1**2, m2**2) * g_angular
        r = torch.stack([r1, r2], dim=1)

        return (r, s, m_out), 1 / gs

    def density(self, inputs: TensorList, condition=None, inverse=False):
        del condition
        if inverse:
            p_decay = inputs[0]
            p1sq = lsquare(p_decay)[:, 0]
            p2sq = lsquare(p_decay)[:, 1]
            # Get full COM
            p0 = p_decay.sum(dim=1)
            s = lsquare(p0)
            gs = two_particle_density(s, p1sq, p2sq) * (4 * pi)
            return 1 / gs

        s = inputs[1]
        m1 = inputs[2][:, 0]
        m2 = inputs[2][:, 1]
        gs = two_particle_density(s, m1**2, m2**2) * (4 * pi)
        return gs


class TwoBodyDecayLAB(PhaseSpaceMapping):
    """
    Implement isotropic 2-body phase-space, based on the mapping described in
        [2] https://arxiv.org/abs/hep-ph/0008033
        [3] https://freidok.uni-freiburg.de/data/154629

    This is expressed in the LAB-frame and thus requires the input momentum
    in the lab frame. It also requires the masses (or virtual ones) to construct the final decay momenta.
    Parametrizes the 2-body kinematics via two angles (theta, phi).
    """

    def __init__(self):
        dims_in = [(2,), (4,), (2,)]
        dims_out = [(2, 4)]
        super().__init__(dims_in, dims_out)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs (TensorList): list of two tensors [r, s, m_out]
                r: random numbers with shape=(b,2)
                p0: total momentum in lab frame with shape=(b,4)
                m_out: (virtual) masses of outgoing particles
                    with shape=(b,2)

        Returns:
            p_decay (Tensor): decay momenta (lab frame) with shape=(b,2,4)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r, p0, m_out = inputs[0], inputs[1], inputs[2]
        p1 = torch.zeros(r.shape[0], 4, device=r.device)
        s = lsquare(p0)

        # with torch.no_grad():
        #    if torch.any(s < 0):
        #        raise ValueError(f"s needs to be always positive")

        r1, r2 = r[:, 0], r[:, 1]

        # Define the angles
        phi = 2 * pi * r1 - pi
        costheta = 2 * r2 - 1
        g_angular = 4 * pi

        # Define the momenta (in COM frame of decaying particle)
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]
        p1[:, 0] = (s + m1**2 - m2**2) / (2 * sqrt(s))
        p1[:, 3] = sqrt(kaellen(s, m1**2, m2**2)) / (2 * sqrt(s))

        # First rotate, then boost into lab-frame
        p1 = rotate_zy(p1, phi, costheta)
        p1 = boost(p1, p0)
        p2 = p0 - p1

        # get the density and decay momenta
        gs = two_particle_density(s, m1**2, m2**2) * g_angular
        p_decay = torch.stack([p1, p2], dim=1)

        return (p_decay,), gs

    def map_inverse(self, inputs: TensorList, condition=None):
        """Inverse map from decay momenta onto random numbers

        Args:
            inputs (TensorList): list with only one tensor [p_decay]
                p_decay: decay momenta (lab frame) with shape=(b,2,4)

        Returns:
            r (Tensor): random numbers with shape=(b,2)
            det (Tensor): log det of mapping with shape=(b,)
        """
        del condition
        p_decay = inputs[0]
        m_out = mass(p_decay)
        m1 = m_out[:, 0]
        m2 = m_out[:, 1]

        # Decaying particle in lab-frame
        p0 = p_decay.sum(dim=1)
        s = lsquare(p0)
        p1 = p_decay[:, 0]

        # Boost p1 into COM
        p1 = boost(p1, p0, inverse=True)

        # get p1 absolute momentum
        p1mag = pmag(p1)

        # Extract phi and theta
        costheta = p1[:, 3] / p1mag.clip(min=EPS)
        phi = atan2(p1[:, 2], p1[:, 1])

        # Get the random numbers
        r1 = phi / (2 * pi) + 0.5
        r2 = (costheta + 1) / 2
        g_angular = 4 * pi

        # get the density
        gs = two_particle_density(s, m1**2, m2**2) * g_angular
        r = torch.stack([r1, r2], dim=1)

        return (r, p0, m_out), 1 / gs

    def density(self, inputs: TensorList, condition=None, inverse=False):
        del condition
        if inverse:
            p_decay = inputs[0]
            p1sq = lsquare(p_decay)[:, 0]
            p2sq = lsquare(p_decay)[:, 1]
            # Get full COM
            p0 = p_decay.sum(dim=1)
            s = lsquare(p0)
            gs = two_particle_density(s, p1sq, p2sq) * (4 * pi)
            return 1 / gs

        s = lsquare(inputs[1])
        m1 = inputs[2][:, 0]
        m2 = inputs[2][:, 1]
        gs = two_particle_density(s, m1**2, m2**2) * (4 * pi)
        return gs


class TwoToTwoScatteringCOM(PhaseSpaceMapping):
    """
    Implement 2->2 scattering, based on the mapping described in
        [1] https://arxiv.org/abs/hep-ph/0008033
        [2] https://freidok.uni-freiburg.de/data/154629

    This is expressed in the COM-frame and thus only requires the COM energy
    and the masses (or virtual ones) to construct the final decay momenta.
    Parametrizes the 2->2 kinematics via the Mandelstam t and an azimuthal angle phi.
    """

    def __init__(
        self,
        mt: Optional[Tensor] = None,
        wt: Optional[Tensor] = None,
        nu: float = 1.4,
        flat: bool = False,
    ):
        dims_in = [(2,), (2,)]
        dims_out = [(2, 4)]
        dims_c = [(2, 4)]
        super().__init__(dims_in, dims_out, dims_c)

        # Define which t-mapping is used
        # MadGraph only uses flat mappings for t? (Check with Olivio)
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

        # sample t invariant
        tmin, tmax = tminmax(s, p1_2, p2_2, m1, m2)
        (t,), det_t = self.t_map.map([r2], condition=[-tmax, -tmin])

        # get the density and decay momenta
        k1, det_two_particle_inv = gen_mom_from_t_and_phi_com(p1, p2, t, phi, m1, m2)
        k2 = ptot - k1
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

        # Get angles of incoming momenta
        p1mag = pmag(p1)
        phi1 = atan2(p1[:, 2], p1[:, 1])
        costheta1 = p1[:, 3] / p1mag.clip(min=EPS)

        # Extraxt individual momenta
        k1 = p_out[:, 0]
        k2 = p_out[:, 1]

        # get invariants and outgoing masses
        m1 = mass(k1)
        m2 = mass(k2)
        m_out = torch.stack([m1, m2], dim=1)

        # get rotate away angles from incoming momenta
        # and then extract phi and theta
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


class TwoToTwoScatteringLAB(PhaseSpaceMapping):
    """
    Implement 2->2 scattering, based on the mapping described in
        [1] https://arxiv.org/abs/hep-ph/0008033
        [2] https://freidok.uni-freiburg.de/data/154629

    This is expressed in the LAB-frame and thus requires the input momenta
    in the lab frame and the masses (or virtual ones) to construct the final decay momenta.
    Parametrizes the 2->2 kinematics via the Mandelstam t and an azimuthal angle phi.
    """

    def __init__(
        self,
        mt: Optional[Tensor] = None,
        wt: Optional[Tensor] = None,
        nu: float = 0.8,
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
