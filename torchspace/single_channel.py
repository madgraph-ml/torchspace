""" Implement ps-mappings
    for two dominant WWW and
    VBS diagrams
"""


from typing import Tuple, Optional
import torch
from torch import Tensor, sqrt, log
from math import pi

from .base import PhaseSpaceMapping, TensorList
from .functional.kinematics import boost_beam, lsquare
from .twoparticle import (
    TwoToTwoScatteringCOM,
    TwoToTwoScatteringLAB,
    TwoBodyDecayLAB,
    TwoBodyDecayCOM,
)
from .luminosity import Luminosity, ResonantLuminosity
from .invariants import (
    BreitWignerInvariantBlock,
    UniformInvariantBlock,
    MasslessInvariantBlock,
)


class SingleChannelWWW(PhaseSpaceMapping):
    """
    Dominant channel for triple WWW:

                              (k1)
                               W+
                               <
                               <
                         s12   <
    (p1) d~ --- < ---*vvvvvvvvv*vvvvvvvvv W- (k2)
                     |
                     ^ t1
                     |
    (p2) u  --- > ---*vvvvvvvvv W+ (k3)

    s-Invariants:
      shat = (k1 + k2 + k3)^2 = (p1 + p2)^2
       s12 = (k1 + k2)^2      = (k12)^2

    t-Invariants:
        t1 = (k12 - p1)^2 = (k3 - p2)^2

    Fields:
        incoming: u d~
        outgoing: W+ W- W+
    """

    def __init__(
        self,
        s_lab: Tensor,
        mw: Tensor,
        mz: Tensor,
        wz: Tensor,
    ):
        """
        Args:
            s_lab (Tensor): squared hadronic COM energy
            mw (Tensor): mass of W Boson
            mz (Tensor): mass of Z Boson
            wz (Tensor): width of Z Boson
        """
        dims_in = [(7,)]
        dims_out = [(5, 4), (2,)]
        super().__init__(dims_in, dims_out)

        # Get masses
        self.mw = mw
        self.mz = mz

        # get minimum cuts
        self.s_lab = s_lab
        self.s_hat_min = (3 * mw) ** 2

        # Define mappings
        self.luminosity = Luminosity(s_lab, self.s_hat_min)  # 2dof
        self.t1 = TwoToTwoScatteringCOM(flat=True)  # 2dof
        self.s12 = MasslessInvariantBlock() #BreitWignerInvariantBlock(mz, wz)  # 1 dof
        self.decay = TwoBodyDecayLAB()  # 2 dof

        # Get correct factors of pi
        # has to be (2*Pi)^(4-3*n)
        n_out = 3
        self.pi_factors = (2 * pi) ** (4 - 3 * n_out)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs: list of one tensors [r]
                r: random numbers with shape=(b,7)

        Returns:
            p_ext (Tensor): external momenta (lab frame) with shape=(b,5,4)
            x1x2 (Tensor): pdf fractions with shape=(b,2)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r = inputs[0]
        r_lumi = r[:, :2]
        r_s12 = r[:, 2:3]
        r_t1 = r[:, 3:5]
        r_d = r[:, 5:]

        # Do luminosity and get s_hat and rapidity
        (x1x2,), det_lumi = self.luminosity.map([r_lumi])
        s_hat = self.s_lab * x1x2.prod(dim=1)
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]

        # construct initial state momenta
        p1 = torch.zeros((r_t1.shape[0], 4), device=r.device)
        p2 = torch.zeros((r_t1.shape[0], 4), device=r.device)
        p1[:, 0] = sqrt(s_hat) / 2
        p1[:, 3] = sqrt(s_hat) / 2
        p2[:, 0] = sqrt(s_hat) / 2
        p2[:, 3] = -sqrt(s_hat) / 2
        p_in = torch.stack([p1, p2], dim=1)

        # Sample k12 propagator
        s12_min = (2 * self.mw) ** 2 * torch.ones_like(r_s12)
        s12_max = (sqrt(s_hat[:, None]) - self.mw) ** 2
        (s12,), det_s12 = self.s12.map([r_s12], condition=[s12_min, s12_max])

        # get masses/virtualities
        m12 = sqrt(s12)
        m3 = torch.ones_like(m12) * self.mw
        m_out = torch.cat([m12, m3], dim=1)
        (k12k3,), det_t1 = self.t1.map([r_t1, m_out], condition=[p_in])

        # prepare decay of z-boson
        k12 = k12k3[:, 0]
        k3 = k12k3[:, 1]
        m1m2 = torch.stack([self.mw, self.mw])[None, :]
        (k1k2,), det_decay = self.decay.map([r_d, k12, m1m2])

        # Pack all momenta including initial state
        k1, k2 = k1k2[:, 0], k1k2[:, 1]
        k_out = torch.stack([k1, k2, k3], dim=1)
        p_ext = torch.cat([p_in, k_out], dim=1)

        # Then boost into hadronic lab frame
        p_ext_lab = boost_beam(p_ext, rap)
        ps_weight = det_lumi * det_s12 * det_t1 * det_decay

        return (p_ext_lab, x1x2), ps_weight * self.pi_factors

    def map_inverse(self, inputs: TensorList, condition=None):
        """Map from momenta to random numbers

        Args:
            inputs: list of one tensors [p_ext, x1x2]
                p_ext: external momenta (lab frame) with shape=(b,5,4)
                x1x2: pdf fractions with shape=(b,2)

        Returns:
            r (Tensor): random numbers with shape=(b,7)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        p_ext_lab = inputs[0]
        x1x2 = inputs[1]

        # Undo boosts etc
        s_hat = self.s_lab * x1x2.prod(dim=1)
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]
        p_ext = boost_beam(p_ext_lab, rap, inverse=True)

        # Get initial states
        p_in = p_ext[:, :2]

        # Get third momentum
        k3 = p_ext[:, 4:5]

        # Undo decay
        k1k2 = p_ext[:, 2:4]
        (r_d, k12, _), det_decay_inv = self.decay.map_inverse([k1k2])

        # Undo t-channel 2->2
        k12k3 = torch.cat([k12[:,None], k3], dim=1)
        (r_t1, _), det_t1_inv = self.t1.map_inverse([k12k3], condition=[p_in])

        # Undo s-channel sampling
        s12 = lsquare(k12)[:, None]
        s12_min = (2 * self.mw) ** 2 * torch.ones_like(s12)
        s12_max = (sqrt(s_hat[:, None]) - self.mw) ** 2
        (r_s12,), det_s12_inv = self.s12.map_inverse(
            [s12], condition=[s12_min, s12_max]
        )

        # Undo lumi param
        (r_lumi,), det_lumi_inv = self.luminosity.map_inverse([x1x2])

        # Pack all together
        r = torch.cat([r_lumi, r_s12, r_t1, r_d], dim=1)
        r_weight = det_lumi_inv * det_s12_inv * det_t1_inv * det_decay_inv

        return (r,), r_weight / self.pi_factors

    def density(self, inputs: TensorList, condition=None, inverse=False):
        del condition
        if inverse:
            _, det = self.map_inverse(inputs)
            return det

        _, det = self.map(inputs)
        return det


class SingleChannelVBS(PhaseSpaceMapping):
    """
    Dominant channel for VBS :
                                      _ _ _ _ _ _
    (p1) c --- > ---*vvvvvvvvv W+ (k1)  |       |
                    |                   |       |
                    | t3                | k12   |
                    |                   |       |
                    *--- > --- s (k2)_ _|       | k123
                    g                           |
                    g t2                        |
                    g                           |
                    *--- > --- d (k3)_ _ _ _ _ _|
                    |
                    | t1
                    |
    (p2) u --- > ---*vvvvvvvvv W+ (k4)

    s-Invariants:
      shat = (k1 + k2 + k3 + k4)^2 = (p1 + p2)^2
      s123 = (k1 + k2 + k3)^2      = (k123)^2
       s12 = (k1 + k2)^2           = (k12)^2

    t-Invariants:
        t1 = (k123 - p1)^2 = (k4 - p2)^2
        t2 = (k12  - p1)^2 = (k3 + k4 - p2)^2
        t3 = (k1   - p1)^2 = (k2 + k3 + k4 - p2)^2

    Fields:
        incoming: u c
        outgoing: W+ W+ d s
    """

    def __init__(
        self,
        s_lab: Tensor,
        mw: Tensor,
    ):
        """
        Args:
            s_lab (Tensor): squared hadronic COM energy
            mw (Tensor): mass of W Boson
        """
        dims_in = [(10,)]
        dims_out = [(6, 4), (2,)]
        super().__init__(dims_in, dims_out)

        # Get masses
        self.mw = mw

        # get minimum cuts
        self.s_lab = s_lab
        s_hat_min = (2 * mw) ** 2

        # Define mappings
        self.luminosity = Luminosity(s_lab, s_hat_min)  # 2dof
        self.t1 = TwoToTwoScatteringCOM(flat=True)  # 2dof
        self.t2 = TwoToTwoScatteringLAB(flat=True)  # 2dof
        self.t3 = TwoToTwoScatteringLAB(flat=True)  # 2dof

        self.s12 = UniformInvariantBlock()  # 1 dof
        self.s123 = UniformInvariantBlock()  # 1 dof

        # Get correct factors of pi
        # has to be (2*Pi)^(4-3*n)
        n_out = 4
        self.pi_factors = (2 * pi) ** (4 - 3 * n_out)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to moment

        Args:
            inputs: list of one tensors [r]
                r: random numbers with shape=(b,10)

        Returns:
            p_ext (Tensor): external momenta (lab frame) with shape=(b,6,4)
            x1x2 (Tensor): pdf fractions with shape=(b,2)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        # Unpack random numbers
        r = inputs[0]
        r_lumi = r[:, :2]  # 2 dof
        r_s12 = r[:, 2:3]  # 1 dof
        r_s123 = r[:, 3:4]  # 1 dof
        r_t1 = r[:, 4:6]  # 2 dof
        r_t2 = r[:, 6:8]  # 2 dof
        r_t3 = r[:, 8:10]  # 2 dof

        # Do luminosity and get s_hat and rapidity
        (x1x2,), det_lumi = self.luminosity.map([r_lumi])
        s_hat = self.s_lab * x1x2.prod(dim=1)
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]

        # construct initial state momenta
        p1 = torch.zeros((r_t1.shape[0], 4), device=r.device)
        p2 = torch.zeros((r_t1.shape[0], 4), device=r.device)
        p1[:, 0] = sqrt(s_hat) / 2
        p1[:, 3] = sqrt(s_hat) / 2
        p2[:, 0] = sqrt(s_hat) / 2
        p2[:, 3] = -sqrt(s_hat) / 2
        p_in = torch.stack([p1, p2], dim=1)

        # Sample s-invariants
        s12_min = torch.ones_like(rap) * self.mw**2
        s12_max = (sqrt(s_hat[:, None]) - self.mw) ** 2
        (s12,), det_s12 = self.s12.map([r_s12], condition=[s12_min, s12_max])

        s123_min = s12
        s123_max = (sqrt(s_hat[:, None]) - self.mw) ** 2
        (s123,), det_s123 = self.s123.map([r_s123], condition=[s123_min, s123_max])

        # Then do t-invariant maps

        # do t maps
        m123 = sqrt(s123)
        m4 = torch.ones_like(m123) * self.mw
        m_t1 = torch.cat([m123, m4], dim=1)
        (k123k4,), det_t1 = self.t1.map([r_t1, m_t1], condition=[p_in])

        # Then do the next
        k4 = k123k4[:, 1]
        qt2 = p2 - k4
        p_t2_in = torch.stack([p1, qt2], dim=1)
        m12 = sqrt(s12)
        m3 = torch.zeros_like(m12)
        m_t2 = torch.cat([m12, m3], dim=1)
        (k12k3,), det_t2 = self.t2.map([r_t2, m_t2], condition=[p_t2_in])

        # Then do the next
        k3 = k12k3[:, 1]
        qt3 = qt2 - k3
        p_t3_in = torch.stack([p1, qt3], dim=1)
        m1 = torch.ones_like(m12) * self.mw
        m2 = torch.zeros_like(m12)
        m_t3 = torch.cat([m1, m2], dim=1)
        (k1k2,), det_t3 = self.t3.map([r_t3, m_t3], condition=[p_t3_in])

        # Get all outgoing momenta
        k1, k2 = k1k2[:, 0], k1k2[:, 1]
        k_out = torch.stack([k1, k2, k3, k4], dim=1)

        # And then all external momenta
        p_ext = torch.cat([p_in, k_out], dim=1)

        # Then boost into hadronic lab frame
        p_ext_lab = boost_beam(p_ext, rap)
        ps_weight = det_lumi * det_s12 * det_s123 * det_t1 * det_t2 * det_t3

        return (p_ext_lab, x1x2), ps_weight * self.pi_factors

    def map_inverse(self, inputs: TensorList, condition=None):
        del condition
        p_ext_lab = inputs[0]
        x1x2 = inputs[1]

        # go back into partonic COM frame
        s_hat = self.s_lab * x1x2.prod(dim=1)
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]
        p_ext = boost_beam(p_ext_lab, rap, inverse=True)

        # Get initial states and outgoing momenta
        p_in = p_ext[:, :2]
        p1 = p_in[:, 0]
        p2 = p_in[:, 1]

        # Get outgoing momenta
        k_out = p_ext[:, 2:]
        k1 = k_out[:, 0]
        k2 = k_out[:, 1]
        k3 = k_out[:, 2]
        k4 = k_out[:, 3]

        # Undo t-maps
        # t1 map
        k123 = k1 + k2 + k3
        k123k4 = torch.stack([k123, k4], dim=1)
        (r_t1, _), det_t1_inv = self.t1.map_inverse([k123k4], condition=[p_in])

        # t1 map
        k12 = k1 + k2
        qt2 = p1 - k4
        p_t2_in = torch.stack([p1, qt2], dim=1)
        k12k3 = torch.stack([k12, k3], dim=1)
        (r_t2, _), det_t2_inv = self.t2.map_inverse([k12k3], condition=[p_t2_in])

        # t3 map
        qt3 = qt2 - k3
        p_t3_in = torch.stack([p1, qt3], dim=1)
        k1k2 = torch.stack([k1, k2], dim=1)
        (r_t3, _), det_t3_inv = self.t3.map_inverse([k1k2], condition=[p_t3_in])

        # Undo s invariants
        # Sample s-invariants
        s12 = lsquare(k12)[:, None]
        s12_min = torch.ones_like(s12) * self.mw**2
        s12_max = (sqrt(s_hat[:, None]) - self.mw) ** 2
        (r_s12,), det_s12_inv = self.s12.map_inverse(
            [s12], condition=[s12_min, s12_max]
        )

        s123 = lsquare(k123)[:, None]
        s123_min = s12
        s123_max = (sqrt(s_hat[:, None]) - self.mw) ** 2
        (r_s123,), det_s123_inv = self.s123.map_inverse(
            [s123], condition=[s123_min, s123_max]
        )

        # Undo lumi
        # Do luminosity and get s_hat and rapidity
        (r_lumi,), det_lumi_inv = self.luminosity.map_inverse([x1x2])

        # Pack all together
        r = torch.cat([r_lumi, r_s12, r_s123, r_t1, r_t2, r_t3])
        r_weight = (
            det_lumi_inv
            * det_s12_inv
            * det_s123_inv
            * det_t1_inv
            * det_t2_inv
            * det_t3_inv
        )

        return (r,), r_weight / self.pi_factors

    def density(self, inputs: TensorList, condition=None, inverse=False):
        del condition
        if inverse:
            _, det = self.map_inverse(inputs)
            return det

        _, det = self.map(inputs)
        return det


class Diagramm_ww_llvv(PhaseSpaceMapping):
    """
    We implement Figure 4.12 from [1]
        [1] https://arxiv.org/abs/hep-ph/0008033
                                 
    (p1) f~ \                 v*-->-- v_mu (k1) 
             \_              v  \_
             |\          W+ v   |\
               \           v      \ mu+ (k2)
                *vvvvvvvvvv*
               /     V1    v     _/ tau- (k3)
             _/          W- v    /|
             /|              v  /
    (p2) f  /                 v*--<-- v_tau(k4)

    s-Invariants:
      shat = (k1 + k2 + k3 + k4)^2 = (p1 + p2)^2
       s12 = (k1 + k2)^2           = (k12)^2
       s34 = (k3 + k4)^2           = (k34)^2

    Fields:
        incoming: f f~
            f: q -> including luminosity sampling
            f= l -> lepton collider (default)
        intermediate: V1 W+ W-, with V1=gamma,Z
        outgoing: v_mu mu+ tau- v_tau~
    """

    def __init__(
        self,
        s_lab: Tensor,
        mw: Tensor,
        ww: Tensor,
        mV: Tensor = None,
        wV: Tensor = None,
        leptonic: bool = True,
    ):
        """
        Args:
            s_lab (Tensor): squared hadronic COM energy
            mw (Tensor): mass of W Boson
            ww (Tensor): width of W Boson
            mV (Tensor): mass of V Boson. Defaults to None which
                corresponds to the photon channel.
            wV (Tensor): width of V Boson. Defaults to None which
                corresponds to the photon channel.
            leptonic (bool): leptonic or hadronic collider. Default is True.
        """
        if leptonic:
            dims_in = [(10,)]
        else:
            dims_in = [(8,)]

        dims_out = [(6, 4), (2,)]
        super().__init__(dims_in, dims_out)

        # Get masses
        self.mw = mw
        self.ww = ww

        # get minimum cuts
        self.s_lab = s_lab
        s_hat_min = (torch.tensor(20.0)) ** 2

        # Define mappings
        if not leptonic:
            if mV is not None:
                self.luminosity = ResonantLuminosity(s_lab, mV, wV, s_hat_min)  # 2dof
            else:
                self.luminosity = Luminosity(s_lab, s_hat_min)  # 2dof

        self.s12 = BreitWignerInvariantBlock(mw, ww)  # 1 dof
        self.s34 = BreitWignerInvariantBlock(mw, ww)  # 1 dof
        self.decay1 = TwoBodyDecayCOM()  # 2 dof
        self.decay2 = TwoBodyDecayLAB()  # 2 dof
        self.decay3 = TwoBodyDecayLAB()  # 2 dof

        self.leptonic = leptonic

        # Get correct factors of pi
        # has to be (2*Pi)^(4-3*n)
        n_out = 4
        self.pi_factors = (2 * pi) ** (4 - 3 * n_out)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to moment

        Args:
            inputs: list of one tensors [r]
                r: random numbers with shape=(b,8/10)

        Returns:
            p_ext (Tensor): external momenta (lab frame) with shape=(b,6,4)
            x1x2 (Tensor): pdf fractions with shape=(b,2).
                Defaults to 1.0 in leptonic case.
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        # Unpack random numbers
        r = inputs[0]
        r_s12 = r[:, 0:1]  # 1 dof
        r_s34 = r[:, 1:2]  # 1 dof
        r_d1 = r[:, 2:4]  # 2 dof
        r_d2 = r[:, 4:6]  # 2 dof
        r_d3 = r[:, 6:8]  # 2 dof

        # Do luminosity and get s_hat and rapidity
        if not self.leptonic:
            r_lumi = r[:, 8:]  # 2 dof
            (x1x2,), det_lumi = self.luminosity.map([r_lumi])
            s_hat = self.s_lab * x1x2.prod(dim=1)
            rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]
        else:
            s_hat = self.s_lab * torch.ones((r_s12.shape[0],), device=r.device)
            det_lumi = 1.0
            x1x2 = torch.ones((r_s12.shape[0], 2), device=r.device)

        # construct initial state momenta
        p1 = torch.zeros((r_s12.shape[0], 4), device=r.device)
        p2 = torch.zeros((r_s12.shape[0], 4), device=r.device)
        p1[:, 0] = sqrt(s_hat) / 2
        p1[:, 3] = sqrt(s_hat) / 2
        p2[:, 0] = sqrt(s_hat) / 2
        p2[:, 3] = -sqrt(s_hat) / 2
        p_in = torch.stack([p1, p2], dim=1)

        # Sample s-invariants
        s12_min = torch.zeros_like(s_hat[:, None])
        s12_max = s_hat[:, None]
        (s12,), det_s12 = self.s12.map([r_s12], condition=[s12_min, s12_max])
        s34_min = torch.zeros_like(s_hat[:, None])
        s34_max = (sqrt(s_hat[:, None]) - sqrt(s12)) ** 2
        (s34,), det_s34 = self.s34.map([r_s34], condition=[s34_min, s34_max])

        # Then do decays

        # do first decay p -> k12 k34
        m12 = sqrt(s12)
        m34 = sqrt(s34)
        m_d1 = torch.cat([m12, m34], dim=1)
        (k12k34,), det_d1 = self.decay1.map([r_d1, s_hat, m_d1])

        # Then k12 -> k1 k2
        k12 = k12k34[:, 0]
        m1 = torch.zeros_like(s12)
        m2 = torch.zeros_like(s12)
        m_d2 = torch.cat([m1, m2], dim=1)
        (k1k2,), det_d2 = self.decay2.map([r_d2, k12, m_d2])

        # Then k34 -> k3 k4
        k34 = k12k34[:, 1]
        m3 = torch.zeros_like(s12)
        m4 = torch.zeros_like(s12)
        m_d3 = torch.cat([m3, m4], dim=1)
        (k3k4,), det_d3 = self.decay3.map([r_d3, k34, m_d3])

        # Get all outgoing momenta
        k_out = torch.cat([k1k2, k3k4], dim=1)

        # And then all external momenta
        p_ext = torch.cat([p_in, k_out], dim=1)

        # Then boost into hadronic lab frame
        p_ext_lab = p_ext if self.leptonic else boost_beam(p_ext, rap)
        ps_weight = det_lumi * det_s12 * det_s34 * det_d1 * det_d2 * det_d3

        return (p_ext_lab, x1x2), ps_weight * self.pi_factors

    def map_inverse(self, inputs: TensorList, condition=None):
        raise NotImplementedError("keine lust 2...")

    def density(self, inputs: TensorList, condition=None, inverse=False):
        del condition
        if inverse:
            raise NotImplementedError("keine lust 2...")

        _, det = self.map(inputs)
        return det


class Diagramm_llvvA(PhaseSpaceMapping):
    """
    We implement Figure 4.13 from [1]
        [1] https://arxiv.org/abs/hep-ph/0008033
                                 
    (p1) f~ ----- < ----*------ < ------ f'~ (k1)
                        v 
                        v              / mu-  (k2)
                 W (t2) v             /
                        v            *vvvvv A (k3)
                        v           /
                        *vvvvvvvvvv*
                        v    V1     \
                 W (t1) v            \
                        v             \ mu+   (k4)
                        v
    (p2) f  ----- > ----*------ > ------ f'  (k5)
    
    s-Invariants:
      shat = (p1 + p2)^2
       s23 = (k2 + k3)^2 = (k23)^2
      s234 = (k2 + k3 + k4)^2 = (k234)^2
     s1234 = (k1 + k2 + k3 + k4)^2 = (k1234)^2
        
    t-Invariants:
        t1 = (k1234 - p1)^2 = (k5    - p2)^2
        t2 = (k1    - p1)^2 = (k2345 - p2)^2

    Fields:
        incoming: f~ f
            f = q -> including luminosity sampling
            f = e -> lepton collider (default)
        intermediate: W W V1, with V1=gamma,Z
            default: V1=gamma
        outgoing: f'~ f' mu- mu+ A
    """

    def __init__(
        self,
        s_lab: Tensor,
        mw: Tensor,
        ww: Tensor,
        mV: Tensor = None,
        wV: Tensor = None,
        leptonic: bool = True,
    ):
        """
        Args:
            s_lab (Tensor): squared hadronic COM energy
            mw (Tensor): mass of W Boson
            ww (Tensor): width of W Boson
            mV (Tensor): mass of V1 Boson. Defaults to None which
                corresponds to the photon channel.
            wV (Tensor): width of V1 Boson. Defaults to None which
                corresponds to the photon channel.
            leptonic (bool): leptonic or hadronic collider. Default is True.
        """
        if leptonic:
            dims_in = [(13,)]
        else:
            dims_in = [(11,)]

        dims_out = [(7, 4), (2,)]
        super().__init__(dims_in, dims_out)

        # Get masses
        self.mw = mw
        self.ww = ww

        # get minimum cuts
        self.s_lab = s_lab
        s_hat_min = torch.tensor(20.0**2)
        self.s23_min = torch.tensor(1e-2)

        # Define mappings
        if not leptonic:
            self.luminosity = Luminosity(s_lab, s_hat_min)  # 2dof

        self.s23 = MasslessInvariantBlock()  # 1 dof
        if mV is not None:
            self.s234 = BreitWignerInvariantBlock(mV, wV)  # 1 dof
        else:
            self.s234 = MasslessInvariantBlock()  # 1 dof
        self.s1234 = UniformInvariantBlock()  # 1 dof
        self.t1 = TwoToTwoScatteringCOM(flat=True)  # 2 dof
        self.t2 = TwoToTwoScatteringLAB(flat=True)  # 2 dof
        self.d23 = TwoBodyDecayLAB()  # 2 dof
        self.d234 = TwoBodyDecayLAB()  # 2 dof

        self.leptonic = leptonic

        # Get correct factors of pi
        # has to be (2*Pi)^(4-3*n)
        n_out = 5
        self.pi_factors = (2 * pi) ** (4 - 3 * n_out)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to moment

        Args:
            inputs: list of one tensors [r]
                r: random numbers with shape=(b,11/13)

        Returns:
            p_ext (Tensor): external momenta (lab frame) with shape=(b,7,4)
            x1x2 (Tensor): pdf fractions with shape=(b,2).
                Defaults to 1.0 in leptonic case.
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        # Unpack random numbers
        r = inputs[0]
        r_s23 = r[:, 0:1]  # 1 dof
        r_s234 = r[:, 1:2]  # 1 dof
        r_s1234 = r[:, 2:3]  # 1 dof
        r_t1 = r[:, 3:5]  # 2 dof
        r_t2 = r[:, 5:7]  # 2 dof
        r_d234 = r[:, 7:9]  # 2 dof
        r_d23 = r[:, 9:11]  # 2 dof

        # Do luminosity and get s_hat and rapidity
        if not self.leptonic:
            r_lumi = r[:, 11:]  # 2 dof
            (x1x2,), det_lumi = self.luminosity.map([r_lumi])
            s_hat = self.s_lab * x1x2.prod(dim=1)
            rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]
        else:
            s_hat = self.s_lab * torch.ones((r_s23.shape[0],), device=r.device)
            det_lumi = 1.0
            x1x2 = torch.ones((r_s23.shape[0], 2), device=r.device)

        # construct initial state momenta
        p1 = torch.zeros((r_s23.shape[0], 4), device=r.device)
        p2 = torch.zeros((r_s23.shape[0], 4), device=r.device)
        p1[:, 0] = sqrt(s_hat) / 2
        p1[:, 3] = sqrt(s_hat) / 2
        p2[:, 0] = sqrt(s_hat) / 2
        p2[:, 3] = -sqrt(s_hat) / 2
        p_in = torch.stack([p1, p2], dim=1)

        # ----------------------------
        # Sample s-invariants

        s23_min = torch.ones_like(s_hat[:, None]) * self.s23_min
        s23_max = s_hat[:, None]
        (s23,), det_s23 = self.s23.map([r_s23], condition=[s23_min, s23_max])

        s234_min = s23
        s234_max = s_hat[:, None]
        (s234,), det_s234 = self.s234.map([r_s234], condition=[s234_min, s234_max])

        s1234_min = s234
        s1234_max = s_hat[:, None]
        (s1234,), det_s1234 = self.s1234.map(
            [r_s1234], condition=[s1234_min, s1234_max]
        )

        # ----------------------------
        # Then t mappings

        # first do p1 p2 -> k1234 k5
        m1234 = sqrt(s1234)
        m5 = torch.zeros_like(m1234)
        m_t1 = torch.cat([m1234, m5], dim=1)
        (k1234k5,), det_t1 = self.t1.map([r_t1, m_t1], condition=[p_in])

        # Then do p1 q1 -> k1 k234
        k5 = k1234k5[:, 1]
        q1 = p2 - k5
        p_t2_in = torch.stack([p1, q1], dim=1)
        m1 = torch.zeros_like(s234)
        m234 = sqrt(s234)
        m_t2 = torch.cat([m1, m234], dim=1)
        (k1k234,), det_t2 = self.t2.map([r_t2, m_t2], condition=[p_t2_in])

        # ----------------------------
        # Then do s-channel decays

        # do first decay k234 -> k23 k4
        k1 = k1k234[:, 0]
        k234 = k1k234[:, 1]
        m23 = sqrt(s23)
        m4 = torch.zeros_like(s23)
        m_d234 = torch.cat([m23, m4], dim=1)
        (k23k4,), det_d234 = self.d234.map([r_d234, k234, m_d234])

        # Then k23 -> k2 k3
        k4 = k23k4[:, 1]
        k23 = k23k4[:, 0]
        m2 = torch.zeros_like(s23)
        m3 = torch.zeros_like(s23)
        m_d23 = torch.cat([m2, m3], dim=1)
        (k2k3,), det_d23 = self.d23.map([r_d23, k23, m_d23])

        # Get all outgoing momenta
        k2, k3 = k2k3[:, 0], k2k3[:, 1]
        k_out = torch.stack([k1, k2, k3, k4, k5], dim=1)

        # And then all external momenta
        p_ext = torch.cat([p_in, k_out], dim=1)

        # Then boost into hadronic lab frame
        p_ext_lab = p_ext if self.leptonic else boost_beam(p_ext, rap)
        ps_weight = (
            det_lumi
            * det_s23
            * det_s234
            * det_s1234
            * det_t1
            * det_t2
            * det_d234
            * det_d23
        )

        return (p_ext_lab, x1x2), ps_weight * self.pi_factors

    def map_inverse(self, inputs: TensorList, condition=None):
        raise NotImplementedError("keine lust 2...")

    def density(self, inputs: TensorList, condition=None, inverse=False):
        del condition
        if inverse:
            raise NotImplementedError("keine lust 2...")

        _, det = self.map(inputs)
        return det
