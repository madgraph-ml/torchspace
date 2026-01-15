from math import gamma, pi
from typing import Optional, Tuple

import torch
from torch import Tensor, atan2, log, sqrt

from .base import PhaseSpaceMapping, TensorList
from .functional.kinematics import boost, boost_beam, esquare, mass, sqrt_shat
from .functional.ps_utils import (
    build_p_in,
    map_fourvector_rambo,
    map_fourvector_rambo_diet,
    pin_to_x1x2,
    two_body_decay_factor,
)
from .luminosity import FlatLuminosity, Luminosity, ResonantLuminosity
from .rootfinder.roots import get_u_parameter, get_xi_parameter
from .spline_mapping import SplineMapping

# ------------------------------------------
# Full Phase Space Generators
# ------------------------------------------
#   has outputs (p_ext,) or (p_ext, x1x2)


class Rambo(PhaseSpaceMapping):
    """Rambo algorithm as presented in
    [1] Rambo [Comput. Phys. Commun. 40 (1986) 359-373]

    Note: we make ``e_cm`` an input instead of a fixed initialization
          parameter to make it compatible for event sampling
          for hadron colliders.
    """

    def __init__(
        self,
        nparticles: int,
        masses: list[float] = None,
        debug: bool = False,
    ):
        self.nparticles = nparticles

        if masses is not None and sum(masses) > 0.0:
            if debug:
                print("Using massive Rambo!")
            self.masses = torch.tensor(masses)
            assert len(self.masses) == self.nparticles
            self.e_min = sum(masses)
        else:
            if debug:
                print("Using massless Rambo!")
            self.masses = None
            self.e_min = 0.0

        dims_in = [(4 * nparticles,), ()]
        dims_out = [(nparticles, 4)]
        super().__init__(dims_in, dims_out)

        self.pi_factors = (2 * pi) ** (4 - 3 * self.nparticles)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs: list of tensors [r, e_cm]
                r: random numbers with shape=(b,4*n)
                e_cm: COM energy with shape=(b,) or shape=()

        Returns:
            p_ext (Tensor): external momenta with shape=(b,n+2,4)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r = inputs[0]  # has dims (b,4*n)
        e_cm = inputs[1]  # has dims (b,) or ()

        # reshape random numbers to future shape=(b,n,4)
        r = r.reshape((-1, self.nparticles, 4))

        # Check if the partonic COM energy is large enough
        with torch.no_grad():
            if torch.any(e_cm <= self.e_min):
                raise ValueError(
                    f"partonic COM energy needs to be larger than sum of external masses!"
                )

        # Construct intermediate particle momenta
        q = map_fourvector_rambo(r)

        # sum over all particles
        Q = q.sum(dim=1, keepdim=True)  # has shape (b,1,4)

        # Get scaling factor and match dimensions
        M = mass(Q)  # has shape (b,1)
        x = e_cm[..., None] / M  # has shape (b,1)

        # Boost and refactor
        p_out = boost(q, Q, inverse=True)
        p_out = x[..., None] * p_out

        torch_ones = torch.ones((r.shape[0],), device=r.device)
        w0 = torch_ones * self._massles_weight(e_cm)

        # construct initial state momenta
        p_in = build_p_in(e_cm)

        if self.masses is not None:
            # match dimensions of masses
            m = self.masses[None, ...].to(device=r.device)

            # solve for xi in massive case, see Ref. [1]
            xi = get_xi_parameter(p_out[:, :, 0], m)

            # Make momenta massive
            xi = xi[:, None, None]
            k_out = torch.empty_like(p_out)
            k_out[:, :, 0] = torch.sqrt(m**2 + xi[:, :, 0] ** 2 * p_out[:, :, 0] ** 2)
            k_out[:, :, 1:] = xi * p_out[:, :, 1:]

            # Get massive density
            w_m = self._massive_weight(k_out, p_out, xi[:, 0, 0])

            p_ext = torch.cat([p_in, k_out], dim=1)
            return (p_ext,), w_m * w0

        p_ext = torch.cat([p_in, p_out], dim=1)
        return (p_ext,), w0

    def map_inverse(self, inputs, condition=None):
        """Does not exist for Rambo"""
        raise NotImplementedError

    def _massles_weight(self, e_cm):
        w0 = (
            (pi / 2.0) ** (self.nparticles - 1)
            * e_cm ** (2 * self.nparticles - 4)
            / (gamma(self.nparticles) * gamma(self.nparticles - 1))
        )
        return w0 * self.pi_factors

    def _massive_weight(
        self,
        k: torch.Tensor,
        p: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            k (torch.Tensor): massive momenta in shape=(b,n,4)
            p (torch.Tensor): massless momenta in shape=(b,n,4)
            xi (torch.Tensor, Optional): shift variable with shape=(b,)

        Returns:
            torch.Tensor: massive weight
        """
        # get correction factor for massive ones
        ks2 = k[:, :, 1] ** 2 + k[:, :, 2] ** 2 + k[:, :, 3] ** 2
        ps2 = p[:, :, 1] ** 2 + p[:, :, 2] ** 2 + p[:, :, 3] ** 2
        k0 = k[:, :, 0]
        p0 = p[:, :, 0]
        w_M = (
            xi ** (3 * self.nparticles - 3)
            * torch.prod(p0 / k0, dim=1)
            * torch.sum(ps2 / p0, dim=1)
            / torch.sum(ks2 / k0, dim=1)
        )
        return w_M

    def density(self, inputs, condition=None, inverse=False):
        del condition
        if inverse:
            raise NotImplementedError

        _, gs = self.map(self, inputs)
        return gs


class RamboOnDiet(PhaseSpaceMapping):
    """Rambo on Diet algorithm as presented in
        [2] Rambo on diet - https://arxiv.org/abs/1308.2922

    Note, that here is an error in the algorithm of [2], which has been fixed.
    For details see:
        [3] RW's PhD thesis - https://doi.org/10.11588/heidok.00029154
        [4] ELSA paper - https://arxiv.org/abs/2305.07696
    """

    def __init__(
        self,
        nparticles: int,
        masses: list[float] = None,
        check_emin: bool = False,
        debug: bool = False,
    ):
        self.nparticles = nparticles
        self.check_emin = check_emin

        if masses is not None and sum(masses) > 0.0:
            if debug:
                print("Using massive RamboOnDiet!")
            self.masses = torch.tensor(masses)
            assert len(self.masses) == self.nparticles
            self.e_min = sum(masses)
        else:
            if debug:
                print("Using massless RamboOnDiet!")
            self.masses = None
            self.e_min = 0.0

        dims_in = [(3 * nparticles - 4,), ()]
        dims_out = [(nparticles, 4)]

        self.pi_factors = (2 * pi) ** (4 - 3 * self.nparticles)

        super().__init__(dims_in, dims_out)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs: list of tensors [r, e_cm]
                r: random numbers with shape=(b,3*n-4)
                e_cm: COM energy with shape=(b,)

        Returns:
            p_ext (Tensor): external momenta with shape=(b,n+2,4)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r = inputs[0]  # has dims (b,3*n-4)
        e_cm = inputs[1]  # has dims (b,) or ()

        # Check if the partonic COM energy is large enough
        with torch.no_grad():
            if self.check_emin:
                if torch.any(e_cm <= self.e_min):
                    raise ValueError(
                        f"partonic COM energy needs to be larger than sum of external masses!"
                    )

        # Do rambo on diet loop
        # prepare momenta
        p_out = torch.empty((r.shape[0], self.nparticles, 4), device=r.device)

        # split random numbers in energy and angular
        ru, romega = r[:, : self.nparticles - 2], r[:, self.nparticles - 2 :]

        # Solve rambo equation numerically for all u directly
        # u has shape=(b, nparticles - 2)
        u = get_u_parameter(ru)

        # split into rcos and rphi
        rcos, rphi = romega.split(self.nparticles - 1, dim=1)

        # Define all angles vectorized
        cos_theta = 2 * rcos - 1
        phi = 2 * pi * rphi

        # Define intermediate masses
        M = torch.zeros((r.shape[0], self.nparticles), device=r.device)
        M[:, 0] = e_cm
        M[:, 1:-1] = torch.cumprod(u, dim=1) * e_cm[:, None]

        # Define first n-1 energies
        # gets shape (b, nparticles - 1)
        q = 4 * M[:, :-1] * two_body_decay_factor(M[:, :-1], M[:, 1:], 0)

        # Define first (n-1) particles
        pnm1 = map_fourvector_rambo_diet(q, cos_theta, phi)

        # Define Qs
        Q = e_cm[:, None] * torch.tile(torch.tensor([1, 0, 0, 0]), (r.shape[0], 1))

        # Define loop over (n-1 particles) boosts
        for i in range(self.nparticles - 1):
            # Define Qi
            Q0_i = sqrt(q[:, i] ** 2 + M[:, i + 1] ** 2)
            Qp_i = -pnm1[:, i, 1:]
            Q_i = torch.concat([Q0_i[:, None], Qp_i], dim=1)

            # Boost p_i and Q_i along Q
            p_out[:, i] = boost(pnm1[:, i], Q)
            Q = boost(Q_i, Q)

        # Define final particle
        p_out[:, self.nparticles - 1] = Q

        # Get massless phase-space weights
        torch_ones = torch.ones((r.shape[0],), device=r.device)
        w0 = torch_ones * self._massles_weight(e_cm)

        # construct initial state momenta
        p_in = build_p_in(e_cm)

        if self.masses is not None:
            # match dimensions of masses
            m = self.masses[None, ...].to(device=r.device)

            # solve for xi in massive case, see Ref. [1]
            xi = get_xi_parameter(p_out[:, :, 0], m)

            # Make momenta massive
            xi = xi[:, None, None]
            k_out = torch.empty_like(p_out)
            k_out[:, :, 0] = sqrt(m**2 + xi[:, :, 0] ** 2 * p_out[:, :, 0] ** 2)
            k_out[:, :, 1:] = xi * p_out[:, :, 1:]
            # Get massive density corr. factor
            w_m = self._massive_weight(k_out, p_out, xi[:, 0, 0])

            p_ext = torch.cat([p_in, k_out], dim=1)
            return (p_ext,), w_m * w0

        p_ext = torch.cat([p_in, p_out], dim=1)
        return (p_ext,), w0

    def map_inverse(self, inputs: TensorList, condition=None):
        """Map from momenta to random numbers

        Args:
            inputs: list of tensors [p_ext]
                p_ext: external momenta with shape=(b,n+2,4)

        Returns:
            r (Tensor): random numbers with shape=(b,3*n-4)
            e_cm (Tensor): COM energy with shape=(b,)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        # Get input momenta
        p_ext = inputs[0]
        k = p_ext[:, 2:]
        e_cm = k.sum(dim=1)
        w0 = self._massles_weight(e_cm)

        # Make momenta massless before going back to random numbers
        p = torch.empty((k.shape[0], self.nparticles, 4), device=p_ext.device)
        if self.masses is not None:
            # Define masses
            m = self.masses[None, ...]
            # solve for xi in massive case, see Ref. [1], here analytic result possible!
            xi = torch.sum(sqrt(k[:, :, 0] ** 2 - m**2), dim=-1) / e_cm

            # Make them massless
            xi = xi[:, None, None]
            p[:, :, 0] = torch.sqrt(k[:, :, 0] ** 2 - m**2) / xi[:, :, 0]
            p[:, :, 1:] = k[:, :, 1:] / xi
            wm = self._massive_weight(k, p, xi[:, 0, 0])
        else:
            xi = None
            wm = 1.0
            p[:, :, 0] = k[:, :, 0]
            p[:, :, 1:] = k[:, :, 1:]

        # Get random numbers associated to the intermediate masses
        # have shapees (b, n-1)
        P = torch.cumsum(p.flip(1), dim=1)[:, 1:]  # has shape (b, n-1)
        M = mass(P)
        # have shapes (b, n-2)
        um = (M[:, :-1] / M[:, 1:]).flip(1)
        iarray = torch.arange(2, self.nparticles, device=p.device)[None, :]
        uc = self.nparticles + 1 - iarray
        uexp = 2 * (self.nparticles - iarray)
        ru = uc * um**uexp - (uc - 1) * um ** (uexp + 2)

        # Get the angles in correct frames
        # Here: do all boosts in one go
        Q = P.flip(1)
        p_prime = boost(p[:, :-1], Q, inverse=True)
        pmag = sqrt(esquare(p_prime[..., 1:]))
        costheta = p_prime[..., 3] / pmag
        phi = atan2(p_prime[..., 2], p_prime[..., 1])

        # Define the random numbers
        rcos = 0.5 * (costheta + 1.0)
        rphi = phi / (2 * pi) + (phi < 0)

        # Concat angular and energy random numbers
        r = torch.cat([ru, rcos, rphi], dim=-1)
        return (r, e_cm), 1.0 / w0 / wm

    def _massles_weight(self, e_cm):
        w0 = (
            (pi / 2.0) ** (self.nparticles - 1)
            * e_cm ** (2 * self.nparticles - 4)
            / (gamma(self.nparticles) * gamma(self.nparticles - 1))
        )
        return w0 * self.pi_factors

    def _massive_weight(
        self,
        k: torch.Tensor,
        p: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            k (torch.Tensor): massive momenta in shape=(b,n,4)
            p (torch.Tensor): massless momenta in shape=(b,n,4)
            xi (torch.Tensor, Optional): shift variable with shape=(b,)

        Returns:
            torch.Tensor: massive weight
        """
        # get correction factor for massive ones
        ks2 = k[:, :, 1] ** 2 + k[:, :, 2] ** 2 + k[:, :, 3] ** 2
        ps2 = p[:, :, 1] ** 2 + p[:, :, 2] ** 2 + p[:, :, 3] ** 2
        k0 = k[:, :, 0]
        p0 = p[:, :, 0]
        w_M = (
            xi ** (3 * self.nparticles - 3)
            * torch.prod(p0 / k0, dim=1)
            * torch.sum(ps2 / p0, dim=1)
            / torch.sum(ks2 / k0, dim=1)
        )
        return w_M

    def density(self, inputs, condition=None, inverse=False):
        del condition
        if inverse:
            _, gs_inv = self.map_inverse(self, inputs)
            return gs_inv

        _, gs = self.map(self, inputs)
        return gs


class Mahambo(PhaseSpaceMapping):
    """
    (Massive) RamboOnDiet including luminosity sampling
    for hadron colliders.
    """

    def __init__(
        self,
        e_beam: float,
        nparticles: int,
        lumi_func: str = "non-resonant",
        lumi_mass: Tensor = None,
        lumi_width: Tensor = None,
        masses: list[float] = None,
        e_min: float = 0.0,
    ):
        self.e_beam = e_beam
        self.nparticles = nparticles

        if masses is not None:
            self.masses = torch.tensor(masses)
            assert len(self.masses) == self.nparticles
            self.e_min = torch.max(self.masses.sum(), torch.tensor(e_min))
        else:
            self.masses = masses
            self.e_min = torch.tensor(e_min)

        dims_in = [(3 * nparticles - 2,)]
        dims_out = [(nparticles + 2, 4), (2,)]
        super().__init__(dims_in, dims_out)

        self.s_lab = torch.tensor(4 * e_beam**2)
        self.shat_min = self.e_min**2
        self.shat_max = self.s_lab

        if lumi_func == "non-resonant":
            self.luminosity = Luminosity(self.s_lab, self.shat_min, self.shat_max)
        elif lumi_func == "resonant":
            self.luminosity = ResonantLuminosity(
                self.s_lab, lumi_mass, lumi_width, self.shat_min, self.shat_max
            )
        else:
            self.luminosity = FlatLuminosity(self.s_lab, self.shat_min, self.shat_max)

        # Already includes correct factors of 2Pi
        self.rambo = RamboOnDiet(nparticles, masses, check_emin=False)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs: list of tensors [r]
                r: random numbers with shape=(b,3*n-2)

        Returns:
            p_lab (Tensor): external momenta (lab frame) with shape=(b,n+2,4)
            x1x2 (Tensor): pdf fractions with shape=(b,2)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r = inputs[0]
        r_lumi, r_rambo = r[:, :2], r[:, 2:]

        # Get x1 and x2
        (x1x2,), lumi_det = self.luminosity.map([r_lumi])

        # Get rambo momenta
        e_cm = sqrt(x1x2.prod(dim=1) * self.s_lab)
        (p_com,), w_rambo = self.rambo.map([r_rambo, e_cm])

        # Get output momenta an full ps-weight
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]  # needs shape (b,1)
        p_lab = boost_beam(p_com, rap)
        ps_weight = lumi_det * w_rambo

        return (p_lab, x1x2), ps_weight

    def map_inverse(self, inputs: TensorList, condition=None):
        """Map from momenta to random numbers

        Args:
            inputs: list of tensors [p_lab, x1x2]
                p_lab: external momenta with shape=(b,n+2,4)
                x1x2: pdf fractions with shape=(b,2)

        Returns:
            r (Tensor): random numbers with shape=(b,3*n-2)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        p_lab = inputs[0]
        x1x2 = inputs[1]

        # Get rapidities
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]

        # Get lumi rands
        (r_lumi,), det_lumi_inv = self.luminosity.map_inverse([x1x2])

        # boost into com frame
        p_com = boost_beam(p_lab, rap, inverse=True)

        # Get rambo random numbers
        (r_rambo, _), w_rambo_inv = self.rambo.map_inverse([p_com])

        # pack all together and get full weight
        r = torch.concat([r_lumi, r_rambo], dim=1)
        inv_ps_weight = w_rambo_inv * det_lumi_inv

        return (r,), inv_ps_weight

    def density(self, inputs, condition=None, inverse=False):
        del condition
        if inverse:
            _, gs_inv = self.map_inverse(self, inputs)
            return gs_inv

        _, gs = self.map(self, inputs)
        return gs


class AutoregressiveMahambo(PhaseSpaceMapping):
    """
    (Massive) RamboOnDiet including luminosity sampling
    for hadron colliders with trainable autoregressive blocks.
    """

    def __init__(
        self,
        e_beam: float,
        nparticles: int,
        bins: int = 3,
        lumi_func: str = "non-resonant",
        lumi_mass: Tensor = None,
        lumi_width: Tensor = None,
        masses: list[float] = None,
        e_min: float = 0.0,
    ):
        self.e_beam = e_beam
        self.nparticles = nparticles

        if masses is not None:
            self.masses = torch.tensor(masses)
            assert len(self.masses) == self.nparticles
            self.e_min = torch.max(self.masses.sum(), torch.tensor(e_min))
        else:
            self.masses = masses
            self.e_min = torch.tensor(e_min)

        dims_in = [(3 * nparticles - 2,)]
        dims_out = [(nparticles + 2, 4), (2,)]
        super().__init__(dims_in, dims_out)

        self.s_lab = torch.tensor(4 * e_beam**2)
        self.shat_min = self.e_min**2
        self.shat_max = self.s_lab

        # Define luminosity and its correlations
        #

        if lumi_func == "non-resonant":
            luminosity = Luminosity(self.s_lab, self.shat_min, self.shat_max)
        elif lumi_func == "resonant":
            luminosity = ResonantLuminosity(
                self.s_lab, lumi_mass, lumi_width, self.shat_min, self.shat_max
            )
        else:
            luminosity = FlatLuminosity(self.s_lab, self.shat_min, self.shat_max)

        self.luminosity = SplineMapping(
            mapping=luminosity,
            flow_dims_c=[],
            correlations="all",
            bins=bins,
        )

        # Define Rambo part (2->n) block
        # with autoregressive correlations
        # and with overal s_hat/s condition
        #

        rambo = RamboOnDiet(nparticles, masses, check_emin=False)
        periodic_indices = list(range(2 * self.nparticles - 3, 3 * self.nparticles - 4))
        self.rambo = SplineMapping(
            mapping=rambo,
            flow_dims_c=[(1,)],
            correlations="autoregressive",
            periodic=periodic_indices,
            bins=bins,
        )

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs: list of tensors [r]
                r: random numbers with shape=(b,3*n-2)

        Returns:
            p_lab (Tensor): external momenta (lab frame) with shape=(b,n+2,4)
            x1x2 (Tensor): pdf fractions with shape=(b,2)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r = inputs[0]
        r_lumi, r_rambo = r[:, :2], r[:, 2:]

        # Get x1 and x2
        (x1x2,), lumi_det = self.luminosity.map([r_lumi])

        # Get rambo momenta
        e_cm = sqrt(x1x2.prod(dim=1) * self.s_lab)
        sq_tau = sqrt(x1x2.prod(dim=1, keepdim=True))
        (p_com,), w_rambo = self.rambo.map([r_rambo, e_cm], condition=[sq_tau])

        # Get output momenta an full ps-weight
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]  # needs shape (b,1)
        p_lab = boost_beam(p_com, rap)
        ps_weight = lumi_det * w_rambo

        return (p_lab, x1x2), ps_weight

    def map_inverse(self, inputs: TensorList, condition=None):
        """Map from momenta to random numbers

        Args:
            inputs: list of tensors [p_lab, x1x2]
                p_lab: external momenta with shape=(b,n+2,4)
                x1x2: pdf fractions with shape=(b,2)

        Returns:
            r (Tensor): random numbers with shape=(b,3*n-2)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        p_lab = inputs[0]
        x1x2 = inputs[1]

        # Get rapidities
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]
        sq_tau = sqrt(x1x2.prod(dim=1, keepdim=True))

        # Get lumi rands
        (r_lumi,), det_lumi_inv = self.luminosity.map_inverse([x1x2])

        # boost into com frame
        p_com = boost_beam(p_lab, rap, inverse=True)

        # Get rambo random numbers
        (r_rambo, _), w_rambo_inv = self.rambo.map_inverse([p_com], condition=[sq_tau])

        # pack all together and get full weight
        r = torch.concat([r_lumi, r_rambo], dim=1)
        inv_ps_weight = w_rambo_inv * det_lumi_inv

        return (r,), inv_ps_weight

    def density(self, inputs, condition=None, inverse=False):
        del condition
        if inverse:
            _, gs_inv = self.map_inverse(self, inputs)
            return gs_inv

        _, gs = self.map(self, inputs)
        return gs


# ------------------------------------------
# Phase Space Generator Blocks
# ------------------------------------------
#   has outputs (p_in, p_out)


class tRamboBlock(PhaseSpaceMapping):
    """Rambo on Diet algorithm as presented in
        [2] Rambo on diet - https://arxiv.org/abs/1308.2922

    We allow for external masses as input parameter to combine this
    as global t-channel block with additional s-channel decays.
    """

    def __init__(
        self,
        nparticles: int,
        check_emin: bool = False,
    ):
        self.nparticles = nparticles
        self.check_emin = check_emin

        dims_in = [(3 * nparticles - 4,), (), (nparticles,)]
        dims_out = [(nparticles, 4)]

        super().__init__(dims_in, dims_out)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs: list of tensors [r, e_cm, m_out]
                r: random numbers with shape=(b,3*n-4)
                e_cm: COM energy with shape=(b,) or shape=()
                m_out: (virtual, time-like) masses of outgoing particles with shape=(b,n)

        Returns:
            p_in (Tensor): incoming momenta with shape=(b,2,4)
            p_ext (Tensor): external momenta with shape=(b,n,4)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r = inputs[0]  # has dims (b,3*n-4)
        e_cm = inputs[1]  # has dims (b,) or ()
        m_out = inputs[2]  # has dims (b,n)

        # Check if the partonic COM energy is large enough
        # Only for debugging, maybe remove later
        with torch.no_grad():
            if self.check_emin:
                e_min = m_out.sum(dim=1)
                if torch.any(e_cm <= e_min):
                    raise ValueError(
                        f"partonic COM energy needs to be larger than sum of external masses!"
                    )

        # Do rambo on diet loop
        # prepare momenta
        p_out = torch.empty((r.shape[0], self.nparticles, 4), device=r.device)

        # split random numbers in energy and angular
        ru, romega = r[:, : self.nparticles - 2], r[:, self.nparticles - 2 :]

        # Solve rambo equation numerically for all u directly
        # u has shape=(b, nparticles - 2)
        u = get_u_parameter(ru)

        # split into rcos and rphi
        rcos, rphi = romega.split(self.nparticles - 1, dim=1)

        # Define all angles vectorized
        cos_theta = 2 * rcos - 1
        phi = 2 * pi * rphi

        # Define intermediate masses
        M = torch.zeros((r.shape[0], self.nparticles))
        M[:, 0] = e_cm
        M[:, 1:-1] = torch.cumprod(u, dim=1) * e_cm[:, None]

        # Define first n-1 energies
        # gets shape (b, nparticles - 1)
        q = 4 * M[:, :-1] * two_body_decay_factor(M[:, :-1], M[:, 1:], 0)

        # Define first (n-1) particles
        pnm1 = map_fourvector_rambo_diet(q, cos_theta, phi)

        # Define Qs
        Q = e_cm[:, None] * torch.tile(torch.tensor([1, 0, 0, 0]), (r.shape[0], 1))

        # Define loop over (n-1 particles) boosts
        for i in range(self.nparticles - 1):
            # Define Qi
            Q0_i = sqrt(q[:, i] ** 2 + M[:, i + 1] ** 2)
            Qp_i = -pnm1[:, i, 1:]
            Q_i = torch.concat([Q0_i[:, None], Qp_i], dim=1)

            # Boost p_i and Q_i along Q
            p_out[:, i] = boost(pnm1[:, i], Q)
            Q = boost(Q_i, Q)

        # Define final particle
        p_out[:, self.nparticles - 1] = Q

        # Get massless phase-space weights
        torch_ones = torch.ones((r.shape[0],), device=r.device)
        w0 = torch_ones * self._massles_weight(e_cm)

        # construct initial state momenta
        p_in = build_p_in(e_cm)

        # solve for xi in massive case, see Ref. [1]
        xi = get_xi_parameter(p_out[:, :, 0], m_out)

        # Make momenta massive
        xi = xi[:, None, None]
        k_out = torch.empty_like(p_out)
        k_out[:, :, 0] = sqrt(m_out**2 + xi[:, :, 0] ** 2 * p_out[:, :, 0] ** 2)
        k_out[:, :, 1:] = xi * p_out[:, :, 1:]

        # Get massive density corr. factor
        w_m = self._massive_weight(k_out, p_out, xi[:, 0, 0])

        return (p_in, k_out), w_m * w0

    def map_inverse(self, inputs: TensorList, condition=None):
        """Map from momenta to random numbers

        Args:
            inputs: list of tensors [p_in, p_out]
                p_in: incoming momenta with shape=(b,2,4)
                p_out: outgoing momenta with shape=(b,n,4)

        Returns:
            r (Tensor): random numbers with shape=(b,3*n-4)
            e_cm (Tensor): COM energy with shape=(b,)
            m_out (Tensor): (virtual, time-like) masses  of outgoing particles with shape=(b,n)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        # Get input momenta
        p_in, k = inputs
        e_cm = sqrt_shat(p_in)
        m_out = mass(k)  # has shape (b,n)
        w0 = self._massles_weight(e_cm)

        # Make momenta massless before going back to random numbers
        p = torch.empty((k.shape[0], self.nparticles, 4), device=k.device)
        # solve for xi in massive case, see Ref. [1], here analytic result possible!
        xi = torch.sum(sqrt(k[:, :, 0] ** 2 - m_out**2), dim=-1) / e_cm

        # Make them massless
        xi = xi[:, None, None]
        p[:, :, 0] = torch.sqrt(k[:, :, 0] ** 2 - m_out**2) / xi[:, :, 0]
        p[:, :, 1:] = k[:, :, 1:] / xi
        wm = self._massive_weight(k, p, xi[:, 0, 0])

        # Get random numbers associated to the intermediate masses
        # have shapees (b, n-1)
        P = torch.cumsum(p.flip(1), dim=1)[:, 1:]  # has shape (b, n-1)
        M = mass(P)
        # have shapes (b, n-2)
        um = (M[:, :-1] / M[:, 1:]).flip(1)
        iarray = torch.arange(2, self.nparticles, device=p.device)[None, :]
        uc = self.nparticles + 1 - iarray
        uexp = 2 * (self.nparticles - iarray)
        ru = uc * um**uexp - (uc - 1) * um ** (uexp + 2)

        # Get the angles in correct frames
        # Here: do all boosts in one go
        Q = P.flip(1)
        p_prime = boost(p[:, :-1], Q, inverse=True)
        pmag = sqrt(esquare(p_prime[..., 1:]))
        costheta = p_prime[..., 3] / pmag
        phi = atan2(p_prime[..., 2], p_prime[..., 1])

        # Define the random numbers
        rcos = 0.5 * (costheta + 1.0)
        rphi = phi / (2 * pi) + (phi < 0)

        # Concat angular and energy random numbers
        r = torch.cat([ru, rcos, rphi], dim=-1)
        return (r, e_cm, m_out), 1 / wm / w0

    def _massles_weight(self, e_cm):
        w0 = (
            (pi / 2.0) ** (self.nparticles - 1)
            * e_cm ** (2 * self.nparticles - 4)
            / (gamma(self.nparticles) * gamma(self.nparticles - 1))
        )
        return w0

    def _massive_weight(
        self,
        k: torch.Tensor,
        p: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            k (torch.Tensor): massive momenta in shape=(b,n,4)
            p (torch.Tensor): massless momenta in shape=(b,n,4)
            xi (torch.Tensor, Optional): shift variable with shape=(b,)

        Returns:
            torch.Tensor: massive weight
        """
        # get correction factor for massive ones
        ks2 = k[:, :, 1] ** 2 + k[:, :, 2] ** 2 + k[:, :, 3] ** 2
        ps2 = p[:, :, 1] ** 2 + p[:, :, 2] ** 2 + p[:, :, 3] ** 2
        k0 = k[:, :, 0]
        p0 = p[:, :, 0]
        w_M = (
            xi ** (3 * self.nparticles - 3)
            * torch.prod(p0 / k0, dim=1)
            * torch.sum(ps2 / p0, dim=1)
            / torch.sum(ks2 / k0, dim=1)
        )
        return w_M

    def density(self, inputs, condition=None, inverse=False):
        del condition
        if inverse:
            _, gs_inv = self.map_inverse(self, inputs)
            return gs_inv

        _, gs = self.map(self, inputs)
        return gs


class tMahamboBlock(PhaseSpaceMapping):
    """
    (Massive) RamboOnDiet including luminosity sampling
    for hadron colliders.

    We allow for external masses as input parameter to combine this
    as global t-channel Block with additional s-channel decays.
    """

    def __init__(
        self,
        e_beam: float,
        nparticles: int,
        e_hat_min: float,
        lumi_func: str = "non-resonant",
        lumi_mass: Tensor = None,
        lumi_width: Tensor = None,
    ):
        self.e_beam = e_beam
        self.nparticles = nparticles

        dims_in = [(3 * nparticles - 2,)]
        dims_out = [(nparticles, 4), (2,)]

        self.s_lab = torch.tensor(4 * e_beam**2)
        self.shat_min = torch.tensor(e_hat_min**2)
        self.shat_max = self.s_lab

        if lumi_func == "non-resonant":
            self.luminosity = Luminosity(self.s_lab, self.shat_min, self.shat_max)
        elif lumi_func == "resonant":
            self.luminosity = ResonantLuminosity(
                self.s_lab, lumi_mass, lumi_width, self.shat_min, self.shat_max
            )
        else:
            self.luminosity = FlatLuminosity(self.s_lab, self.shat_min, self.shat_max)

        self.rambo = tRamboBlock(nparticles, check_emin=False)

        super().__init__(dims_in, dims_out)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs: list of tensors [r, m_out]
                r: random numbers with shape=(b,3*n-2)
                m_out: (virtual) masses of outgoing particles with shape=(b,n)

        Returns:
            p_lab (Tensor): external momenta (lab frame) with shape=(b,n+2,4)
            x1x2 (Tensor): pdf fractions with shape=(b,2)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r = inputs[0]  # has dims (b,3*n-2)
        m_out = inputs[1]  # has dims (b,n)
        r_lumi, r_rambo = r[:, :2], r[:, 2:]

        # Get x1 and x2
        (x1x2,), lumi_det = self.luminosity.map([r_lumi])

        # Get rambo momenta
        e_cm = sqrt(x1x2.prod(dim=1) * self.s_lab)
        (p_in, p_out), w_rambo = self.rambo.map([r_rambo, e_cm, m_out])

        # Get output momenta an full ps-weight
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]  # needs shape (b,1)
        p_in_lab = boost_beam(p_in, rap)
        p_out_lab = boost_beam(p_out, rap)
        ps_weight = lumi_det * w_rambo

        return (p_in_lab, p_out_lab), ps_weight

    def map_inverse(self, inputs: TensorList, condition=None):
        """Map from momenta to random numbers

        Args:
            inputs: list of tensors [p_in_lab, p_out_lab]
                p_in_lab: incoming LAB momenta with shape=(b,2,4)
                p_out_lab: outgoing LAB momenta with shape=(b,n,4)

        Returns:
            r (Tensor): random numbers with shape=(b,3*n-2)
            m_out (Tensor): (virtual) masses of outgoing particles with shape=(b,n)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        p_in_lab = inputs[0]
        p_out_lab = inputs[1]
        sqrt_s = sqrt_shat(p_in_lab)
        x1x2 = pin_to_x1x2(p_in_lab, sqrt_s)

        # Get rapidities
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]

        # Get lumi rands
        (r_lumi,), det_lumi_inv = self.luminosity.map_inverse([x1x2])

        # boost into com frame
        p_in_com = boost_beam(p_in_lab, rap, inverse=True)
        p_out_com = boost_beam(p_out_lab, rap, inverse=True)

        # Get rambo random numbers
        (r_rambo, _, m_out), w_rambo_inv = self.rambo.map_inverse([p_in_com, p_out_com])

        # pack all together and get full weight
        r = torch.concat([r_lumi, r_rambo], dim=1)
        inv_ps_weight = w_rambo_inv * det_lumi_inv

        return (r, m_out), inv_ps_weight

    def density(self, inputs, condition=None, inverse=False):
        del condition
        if inverse:
            _, gs_inv = self.map_inverse(self, inputs)
            return gs_inv

        _, gs = self.map(self, inputs)
        return gs


class tAutoregressiveMahamboBlock(PhaseSpaceMapping):
    """
    (Massive) RamboOnDiet including luminosity sampling
    for hadron colliders with trainable autorgressive blocks.

    We allow for external masses as input parameter to combine this
    as global t-channel Block with additional s-channel decays.
    """

    def __init__(
        self,
        e_beam: float,
        nparticles: int,
        e_hat_min: float,
        bins: int = 3,
        lumi_func: str = "non-resonant",
        lumi_mass: Tensor = None,
        lumi_width: Tensor = None,
    ):
        self.e_beam = e_beam
        self.nparticles = nparticles

        dims_in = [(3 * nparticles - 2,)]
        dims_out = [(nparticles, 4), (2,)]

        self.s_lab = torch.tensor(4 * e_beam**2)
        self.shat_min = torch.tensor(e_hat_min**2)
        self.shat_max = self.s_lab

        # Define luminosity and its correlations
        #

        if lumi_func == "non-resonant":
            luminosity = Luminosity(self.s_lab, self.shat_min, self.shat_max)
        elif lumi_func == "resonant":
            luminosity = ResonantLuminosity(
                self.s_lab, lumi_mass, lumi_width, self.shat_min, self.shat_max
            )
        else:
            luminosity = FlatLuminosity(self.s_lab, self.shat_min, self.shat_max)

        self.luminosity = SplineMapping(
            mapping=luminosity,
            flow_dims_c=[],
            correlations="all",
            bins=bins,
        )

        # Define Rambo part (2->n) block
        # with autoregressive correlations
        # and with overal s_hat/s condition
        #

        rambo = tRamboBlock(nparticles, check_emin=False)
        # get periodic indices
        periodic_indices = list(range(2 * self.nparticles - 3, 3 * self.nparticles - 4))
        self.rambo = SplineMapping(
            mapping=rambo,
            flow_dims_c=[(1,)],
            correlations="autoregressive",
            periodic=periodic_indices,
            bins=bins,
        )

        super().__init__(dims_in, dims_out)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs: list of tensors [r, m_out]
                r: random numbers with shape=(b,3*n-2)
                m_out: (virtual) masses of outgoing particles with shape=(b,n)

        Returns:
            p_lab (Tensor): external momenta (lab frame) with shape=(b,n+2,4)
            x1x2 (Tensor): pdf fractions with shape=(b,2)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        r = inputs[0]  # has dims (b,3*n-2)
        m_out = inputs[1]  # has dims (b,n)
        r_lumi, r_rambo = r[:, :2], r[:, 2:]

        # Get x1 and x2
        (x1x2,), lumi_det = self.luminosity.map([r_lumi])

        # Get rambo momenta
        e_cm = sqrt(x1x2.prod(dim=1) * self.s_lab)
        sq_tau = sqrt(x1x2.prod(dim=1, keepdim=True))
        (p_in, p_out), w_rambo = self.rambo.map(
            [r_rambo, e_cm, m_out], condition=[sq_tau]
        )

        # Get output momenta an full ps-weight
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]  # needs shape (b,1)
        p_in_lab = boost_beam(p_in, rap)
        p_out_lab = boost_beam(p_out, rap)
        ps_weight = lumi_det * w_rambo

        return (p_in_lab, p_out_lab), ps_weight

    def map_inverse(self, inputs: TensorList, condition=None):
        """Map from momenta to random numbers

        Args:
            inputs: list of tensors [p_in_lab, p_out_lab]
                p_in_lab: incoming LAB momenta with shape=(b,2,4)
                p_out_lab: outgoing LAB momenta with shape=(b,n,4)

        Returns:
            r (Tensor): random numbers with shape=(b,3*n-2)
            m_out (Tensor): (virtual) masses of outgoing particles with shape=(b,n)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition
        p_in_lab = inputs[0]
        p_out_lab = inputs[1]
        sqrt_s = sqrt_shat(p_in_lab)
        x1x2 = pin_to_x1x2(p_in_lab, sqrt_s)

        # Get rapidities
        rap = 0.5 * log(x1x2[:, 0] / x1x2[:, 1])[:, None]
        sq_tau = sqrt(x1x2.prod(dim=1, keepdim=True))

        # Get lumi rands
        (r_lumi,), det_lumi_inv = self.luminosity.map_inverse([x1x2])

        # boost into com frame
        p_in_com = boost_beam(p_in_lab, rap, inverse=True)
        p_out_com = boost_beam(p_out_lab, rap, inverse=True)

        # Get rambo random numbers
        (r_rambo, _, m_out), w_rambo_inv = self.rambo.map_inverse(
            [p_in_com, p_out_com], condition=[sq_tau]
        )

        # pack all together and get full weight
        r = torch.concat([r_lumi, r_rambo], dim=1)
        inv_ps_weight = w_rambo_inv * det_lumi_inv

        return (r, m_out), inv_ps_weight

    def density(self, inputs, condition=None, inverse=False):
        del condition
        if inverse:
            _, gs_inv = self.map_inverse(self, inputs)
            return gs_inv

        _, gs = self.map(self, inputs)
        return gs
