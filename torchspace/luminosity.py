""" Implements mappings for the luminosity (PDF)
    integration. Partially based on methods presented in
    [1] https://freidok.uni-freiburg.de/data/154629
    [2] https://arxiv.org/abs/hep-ph/0206070v2
    [3] https://arxiv.org/abs/hep-ph/0008033
"""


from typing import Tuple
import torch
from torch import Tensor, log

from .base import PhaseSpaceMapping, TensorList, TensorTuple
from .functional.propagators import (
    breit_wigner_propagator,
    massless_propagator_nu,
    massles_propagator,
    uniform_propagator,
)


def r2tau_to_x1x2(r2: Tensor, tau: Tensor) -> Tuple[TensorTuple, Tensor]:
    x1 = tau**r2
    x2 = tau ** (1 - r2)
    det = log(tau)
    return (x1, x2), det.abs()


def x1x2_to_r2tau(x1: Tensor, x2: Tensor) -> Tuple[TensorTuple, Tensor]:
    tau = x1 * x2
    r2 = log(x1) / log(tau)
    det = 1 / log(tau)
    return (r2, tau), det.abs()


class _Luminosity(PhaseSpaceMapping):
    """
    Implements a base luminosity mapping
    """

    def __init__(
        self,
        s_lab: Tensor,
        shat_min: Tensor,
        shat_max: Tensor = None,
    ):
        """
        Args:
            s_lab: squared COM energy of the lab frame
            shat_min (Tensor): minimum s_hat with shape=()
            shat_max (Tensor): maximum s_hat with shape=(). Defaults to None.
                None means s_max = s_lab.
        """
        super().__init__(dims_in=[(2,)], dims_out=[(2,)], dims_c=None)
        self.s_lab = s_lab
        self.shat_min = shat_min
        self.shat_max = shat_max if shat_max is not None else s_lab

    def _shat_map(self, r1: Tensor):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide shat_map(...) method"
        )

    def _shat_map_inverse(self, shat: Tensor):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide shat_map_inverse(...) method"
        )

    # ==================================================
    #

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to x1 and x2

        Args:
            inputs: list of one tensor [r]
                r: random numbers with shape=(b,2)

        Returns:
            x1x2 (Tensor): parton fractions x1 and x2 with shape=(b,2)
            det (Tensor): log det of mapping with shape=(b,)
        """
        del condition
        r = inputs[0]
        r1, r2 = r[:, 0], r[:, 1]

        # map r1 to shat
        shat, s_det = self._shat_map(r1)

        # Define tau from shat
        tau = shat / self.s_lab
        tau_det = 1 / self.s_lab

        # Get x1x2 and its density
        (x1, x2), x_det = r2tau_to_x1x2(r2, tau)

        # Pack it all together
        x1x2 = torch.stack([x1, x2], dim=1)
        det = s_det * tau_det * x_det

        return (x1x2,), det

    def map_inverse(self, inputs: TensorList, condition=None):
        """Inverse map from x1,x2 to the random numbers

        Args:
            inputs (TensorList): list with only one tensor [x1x2]
                x1x2: parton fractions x1 and x2 with shape=(b,2)

        Returns:
            r (Tensor): random numbers with shape=(b,2)
            det (Tensor): log det of mapping with shape=(b,)
        """
        del condition
        x = inputs[0]
        x1, x2 = x[:, 0], x[:, 1]

        # Define tau variable
        (r2, tau), x_det_inv = x1x2_to_r2tau(x1, x2)

        # Get s_hat from tau
        # Define tau from shat
        shat = tau * self.s_lab
        tau_det_inv = self.s_lab

        # Get random number r1 from shat
        r1, s_det_inv = self._shat_map_inverse(shat)

        # Get the density and stack together
        det_inv = x_det_inv * tau_det_inv * s_det_inv
        r = torch.stack([r1, r2], dim=1)

        return (r,), det_inv

    def density(self, inputs: TensorList, condition=None, inverse=False):
        if inverse:
            _, det = self.map_inverse(inputs, condition)
            return det

        _, det = self.map(inputs, condition)
        return det


class Luminosity(_Luminosity):
    """
    Implement luminosity mapping as suggested in
        [1] https://freidok.uni-freiburg.de/data/154629
        [2] https://arxiv.org/abs/hep-ph/0206070v2
        [3] https://arxiv.org/abs/hep-ph/0008033

    which aims to smooth the overall 1/s dependency of the
    partonic cross section (due to the flux factor)

    ### Note:
    In contrast to [1] we also allow for an tau_max (or s_max)
    to potentially split up the luminosity integral into different regions
    as been done by MG5. Then, we use the massless mapping as suggested
    in [1-3]. This slightly alters the map as:

        [1:lumi] Eq.(H.5)
            tau = tau_min ** r1
        [1-3:massless]
            tau = tau_max ** r1 * tau_min ** (1-r1)

    result in some slight difference in the determinant.

    """

    def __init__(
        self,
        s_lab: Tensor,
        shat_min: Tensor,
        shat_max: Tensor = None,
        nu: float = 1.0,
    ):
        """
        Args:
            nu (float): nu parameter from [1-3]
        """
        super().__init__(s_lab, shat_min, shat_max)

        if nu == 1.0:
            self.shat_function = massles_propagator
        else:
            self.shat_function = massless_propagator_nu

    def _shat_map(self, r1: Tensor):
        shat, s_det = self.shat_function(
            r1,
            self.shat_min,
            self.shat_max,
        )
        return shat, s_det

    def _shat_map_inverse(self, shat: Tensor):
        r1, s_det_inv = self.shat_function(
            shat,
            self.shat_min,
            self.shat_max,
            inverse=True,
        )
        return r1, s_det_inv


class ResonantLuminosity(_Luminosity):
    r"""
    Implement luminosity mapping that takes into account resonant
    particles in the s-channel related to the partonic COM
    energy s_hat.

    As this might only be used in a regime s_hat in [M-n\Gamma, M+n\Gamma],
    this translates in both lower (tau_min) and upper limits (tau_max).
    """

    def __init__(
        self,
        s_lab: Tensor,
        mass: Tensor,
        width: Tensor,
        shat_min: Tensor,
        shat_max: Tensor = None,
    ):
        """
        Args:
            mass (Tensor):
            width (Tensor):
        """
        super().__init__(s_lab, shat_min, shat_max)
        self.mass = mass
        self.width = width

    def _shat_map(self, r1: Tensor):
        shat, s_det = breit_wigner_propagator(
            r1,
            self.mass,
            self.width,
            self.shat_min,
            self.shat_max,
        )
        return shat, s_det

    def _shat_map_inverse(self, shat: Tensor):
        r1, s_det_inv = breit_wigner_propagator(
            shat,
            self.mass,
            self.width,
            self.shat_min,
            self.shat_max,
            inverse=True,
        )
        return r1, s_det_inv


class FlatLuminosity(_Luminosity):
    """
    Implement luminosity mapping which maps out tau flat:
        tau = tau_min + (tau_max - tau_min) * r1
    """

    def __init__(
        self,
        s_lab: Tensor,
        shat_min: Tensor,
        shat_max: Tensor = None,
    ):
        super().__init__(s_lab, shat_min, shat_max)

    def _shat_map(self, r1: Tensor):
        shat, s_det = uniform_propagator(r1, self.shat_min, self.shat_max)
        return shat, s_det

    def _shat_map_inverse(self, shat: Tensor):
        r1, s_det_inv = uniform_propagator(
            shat,
            self.shat_min,
            self.shat_max,
            inverse=True,
        )
        return r1, s_det_inv
