""" Implement invariant mappings.
    Bases on the mappings described in
    [1] https://freidok.uni-freiburg.de/data/154629
    [2] https://arxiv.org/abs/hep-ph/0008033
"""


from torch import Tensor
from .base import PhaseSpaceMapping, TensorList
from .functional.propagators import (
    uniform_propagator,
    breit_wigner_propagator,
    massles_propagator,
    massless_propagator_nu,
    stable_propagator,
    stable_propagator_nu,
)


class _Invariants(PhaseSpaceMapping):
    def __init__(self):
        """Fix the dimensions of all invariant mappings

        Dimensions:
            dims_in: random number r with shape=(b,1)
            dims_out: invariant s with shape=(b,1)
            dims_c: s_min and s_max with shapes=(b,)
        """
        super().__init__(dims_in=[(1,)], dims_out=[(1,)], dims_c=[(), ()])


class UniformInvariantBlock(_Invariants):
    """Implements uniform sampling of invariants"""

    def __init__(self):
        """see parent docstring"""
        super().__init__()

    def map(self, inputs: TensorList, condition: TensorList):
        r = inputs[0]
        smin, smax = condition[0], condition[1]
        s, det = uniform_propagator(r, smin, smax)
        return (s,), det

    def map_inverse(self, inputs: TensorList, condition: TensorList):
        """Map from random number to s"""
        s = inputs[0]
        smin, smax = condition[0], condition[1]
        r, det = uniform_propagator(s, smin, smax, inverse=True)
        return (r,), det

    def density(self, inputs: TensorList, condition: TensorList, inverse=False):
        """Inverse map from s onto random number"""
        del inputs
        smin, smax = condition[0], condition[1]
        gs = smax - smin
        det = 1 / gs if inverse else gs
        return det


class BreitWignerInvariantBlock(_Invariants):
    """
    Performs the Breit-Wigner mapping as described in
        [2] https://arxiv.org/abs/hep-ph/0008033
        [3] https://freidok.uni-freiburg.de/data/154629
    """

    def __init__(
        self,
        mass: Tensor,
        width: Tensor,
    ):
        """
        Args:
            mass (Tensor): Mass of propagator particle
            width (Tensor): width of propagator particle
        """
        super().__init__()
        self.mass = mass
        self.width = width

    def map(self, inputs: TensorList, condition: TensorList):
        """Map from random number to s"""
        r = inputs[0]
        smin, smax = condition[0], condition[1]
        s, det = breit_wigner_propagator(r, self.mass, self.width, smin, smax)
        return (s,), det

    def map_inverse(self, inputs: TensorList, condition: TensorList):
        """Map from random number to s"""
        s = inputs[0]
        smin, smax = condition[0], condition[1]
        r, det = breit_wigner_propagator(
            s,
            self.mass,
            self.width,
            smin,
            smax,
            inverse=True,
        )
        return (r,), det

    def density(self, inputs: TensorList, condition: TensorList, inverse=False):
        """Inverse map from s onto random number"""
        smin, smax = condition[0], condition[1]
        _, det = breit_wigner_propagator(
            inputs[0],
            self.mass,
            self.width,
            smin,
            smax,
            inverse=inverse,
        )
        return det


class StableInvariantBlock(_Invariants):
    """
    Performs the massive, vanishing width propagator as described in
        [2] https://arxiv.org/abs/hep-ph/0008033
        [3] https://freidok.uni-freiburg.de/data/154629
    """

    def __init__(
        self,
        mass: Tensor,
        nu: float = 1.4,
    ):
        """
        Args:
            mass (Tensor): Mass of propagator particle
            nu (float, optional): controls nu parameter
        """
        super().__init__()
        self.mass = mass
        self.nu = nu

        if nu == 1.0:
            self.prop_function = stable_propagator
        else:
            self.prop_function = stable_propagator_nu

    def map(self, inputs: TensorList, condition: TensorList):
        """Map from random number to s"""
        r = inputs[0]
        smin, smax = condition[0], condition[1]
        s, det = self.prop_function(r, self.mass, smin, smax, self.nu)
        return (s,), det

    def map_inverse(self, inputs: TensorList, condition: TensorList):
        """Map from random number to s"""
        s = inputs[0]
        smin, smax = condition[0], condition[1]
        r, det = self.prop_function(s, self.mass, smin, smax, self.nu, inverse=True)
        return (r,), det

    def density(self, inputs: TensorList, condition: TensorList, inverse=False):
        """Inverse map from s onto random number"""
        smin, smax = condition[0], condition[1]
        _, det = self.prop_function(
            inputs[0], self.mass, smin, smax, self.nu, inverse=inverse
        )
        return det


class MasslessInvariantBlock(_Invariants):
    """
    Performs the massless propagator as described in
        [2] https://arxiv.org/abs/hep-ph/0008033
        [3] https://freidok.uni-freiburg.de/data/154629
    """

    def __init__(
        self,
        nu: float = 1.4,
    ):
        """
        Args:
            nu (float, optional): controls nu parameter
        """
        super().__init__()
        self.nu = nu

        if nu == 1.0:
            self.prop_function = massles_propagator
        else:
            self.prop_function = massless_propagator_nu

    def map(self, inputs: TensorList, condition: TensorList):
        """Map from random number to s"""
        r = inputs[0]
        smin, smax = condition[0], condition[1]
        s, det = self.prop_function(r, smin, smax, self.nu)
        return (s,), det

    def map_inverse(self, inputs: TensorList, condition: TensorList):
        """Map from random number to s"""
        s = inputs[0]
        smin, smax = condition[0], condition[1]
        r, det = self.prop_function(s, smin, smax, self.nu, inverse=True)
        return (r,), det

    def density(self, inputs: TensorList, condition: TensorList, inverse=False):
        """Inverse map from s onto random number"""
        smin, smax = condition[0], condition[1]
        _, det = self.prop_function(inputs[0], smin, smax, self.nu, inverse=inverse)
        return det
