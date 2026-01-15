from math import prod
from copy import copy

import torch
from torch import Tensor
import torch.nn as nn

from .base import PhaseSpaceMapping, TensorList, ShapeList
from .functional.splines import unconstrained_rational_quadratic_spline

class SplineMapping(PhaseSpaceMapping):
    """
    Learnable mapping based on RQ splines that wraps around other phase space mappings.
    It assumes that the first input of the wrapped mapping is a tensor with random numbers
    in the unit interval. Each component of the random numbers is transformed once using a
    RQ spline. In the simplest case, the spline bin widths/heights/derivatives are learnable
    parameters. The class also allows for more complex conditional transformations where
    the spline w/h/d are predicted by neural networks, making the mapping a minimal
    normalizing flow.
    """

    def __init__(
        self,
        mapping: PhaseSpaceMapping,
        flow_dims_c: ShapeList = [],
        extra_params_c: int = 0,
        correlations: str = "none",
        permutation: list[int] | None = None,
        layers: int = 3,
        hidden_dim: int = 32,
        bins: int = 10,
        activation: type[nn.Module] = nn.ReLU,
        bilinear: bool = True,
        periodic: list[int] = [],
        min_bin_width: float = 1e-3,
        min_bin_height: float = 1e-3,
        min_derivative: float = 1e-3,
    ):
        """
        Constructs spline mapping.

        Args:
            mapping: Phase space mapping that the learnable mapping wraps around. The first
                input to the mapping have to be random numbers from the unit interval.
            flow_dims_c: Dimensions of the tensors that the trainable mapping is
                conditioned on.
            extra_params_c: Trainable parameters that are used as extra conditional inputs to
                the trainable mapping. This argument is only useful when multiple mappings
                otherwise share the same parameters (see shared_mapping) as these are the only
                parameters that are different between the mappings, allowing for some amount
                of specialization.
            correlations: Only relevant for mappings with more than one random variable.
                If "none", all spline transformations are independent. If "autoregressive",
                the transformation of the n-th component is conditioned on the previous n-1
                component. If "all", each transformed component is conditioned on all other
                components.
            permutation: Specifies the order in which the transformations are performed. Only
                relevant if the correlation mode is "autoregressive" or "all".
            layers: number of subnet layers
            hidden_dim: number of nodes of the subnet hidden layers
            bins: number of spline bins
            activation: subnet activation function
            bilinear: build flow with bilinear operation instead of subnet
            periodic: list of component indices for which periodic splines are used
            min_bin_width: minimum spline bin width
            min_bin_height: minimum spline bin height
            min_derivative: minimum spline derivative
        """
        mapping_dims_c = [] if mapping.dims_c is None else mapping.dims_c
        dims_in = mapping.dims_in
        dims_out = mapping.dims_out
        dims_c = mapping_dims_c + flow_dims_c
        super().__init__(dims_in, dims_out, dims_c)

        (self.r_dim,) = dims_in[0]
        self.flat_c_dim = sum(prod(dim) for dim in flow_dims_c) + extra_params_c

        self.mapping_dims_c = mapping_dims_c
        self.flow_dims_c = flow_dims_c
        self.mapping = mapping
        self.correlations = correlations
        self.bins = bins
        self.layers = 1 if bilinear else layers
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.bilinear = bilinear
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative

        if permutation is None:
            permutation = list(range(self.r_dim))
        self.permutation = permutation
        self.inverse_permutation = [permutation.index(i) for i in range(self.r_dim)]
        self.periodic = [permutation[i] in periodic for i in range(self.r_dim)]

        self.submodules = nn.ModuleList()
        self.subparams = nn.ParameterList()
        if correlations == "none":
            self.subnets = [self._build_subnet(self.flat_c_dim) for _ in range(self.r_dim)]
        elif correlations == "autoregressive":
            self.subnets = [self._build_subnet(self.flat_c_dim + i) for i in range(self.r_dim)]
        elif correlations == "all":
            self.subnets = [
                self._build_subnet(self.flat_c_dim + self.r_dim - 1) for _ in range(self.r_dim)
            ]
        else:
            raise ValueError("Unknown correlation mode")

        if extra_params_c == 0:
            self.extra_params_c = None
        else:
            self.extra_params_c = nn.Parameter(torch.zeros(extra_params_c))

    def shared_mapping(self, mapping: PhaseSpaceMapping) -> "SplineMapping":
        """
        Returns a copy of the mapping, where only the wrapped mapping and (if extra_params_c
        is larger than 0), the additional conditional parameters are replaced. All other
        subnets and trainable parameters are shared between the two mappings.

        Args:
            mapping: mapping that is wrapped by the returned spline mapping

        Returns:
            shared: copied mapping
        """
        shared = copy(self)
        shared._parameters = copy(self._parameters)
        shared._modules = copy(self._modules)
        shared.mapping = mapping
        if shared.extra_params_c is not None:
            old_params_c = self.extra_params_c
            shared.extra_params_c = nn.Parameter(torch.zeros_like(self.extra_params_c))
        return shared

    def map(self, inputs: TensorList, condition=None):
        """
        Performs the forward mapping

        Args:
            inputs: list of at least one tensor
                r: random numbers with shape=(b, n_random)
                all other inputs are passed on to the wrapped mapping
            condition: list of tensors
                first: conditions that are passed on to the wrapped mapping
                then: conditions that are used as inputs to the subnets

        Returns:
            out: outputs of the wrapped mapping
            det (Tensor): log det of mapping with shape=(b,)
        """
        r, rest = inputs[0], inputs[1:]
        c_mapping, c_flow = self._split_condition(condition, r.shape[0])
        x, jac_flow = self._map_flow(r, c_flow, inverse=False)
        out, jac_mapping = self.mapping.map([x, *rest], c_mapping)
        return out, jac_flow * jac_mapping

    def map_inverse(self, inputs: TensorList, condition=None):
        """
        Performs the inverse mapping

        Args:
            inputs: list of tensors
                inputs of the wrapped mapping
            condition: list of tensors
                first: conditions that are passed on to the wrapped mapping
                then: conditions that are used as inputs to the subnets

        Returns:
            outputs of the wrapped mapping
                r: random numbers with shape=(b, n_random)
                all other outputs are passed on from the wrapped mapping
            det (Tensor): log det of mapping with shape=(b,)
        """
        c_mapping, c_flow = self._split_condition(condition, inputs[0].shape[0])
        (x, *rest), jac_mapping = self.mapping.map_inverse(inputs, c_mapping)
        r, jac_flow = self._map_flow(x, c_flow, inverse=True)
        return (r, *rest), jac_flow * jac_mapping

    def _build_subnet(
        self,
        dims_in: int,
    ) -> nn.Module | nn.Parameter:
        dims_out = 3 * self.bins + 1
        if dims_in == 0:
            param = nn.Parameter(torch.zeros(dims_out))
            self.subparams.append(param)
            return param

        if self.bilinear:
            dims_in = dims_in * (dims_in + 3) // 2

        modules = []
        layer_dim = dims_in
        for i in range(self.layers - 1):
            modules.append(nn.Linear(layer_dim, self.hidden_dim))
            modules.append(self.activation())
            layer_dim = self.hidden_dim
        last_layer = nn.Linear(layer_dim, dims_out)
        nn.init.zeros_(last_layer.weight)
        nn.init.zeros_(last_layer.bias)
        modules.append(last_layer)

        module = nn.Sequential(*modules)
        self.submodules.append(module)
        return module

    def _split_condition(self, condition: TensorList | None, batch_dim: int):
        if condition is None:
            condition = []
        if self.extra_params_c is not None:
            condition = [*condition, self.extra_params_c[None,:].expand(batch_dim, -1)]
        n_mappings_dims_c = len(self.mapping_dims_c)
        c_mapping = condition[:n_mappings_dims_c]
        if n_mappings_dims_c < len(condition):
            c_flow = torch.cat(
                [c.reshape(batch_dim, -1) for c in condition[n_mappings_dims_c:]], dim=1
            )
        else:
            c_flow = None
        return c_mapping, c_flow

    def _map_flow(self, x: Tensor, c: Tensor | None, inverse: bool) -> tuple[Tensor, Tensor]:
        if inverse:
            x = x[:, self.inverse_permutation]
        else:
            x = x[:, self.permutation]
        if c is None:
            c = x[:, :0]
        log_jac_all = 0.
        subnet_iter = enumerate(self.subnets)
        for i, subnet in (reversed(list(subnet_iter)) if inverse else subnet_iter):
            if isinstance(subnet, nn.Parameter):
                theta = subnet[None, :].expand(x.shape[0], -1)
            else:
                if self.correlations == "none":
                    c_all = c
                elif self.correlations == "autoregressive":
                    c_all = torch.cat((x[:, :i], c), dim=1)
                elif self.correlations == "all":
                    c_all = torch.cat((x[:, :i], x[:, i+1:], c), dim=1)

                if self.bilinear:
                    bilin = c_all[:,:,None] * c_all[:,None,:]
                    indices = torch.triu_indices(
                        c_all.shape[1], c_all.shape[1], device=c_all.device
                    )
                    c_all = torch.cat((c_all, bilin[:, indices[0], indices[1]]), dim=1)

                theta = subnet(c_all)

            x[:, i:i+1], log_jac = unconstrained_rational_quadratic_spline(
                inputs=x[:, i:i+1],
                theta=theta[:, None],
                rev=inverse,
                num_bins=self.bins,
                left=0.,
                right=1.,
                bottom=0.,
                top=1.,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
                periodic=self.periodic[i],
            )
            log_jac_all += log_jac

        if inverse:
            x = x[:, self.permutation]
        else:
            x = x[:, self.inverse_permutation]
        return x, log_jac_all.exp()

    def __deepcopy__(self, memo):
        return self
