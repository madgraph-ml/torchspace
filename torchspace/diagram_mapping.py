from __future__ import annotations

from dataclasses import dataclass, field
from math import pi

import torch
from torch import Tensor

from .base import PhaseSpaceMapping, TensorList
from .chili import tChiliBlock
from .functional.kinematics import boost_beam, lsquare, mass
from .functional.ps_utils import build_p_in
from .invariants import (
    BreitWignerInvariantBlock,
    MasslessInvariantBlock,
    StableInvariantBlock,
    UniformInvariantBlock,
)
from .luminosity import Luminosity, ResonantLuminosity
from .rambo import tRamboBlock
from .twoparticle import (
    TwoBodyDecayCOM,
    TwoBodyDecayLAB,
    TwoToTwoScatteringCOM,
    TwoToTwoScatteringLAB,
)


@dataclass(eq=False)
class Line:
    """
    Class describing a line in a Feynman diagram.

    Args:
        mass: mass of the particle, optional, default 0.
        width: decay width of the particle, optional, default 0.
        name: name for the line, optional
    """

    mass: float = 0.0
    width: float = 0.0
    name: str | None = None
    vertices: list[Vertex] = field(init=False, default_factory=list)
    sqrt_s_min: float | None = field(init=False, default=None)

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.name is not None:
            return self.name
        elif len(self.vertices) == 2:
            return f"{self.vertices[0]} -- {self.vertices[1]}"
        else:
            return "?"


@dataclass(eq=False)
class Vertex:
    lines: list[Line]
    name: str | None = None

    def __post_init__(self):
        for i, line in enumerate(self.lines):
            line.vertices.append(self)
            if len(line.vertices) > 2:
                raise ValueError(f"Line {i} attached to more than two vertices")

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.lines) if self.name is None else self.name


@dataclass
class Diagram:
    incoming: list[Line]
    outgoing: list[Line]
    vertices: list[Vertex]
    t_channel_vertices: list[Vertex] = field(init=False)
    t_channel_lines: list[Line] = field(init=False)
    lines_after_t: list[Line] = field(init=False)
    s_channel_vertices: list[Vertex] = field(init=False)
    s_channel_lines: list[Line] = field(init=False)
    s_decay_layers: list[list[int]] = field(init=False)
    permutation: list[int] = field(init=False)
    inverse_permutation: list[int] = field(init=False)

    def __post_init__(self):
        self._fill_names(self.vertices, "v")
        self._fill_names(self.incoming, "in")
        self._fill_names(self.outgoing, "out")

        (t_channel_lines, self.t_channel_vertices) = self._t_channel_recursive(
            self.incoming[0], None
        )
        self.t_channel_lines = t_channel_lines[1:]
        self._fill_names(self.t_channel_lines, "t")
        self._init_lines_after_t()

        self._init_s_channel()
        self._fill_names(self.s_channel_lines, "s")

    def _fill_names(self, items, prefix):
        for i, item in enumerate(items):
            if item.name is None:
                item.name = f"{prefix}{i+1}"

    def _t_channel_recursive(
        self, line: Line, prev_vertex: Vertex | None
    ) -> tuple[list[Vertex], list[Line]] | None:
        if line is self.incoming[1]:
            return [], []

        if line.vertices[0] is prev_vertex:
            if len(line.vertices) == 1:
                return None
            else:
                vertex = line.vertices[1]
        else:
            vertex = line.vertices[0]

        for out_line in vertex.lines:
            if out_line is line:
                continue
            t_channel = self._t_channel_recursive(out_line, vertex)
            if t_channel is not None:
                return [line, *t_channel[0]], [vertex, *t_channel[1]]
        return None

    def _init_lines_after_t(self):
        t_channel_lines = [self.incoming[0], *self.t_channel_lines, self.incoming[1]]
        self.lines_after_t = []
        for vertex, line_in_1, line_in_2 in zip(
            self.t_channel_vertices, t_channel_lines[:-1], t_channel_lines[1:]
        ):
            for line in vertex.lines:
                if line in [line_in_1, line_in_2]:
                    continue
                self.lines_after_t.append(line)

    def _permutation_recursive(self, lines: list[Line], vertices: list[Vertex]):
        out_indices = []
        for line, parent_vertex in zip(lines, vertices):
            if len(line.vertices) == 1:
                out_indices.append(self.outgoing.index(line))
            else:
                vertex = line.vertices[1 if line.vertices[0] is parent_vertex else 0]
                next_lines = [
                    next_line for next_line in vertex.lines if next_line is not line
                ]
                next_vertices = [vertex] * len(next_lines)
                out_indices.extend(
                    self._permutation_recursive(next_lines, next_vertices)
                )
        return out_indices

    def _init_s_channel(self):
        self.inverse_permutation = self._permutation_recursive(
            self.lines_after_t, self.t_channel_vertices
        )
        self.permutation = [
            self.inverse_permutation.index(i) for i in range(len(self.outgoing))
        ]

        self.s_channel_lines = []
        self.s_channel_vertices = []
        self.s_decay_layers = []
        lines = [self.outgoing[i] for i in self.inverse_permutation]
        vertices = [None] * len(lines)
        has_next_layer = True
        while has_next_layer:
            next_vertices = []
            decay_counts = []
            has_next_layer = False
            for line, parent_vertex in zip(lines, vertices):
                vertex = line.vertices[1 if line.vertices[0] is parent_vertex else 0]
                if vertex not in next_vertices:
                    if vertex in self.t_channel_vertices:
                        next_vertices.append(parent_vertex)
                    else:
                        next_vertices.append(vertex)
                        has_next_layer = True
                    decay_counts.append(0)
                decay_counts[-1] += 1
            self.s_decay_layers.append(decay_counts)

            line_iter = iter(lines)
            vertex_iter = iter(vertices)
            next_lines = []
            for i, (decay_count, vertex) in enumerate(zip(decay_counts, next_vertices)):
                if decay_count == 1:
                    next_vertices[i] = next(vertex_iter)
                    next_lines.append(next(line_iter))
                else:
                    for _ in range(decay_count):
                        next(vertex_iter)
                    decayed_lines = [next(line_iter) for _ in range(decay_count)]
                    next_line = next(
                        line for line in vertex.lines if line not in decayed_lines
                    )
                    next_lines.append(next_line)
                    self.s_channel_lines.append(next_line)
                    self.s_channel_vertices.append(vertex)

            lines = next_lines
            vertices = next_vertices

        del self.s_decay_layers[-1]


class RandomNumbers:
    def __init__(self, random: Tensor):
        self.random = random
        self.index = 0

    def __call__(self, count: int = 1) -> Tensor:
        r = self.random[:, self.index : self.index + count]
        self.index += count
        return r

    def empty(self) -> bool:
        return self.index == self.random.shape[1]


class tDiagramMapping(PhaseSpaceMapping):
    """Implements a mapping for the t-channel part of a Feynman diagram using the algorithm
    described in section 3.2 of
        [1] https://arxiv.org/pdf/2102.00773
    """

    def __init__(self, diagram: Diagram):
        n_particles = len(diagram.lines_after_t)
        if n_particles != len(diagram.t_channel_vertices):
            raise ValueError(
                "Only vertices with 3 lines are supported in the t-channel part of the diagram"
            )
        dims_in = [(3 * n_particles - 4,), (), (n_particles,)]
        dims_out = [(n_particles, 4)]
        super().__init__(dims_in, dims_out)

        none_if_zero = lambda x: None if x == 0 else x

        last_t_line = diagram.t_channel_lines[-1]
        self.t_invariants = [TwoToTwoScatteringCOM(flat=True)]
        self.s_uniform_invariants = []
        for line in reversed(diagram.t_channel_lines[:-1]):
            self.t_invariants.append(TwoToTwoScatteringLAB(flat=True))
            self.s_uniform_invariants.append(UniformInvariantBlock())

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs: list of tensors [r, e_cm, m_out]
                r: random numbers with shape=(b,3*n-4)
                e_cm: COM energy with shape=(b,)
                m_out: (virtual) masses of outgoing particles with shape=(b,n)

        Returns:
            p_in (Tensor): incoming momenta with shape=(b,2,4)
            p_out (Tensor): output momenta with shape=(b,n,4)
            det (Tensor): det of mapping with shape=(b,)
        """
        del condition

        rand = RandomNumbers(inputs[0])  # has dims (b,3*n-4)
        e_cm = inputs[1]  # has dims (b,) or ()
        m_out = inputs[2]  # has dims (b,n)
        det = 1.0

        # construct initial state momenta
        p_in = build_p_in(e_cm)
        p1, p2 = p_in[:, 0], p_in[:, 1]

        # sample s-invariants from the t-channel part of the diagram
        sqrt_s_max = e_cm[:, None] - m_out.flip([1])[:, :-2].cumsum(dim=1).flip([1])
        cumulated_m_out = [m_out[:, :1]]
        for invariant, sqs, sqs_max in zip(
            self.s_uniform_invariants,
            m_out[:, 1:-1].unbind(dim=1),
            sqrt_s_max.unbind(dim=1),
        ):
            s_min = (cumulated_m_out[-1] + sqs[:, None]) ** 2
            s_max = sqs_max[:, None] ** 2
            (s,), jac = invariant.map([rand()], condition=[s_min, s_max])
            if s.isnan().any():
                print("ALARM1!")
            cumulated_m_out.append(s.sqrt())
            det *= jac

        # sample t-invariants and build momenta of t-channel part of the diagram
        p_out = []
        p_t_in = p_in
        p2_rest = p2
        # cnt = len(self.t_invariants)
        for invariant, cum_m_out, mass in zip(
            self.t_invariants,
            reversed(cumulated_m_out),
            reversed(m_out[:, 1:].unbind(dim=1)),
        ):
            # cnt -= 1
            m_t = torch.cat([cum_m_out, mass[:, None]], dim=1)
            r = rand(2)
            (ks,), jac = invariant.map([r, m_t], condition=[p_t_in])
            k_rest, k = ks[:, 0], ks[:, 1]
            p_out.append(k)
            p2_rest = p2_rest - k
            p_t_in = torch.stack([p1, p2_rest], dim=1)
            det *= jac
        p_out.append(k_rest)
        p_out = torch.stack(p_out, dim=1).flip([1])
        return (p_in, p_out), det

    def map_inverse(self, inputs: TensorList, condition=None):
        """Map from momenta to random numbers

        Args:
            inputs: list of tensors [p_in, p_out]
                p_in: incoming momenta with shape=(b,2,4)
                p_out: outgoing momenta with shape=(b,n,4)

        Returns:
            r: random numbers with shape=(b,3*n-4)
            e_cm: COM energy with shape=(b,)
            m_out: (virtual) masses of outgoing particles with shape=(b,n)
            det (Tensor): det of mapping with shape=(b,)

        """
        del condition

        p_in = inputs[0]
        p_out = inputs[1]
        det = 1.0

        random = []

        e_cm = mass(p_in.sum(dim=1))
        m_out = mass(p_out)
        cum_p = p_out.cumsum(dim=1)
        cum_p_reverse = torch.cat(
            [torch.zeros_like(p_out[:, :1]), p_out[:, 2:].flip([1])], dim=1
        ).cumsum(dim=1)
        ss = lsquare(cum_p)
        s_maxs = (
            e_cm[:, None] - m_out.flip([1])[:, :-2].cumsum(dim=1).flip([1])
        ).square()
        s_mins = (ss[:, :-2].sqrt() + m_out[:, 1:-1]).square()
        p_t_ins = torch.stack(
            [
                p_in[:, :1].expand(-1, cum_p_reverse.shape[1], -1),
                p_in[:, 1:] - cum_p_reverse,
            ],
            dim=2,
        )
        k_rest_k = torch.stack([cum_p[:, :-1], p_out[:, 1:]], dim=2)

        for invariant, s, s_min, s_max in zip(
            self.s_uniform_invariants,
            ss[:, 1:-1].unbind(dim=1),
            s_mins.unbind(dim=1),
            s_maxs.unbind(dim=1),
        ):
            (r,), jac = invariant.map_inverse(
                [s[:, None]], condition=[s_min[:, None], s_max[:, None]]
            )
            random.append(r)
            det *= jac

        for invariant, p_t_in, ks in zip(
            self.t_invariants, p_t_ins.unbind(dim=1), reversed(k_rest_k.unbind(dim=1))
        ):
            (r, _), jac = invariant.map_inverse([ks], condition=[p_t_in])
            random.append(r)
            det *= jac

        r = torch.cat(random, dim=1)
        return (r, e_cm, m_out), det


class DiagramMapping(PhaseSpaceMapping):
    """
    TODO:
        - support quartic vertices
    """

    def __init__(
        self,
        diagram: Diagram,
        s_lab: Tensor,
        s_hat_min: float = 0.0,
        leptonic: bool = False,
        t_mapping: str = "diagram",
        s_min_epsilon: float = 1e-2,
    ):
        n_out = len(diagram.outgoing)
        dims_in = [(3 * n_out - 2 - (2 if leptonic else 0),)]
        dims_out = [(n_out + 2, 4), (2,)]
        super().__init__(dims_in, dims_out)

        self.diagram = diagram
        self.s_lab = s_lab
        self.sqrt_s_epsilon = s_min_epsilon**0.5
        self.has_t_channel = len(diagram.t_channel_lines) != 0

        # Initialize s invariants and decay mappings
        sqrt_s_min = [
            diagram.outgoing[diagram.inverse_permutation[i]].mass
            for i in range(len(diagram.outgoing))
        ]
        self.s_decay_invariants = []
        self.s_decays = []
        line_iter = iter(diagram.s_channel_lines)
        s_decay_layers = (
            diagram.s_decay_layers
            if self.has_t_channel
            else diagram.s_decay_layers[:-1]
        )
        for layer in s_decay_layers:
            sqs_iter = iter(sqrt_s_min)
            sqrt_s_min = []
            layer_invariants = []
            layer_decays = []
            for count in layer:
                sqs_min = max(
                    self.sqrt_s_epsilon, sum(next(sqs_iter) for i in range(count))
                )
                sqrt_s_min.append(sqs_min)
                if count == 1:
                    continue
                line = next(line_iter)
                layer_invariants.append(
                    MasslessInvariantBlock(nu=1.4)
                    if line.mass == 0.0 or line.mass < sqs_min
                    else (
                        StableInvariantBlock(mass=line.mass, nu=1.4)
                        if line.width == 0.0
                        else BreitWignerInvariantBlock(mass=line.mass, width=line.width)
                    )
                )
                layer_decays.append(TwoBodyDecayLAB())
            self.s_decay_invariants.append(layer_invariants)
            self.s_decays.append(layer_decays)
        if not self.has_t_channel:
            self.s_decay_invariants.append([])
            self.s_decays.append([TwoBodyDecayCOM()])

        # Initialize luminosity and t-channel mapping
        s_hat_min = torch.tensor(max(sum(sqrt_s_min) ** 2, s_hat_min))
        self.luminosity = None
        if self.has_t_channel:
            self.t_channel_type = t_mapping
            n_lines_after_t = len(diagram.lines_after_t)
            self.t_random_numbers = 3 * n_lines_after_t - 4
            if not (leptonic or t_mapping == "chili"):
                self.luminosity = Luminosity(s_lab, s_hat_min)
            if t_mapping == "diagram":
                self.t_mapping = tDiagramMapping(diagram)
            elif t_mapping == "rambo":
                self.t_mapping = tRamboBlock(n_lines_after_t)
            elif t_mapping == "chili":
                if leptonic:
                    raise ValueError("chili only supports hadronic processes")
                # TODO: allow to set ymax, ptmin
                self.t_mapping = tChiliBlock(
                    n_lines_after_t,
                    ymax=torch.full((n_lines_after_t,), 4.0),
                    ptmin=torch.full((n_lines_after_t,), 20.0),
                )
                self.t_random_numbers += 2
            else:
                raise ValueError(f"Unknown t-channel mapping {t_mapping}")
        elif not leptonic:
            s_line = diagram.s_channel_lines[-1]
            if s_line.mass != 0.0 and s_line.mass > s_hat_min.sqrt():
                self.luminosity = ResonantLuminosity(
                    s_lab, s_line.mass, s_line.width, s_hat_min
                )
            else:
                self.luminosity = Luminosity(s_lab, s_hat_min)

        self.permutation = self.diagram.permutation
        self.inverse_permutation = self.diagram.inverse_permutation
        self.pi_factors = (2 * pi) ** (4 - 3 * n_out)

    def map(self, inputs: TensorList, condition=None):
        random = inputs[0]
        rand = RandomNumbers(random)
        ps_weight = 1.0

        # Do luminosity and get s_hat and rapidity
        if self.luminosity is None:
            s_hat = torch.full((random.shape[0],), self.s_lab, device=random.device)
            x1x2 = torch.ones((random.shape[0], 2), device=random.device)
        else:
            (x1x2,), jac_lumi = self.luminosity.map([rand(2)])
            ps_weight *= jac_lumi
            s_hat = self.s_lab * x1x2.prod(dim=1)
            rap = 0.5 * torch.log(x1x2[:, 0] / x1x2[:, 1])[:, None]
        sqrt_s_hat = s_hat.sqrt()

        # sample s-invariants from decays, starting from the final state particles
        sqrt_s = [
            torch.full_like(
                sqrt_s_hat, self.diagram.outgoing[self.inverse_permutation[i]].mass
            )[:, None]
            for i in range(len(self.diagram.outgoing))
        ]
        decay_masses = []
        for layer_counts, layer_invariants in zip(
            self.diagram.s_decay_layers, self.s_decay_invariants
        ):
            sqrt_s_min = []
            sqrt_s_index = 0
            layer_masses = []
            for decay_count in layer_counts:
                sqs_clip = self.sqrt_s_epsilon if decay_count > 1 else 0.0
                sqrt_s_min.append(
                    torch.clip(
                        sum(sqrt_s[sqrt_s_index + i] for i in range(decay_count)),
                        min=sqs_clip,
                    )
                )
                layer_masses.append(sqrt_s[sqrt_s_index : sqrt_s_index + decay_count])
                sqrt_s_index += decay_count
            decay_masses.append(layer_masses)

            if len(layer_invariants) == 0:
                assert not self.has_t_channel
                continue

            sqrt_s = []
            invariant_iter = iter(layer_invariants)
            for i, decay_count in enumerate(layer_counts):
                if decay_count == 1:
                    sqrt_s.append(sqrt_s_min[i])
                    continue
                s_min = sqrt_s_min[i] ** 2
                s_max = (
                    sqrt_s_hat[:, None] - sum(sqrt_s) - sum(sqrt_s_min[i + 1 :])
                ) ** 2
                (s,), jac = next(invariant_iter).map([rand()], condition=[s_min, s_max])
                sqrt_s.append(s.sqrt())
                ps_weight *= jac

        if self.has_t_channel:
            (p_in, p_out), jac = self.t_mapping.map(
                [rand(self.t_random_numbers), sqrt_s_hat, torch.cat(sqrt_s, dim=1)]
            )
            if self.t_channel_type == "chili":
                x1 = p_in[:, 0, 0] * 2 / sqrt_s_hat
                x2 = p_in[:, 1, 0] * 2 / sqrt_s_hat
                x1x2 = torch.stack([x1, x2], dim=1)
            ps_weight *= jac
            p_out = p_out.unbind(dim=1)
        else:
            p_in = build_p_in(sqrt_s_hat)
            p_out = [s_hat]

        # build the momenta of the decays
        for layer_counts, layer_decays, layer_masses in zip(
            reversed(self.diagram.s_decay_layers),
            reversed(self.s_decays),
            reversed(decay_masses),
        ):
            p_out_prev = p_out
            p_out = []
            decay_iter = iter(layer_decays)
            for count, k_in, masses in zip(layer_counts, p_out_prev, layer_masses):
                if count == 1:
                    p_out.append(k_in)
                    continue
                m_out = torch.cat(masses, dim=1)
                (k_out,), jac = next(decay_iter).map([rand(2), k_in, m_out])
                p_out.extend(k_out.unbind(dim=1))
                ps_weight *= jac
        p_out = torch.stack(p_out, dim=1)

        # we should have consumed all the random numbers
        assert rand.empty()

        # permute and return momenta
        p_ext = torch.cat([p_in, p_out[:, self.permutation]], dim=1)
        p_ext_lab = p_ext if self.luminosity is None else boost_beam(p_ext, rap)
        return (p_ext_lab, x1x2), ps_weight * self.pi_factors

    def map_inverse(self, inputs: TensorList, condition=None):
        del condition
        p_ext_lab, x1x2 = inputs
        random = []
        ps_weight = 1.0

        # Undo boosts etc
        s_hat = self.s_lab * x1x2.prod(dim=1)
        sqrt_s_hat = s_hat.sqrt()
        rap = 0.5 * torch.log(x1x2[:, 0] / x1x2[:, 1])[:, None]
        if self.luminosity is None:
            p_ext = p_ext_lab
        else:
            p_ext = boost_beam(p_ext_lab, rap, inverse=True)
        p_in = p_ext[:, :2]
        p_out = p_ext[:, 2:][:, self.inverse_permutation]

        p_out = p_out.unbind(dim=1)[::-1]
        s_invariant_r = []
        for layer_counts, layer_decays, layer_invariants in zip(
            self.diagram.s_decay_layers,
            self.s_decays,
            self.s_decay_invariants,
        ):
            sqrt_s = lsquare(torch.stack(p_out, dim=1)).sqrt()
            sqrt_s_mins = []
            p_out_iter = iter(p_out)
            p_out = []
            decay_iter = reversed(layer_decays)
            for count in reversed(layer_counts):
                if count == 1:
                    k_out = next(p_out_iter)
                    sqrt_s_mins.append(mass(k_out)[:, None])
                    p_out.append(k_out)
                    continue

                k_out = torch.stack(
                    [next(p_out_iter) for _ in range(count)][::-1], dim=1
                )
                (r, k_in, m_out), jac = next(decay_iter).map_inverse([k_out])
                ps_weight *= jac
                p_out.append(k_in)
                random.append(r)
                sqrt_s_mins.append(
                    torch.clip(m_out.sum(dim=1), min=self.sqrt_s_epsilon)[:, None]
                )

            if len(layer_invariants) == 0:
                assert not self.has_t_channel
                continue

            invariant_iter = reversed(layer_invariants)
            layer_s_invariant_r = []
            sqrt_s_sum = 0.0
            for i, (count, sqrt_s_min, k_in) in enumerate(
                zip(layer_counts, reversed(sqrt_s_mins), reversed(p_out))
            ):
                s = lsquare(k_in)[:, None]
                sqs = torch.clip(s, min=0).sqrt()
                if count == 1:
                    sqrt_s_sum += sqs
                    continue
                s_min = sqrt_s_min.square()
                s_max = (
                    sqrt_s_hat[:, None] - sqrt_s_sum - sum(sqrt_s_mins[: -i - 1])
                ).square()
                sqrt_s_sum += sqs
                (r,), jac = next(invariant_iter).map_inverse(
                    [s], condition=[s_min, s_max]
                )
                layer_s_invariant_r.append(r)
                ps_weight *= jac
            s_invariant_r.append(layer_s_invariant_r)

        if self.has_t_channel:
            (r, sqrt_s_hat, sqrt_s), jac = self.t_mapping.map_inverse(
                [p_in, torch.stack(p_out, dim=1).flip([1])]
            )
            sqrt_s = sqrt_s[:, :, None].unbind(dim=1)
            random.append(r)
            ps_weight *= jac

        for layer_s_invariant_r in reversed(s_invariant_r):
            random.extend(reversed(layer_s_invariant_r))

        # Undo lumi param
        if self.luminosity is not None:
            (r,), jac = self.luminosity.map_inverse([x1x2])
            ps_weight *= jac
            random.append(r)

        r = torch.cat(random[::-1], dim=1)
        return (r,), ps_weight / self.pi_factors
