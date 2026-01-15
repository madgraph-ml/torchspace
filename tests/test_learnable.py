import torch
import torch.nn as nn

from torchspace.functional.ps_utils import build_p_in
from torchspace.invariants import BreitWignerInvariantBlock, UniformInvariantBlock
from torchspace.spline_mapping import SplineMapping
from torchspace.twoparticle import TwoBodyDecayCOM, TwoToTwoScatteringCOM

torch.set_default_dtype(torch.float64)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("------------------------------------------------------------")
print("TEST 1: Learn Breit-Wigner with fixed min/max")
print("------------------------------------------------------------")

uni_inv = UniformInvariantBlock()
bw_inv = BreitWignerInvariantBlock(mass=90.0, width=4.0)
smap = SplineMapping(mapping=uni_inv, flow_dims_c=[])
print("Trainable parameters:", count_parameters(smap))

n = 1024

opt = torch.optim.Adam(smap.parameters(), lr=1e-2)

for i in range(1000):
    opt.zero_grad()
    r = torch.rand((n, 1))
    s_min = torch.full_like(r, 1.0**2)
    s_max = torch.full_like(r, 1000.0**2)
    (s,), jac_fw = smap.map([r], [s_min, s_max])
    (r_out,), jac_inv = bw_inv.map_inverse([s], [s_min, s_max])
    loss = -torch.log(jac_fw * jac_inv).mean()
    loss.backward()
    opt.step()
    if (i + 1) % 50 == 0:
        print(i + 1, loss.item())

print()
print("median |Δr|", (r - r_out).abs().median().item())
print("median |Δlog jac|", (jac_fw.log() + jac_inv.log()).abs().median().item())

print()
print("------------------------------------------------------------")
print("TEST 2: Learn Breit-Wigner conditioned on random min/max")
print("------------------------------------------------------------")

uni_inv = UniformInvariantBlock()
bw_inv = BreitWignerInvariantBlock(mass=90.0, width=4.0)
smap = SplineMapping(mapping=uni_inv, flow_dims_c=[(2, 1)])
print("Trainable parameters:", count_parameters(smap))

n = 1024

opt = torch.optim.Adam(smap.parameters(), lr=1e-2)

for i in range(1000):
    opt.zero_grad()
    r = torch.rand((n, 1))
    r_min = torch.rand((n, 1))
    r_max = torch.rand((n, 1))
    s_min = (r_min * 80 + 1) ** 2
    s_max = (r_max * 1000 + 100) ** 2
    (s,), jac_fw = smap.map([r], [s_min, s_max, r_min, r_max])
    (r_out,), jac_inv = bw_inv.map_inverse([s], [s_min, s_max])
    loss = -torch.log(jac_fw * jac_inv).mean()
    loss.backward()
    opt.step()
    if (i + 1) % 50 == 0:
        print(i + 1, loss.item())

print()
print("median |Δr|", (r - r_out).abs().median().item())
print("median |Δlog jac|", (jac_fw.log() + jac_inv.log()).abs().median().item())

print()
print("------------------------------------------------------------")
print("TEST 3: Learn two Breit-Wigners with shared network")
print("------------------------------------------------------------")

uni1_inv = UniformInvariantBlock()
bw1_inv = BreitWignerInvariantBlock(mass=85.0, width=4.0)
smap1 = SplineMapping(mapping=uni1_inv, flow_dims_c=[(2, 1)], extra_params_c=1)

uni2_inv = UniformInvariantBlock()
bw2_inv = BreitWignerInvariantBlock(mass=95.0, width=4.0)
smap2 = smap1.shared_mapping(uni2_inv)

combined = nn.ModuleList([smap1, smap2])
print("Trainable parameters:", count_parameters(combined))

n = 1024

opt = torch.optim.Adam(combined.parameters(), lr=1e-2)

for i in range(1000):
    opt.zero_grad()
    r = torch.rand((2 * n, 1))
    r_min = torch.rand((2 * n, 1))
    r_max = torch.rand((2 * n, 1))
    s_min = (r_min * 80 + 1) ** 2
    s_max = (r_max * 1000 + 100) ** 2

    (s,), jac_fw1 = smap1.map([r[:n]], [s_min[:n], s_max[:n], r_min[:n], r_max[:n]])
    (r_out1,), jac_inv1 = bw1_inv.map_inverse([s], [s_min[:n], s_max[:n]])

    (s,), jac_fw2 = smap2.map([r[n:]], [s_min[n:], s_max[n:], r_min[n:], r_max[n:]])
    (r_out2,), jac_inv2 = bw2_inv.map_inverse([s], [s_min[n:], s_max[n:]])

    r_out = torch.cat((r_out1, r_out2), dim=0)
    jac_fw = torch.cat((jac_fw1, jac_fw2), dim=0)
    jac_inv = torch.cat((jac_inv1, jac_inv2), dim=0)
    loss = -torch.log(jac_fw * jac_inv).mean()
    loss.backward()
    opt.step()
    if (i + 1) % 50 == 0:
        print(i + 1, loss.item())

print()
print("median |Δr|", (r - r_out).abs().median().item())
print("median |Δlog jac|", (jac_fw.log() + jac_inv.log()).abs().median().item())
print("extra conditions", smap1.extra_params_c.item(), smap2.extra_params_c.item())

print()
print("------------------------------------------------------------")
print("TEST 4: Learn 2D distribution")
print("------------------------------------------------------------")

iso = TwoBodyDecayCOM()
aniso = TwoToTwoScatteringCOM()
smap = SplineMapping(
    mapping=iso,
    flow_dims_c=[],
    correlations="all",
    periodic=[0],
    permutation=[0, 1],
)
print("Trainable parameters:", count_parameters(smap))

n = 1024
opt = torch.optim.Adam(smap.parameters(), lr=1e-2)

sqrt_s = torch.full((n,), 90.0)
s = sqrt_s.square()
m_out = torch.zeros((n, 2))
p_in = build_p_in(sqrt_s)

for i in range(1000):
    opt.zero_grad()
    r = torch.rand((n, 2))
    (p_decay,), jac_fw = smap.map([r, s, m_out])
    (r_out, _), jac_inv = aniso.map_inverse([p_decay], [p_in])
    loss = -torch.log(jac_fw * jac_inv).mean()
    loss.backward()
    opt.step()
    if (i + 1) % 50 == 0:
        print(i + 1, loss.item())


print()
print("median |Δlog jac|", (jac_fw.log() + jac_inv.log()).abs().median().item())

print()
print("Check if the inverse mapping works:")
r = torch.rand((n, 2))
(p_decay,), jac_fw = smap.map([r, s, m_out])
(r_inv, _, _), jac_inv = smap.map_inverse([p_decay])
print("max |Δr|", (r - r_inv).abs().max().item())
print("max |Δjac|", (jac_fw * jac_inv - 1).abs().max().item())
