from torchspace.diagram_mapping import *
from torchspace.single_channel import (
    Diagramm_llvvA,
    Diagramm_ww_llvv,
    SingleChannelVBS,
    SingleChannelWWW,
)

torch.set_default_dtype(torch.float64)

MW = 80.377
WW = 2.085
MZ = 91.1876
WZ = 2.4952

nsamples = 300


def print_info(diagram):
    print("t vertices", diagram.t_channel_vertices)
    print("t lines   ", diagram.t_channel_lines)
    print("s vertices", diagram.s_channel_vertices)
    print("s lines   ", diagram.s_channel_lines)
    print("decays    ", diagram.s_decay_layers)
    print("perm      ", diagram.permutation)


print("====== VBS ======")
in1 = Line()
in2 = Line()

out1 = Line(mass=MW)
out2 = Line()
out3 = Line()
out4 = Line(mass=MW)

t1 = Line()
t2 = Line()
t3 = Line()

v1 = Vertex([in1, out1, t3])
v2 = Vertex([t3, out2, t2])
v3 = Vertex([t2, out3, t1])
v4 = Vertex([t1, out4, in2])

r = torch.rand(nsamples, 10)
vbsmap = SingleChannelVBS(torch.tensor(13000.0**2), torch.tensor([MW]))
(p_hand, x_hand), jac_hand = vbsmap.map([r])

vbs = Diagram(
    incoming=[in1, in2], outgoing=[out1, out2, out3, out4], vertices=[v1, v2, v3, v4]
)
dmap = DiagramMapping(vbs, torch.tensor(13000.0**2))
(p_auto, x_auto), jac_auto = dmap.map([r])
(r_inv,), jac_inv = dmap.map_inverse([p_auto, x_auto])

print_info(vbs)
print("Δp max    ", (p_auto - p_hand).abs().max())
print("Δx max    ", (x_auto - x_hand).abs().max())
print("Δjac max  ", (jac_auto - jac_hand).abs().max())
print("inv Δr    ", (r - r_inv).abs().max())
print("inv Δjac  ", (jac_hand * jac_inv - 1.0).abs().max())

print()
print("====== WWW ======")

in1 = Line()
in2 = Line()

out1 = Line(mass=MW)
out2 = Line(mass=MW)
out3 = Line(mass=MW)

t1 = Line()
s1 = Line(mass=MZ, width=WZ)

v1 = Vertex([in1, s1, t1])
v2 = Vertex([t1, in2, out3])
v3 = Vertex([s1, out1, out2])

r = torch.rand(nsamples, 7)
vbsmap = SingleChannelWWW(
    torch.tensor(13000.0**2), torch.tensor(MW), torch.tensor(MZ), torch.tensor(WZ)
)
(p_hand, x_hand), jac_hand = vbsmap.map([r])
(r_hand_inv,), jac_hand_inv = vbsmap.map_inverse([p_hand, x_hand])
print("inv Δr    ", (r - r_hand_inv).abs().max())
print("inv Δjac  ", (jac_hand * jac_hand_inv - 1.0).abs().max())

www = Diagram(incoming=[in1, in2], outgoing=[out1, out2, out3], vertices=[v1, v2, v3])
dmap = DiagramMapping(www, torch.tensor(13000.0**2))
(p_auto, x_auto), jac_auto = dmap.map([r])
(r_inv,), jac_inv = dmap.map_inverse([p_auto, x_auto])

print_info(www)
print("Δp max    ", (p_auto - p_hand).abs().max())
print("Δx max    ", (x_auto - x_hand).abs().max())
print("Δjac max  ", (jac_auto - jac_hand).abs().max())
print("inv Δr    ", (r - r_inv).abs().max())
print("inv Δjac  ", (jac_hand * jac_inv - 1.0).abs().max())

print()
print("====== llvvA ======")

for leptonic in [False, True]:
    print()
    if leptonic:
        print("------ leptonic ------")
    else:
        print("------ hadronic ------")
    in1 = Line()
    in2 = Line()

    out1 = Line()
    out2 = Line()
    out3 = Line()
    out4 = Line()
    out5 = Line()

    t1 = Line(mass=MW, width=WW)
    t2 = Line(mass=MW, width=WW)
    s1 = Line(mass=MZ, width=WZ)
    s2 = Line()

    v1 = Vertex([in1, out1, t2])
    v2 = Vertex([t2, t1, s1])
    v3 = Vertex([in2, out5, t1])
    v4 = Vertex([s1, s2, out4])
    v5 = Vertex([s2, out2, out3])

    r = torch.rand(nsamples, 11 if leptonic else 13)
    llvva_map = Diagramm_llvvA(
        torch.tensor(13000.0**2),
        torch.tensor(MW),
        torch.tensor(WW),
        mV=torch.tensor(MZ),
        wV=torch.tensor(WZ),
        leptonic=leptonic,
    )
    if leptonic:
        r_perm = r
    else:
        perm = [*range(2, 13), 0, 1]
        r_perm = r[:, perm]
    (p_hand, x_hand), jac_hand = llvva_map.map([r_perm])

    llvva = Diagram(
        incoming=[in1, in2],
        outgoing=[out1, out2, out3, out4, out5],
        vertices=[v1, v2, v3, v4, v5],
    )
    dmap = DiagramMapping(llvva, torch.tensor(13000.0**2), 20.0**2, leptonic=leptonic)
    (p_auto, x_auto), jac_auto = dmap.map([r])
    (r_inv,), jac_inv = dmap.map_inverse([p_auto, x_auto])

    print_info(llvva)
    print("Δp max    ", (p_auto - p_hand).abs().max())
    print("Δx max    ", (x_auto - x_hand).abs().max())
    print("Δjac max  ", (jac_auto - jac_hand).abs().max())
    print("inv Δr    ", (r - r_inv).abs().max())
    print("inv Δjac  ", (jac_hand * jac_inv - 1.0).abs().max())

print()
print("====== ww_llvv ======")

for leptonic in [False, True]:
    print()
    if leptonic:
        print("------ leptonic ------")
    else:
        print("------ hadronic ------")
    in1 = Line()
    in2 = Line()

    out1 = Line()
    out2 = Line()
    out3 = Line()
    out4 = Line()

    s1234 = Line(mass=MZ, width=WZ)
    s12 = Line(mass=MW, width=WW)
    s34 = Line(mass=MW, width=WW)

    v1 = Vertex([in1, in2, s1234])
    v2 = Vertex([s1234, s12, s34])
    v3 = Vertex([s12, out1, out2])
    v4 = Vertex([s34, out3, out4])

    r = torch.rand(nsamples, 8 if leptonic else 10)
    ww_llvv_map = Diagramm_ww_llvv(
        torch.tensor(13000.0**2),
        torch.tensor(MW),
        torch.tensor(WW),
        mV=torch.tensor(MZ),
        wV=torch.tensor(WZ),
        leptonic=leptonic,
    )
    if leptonic:
        r_perm = r
    else:
        perm = [*range(2, 10), 0, 1]
        r_perm = r[:, perm]
    (p_hand, x_hand), jac_hand = ww_llvv_map.map([r_perm])

    ww_llvv = Diagram(
        incoming=[in1, in2],
        outgoing=[out1, out2, out3, out4],
        vertices=[v1, v2, v3, v4],
    )
    dmap = DiagramMapping(
        ww_llvv, torch.tensor(13000.0**2), 20.0**2, leptonic=leptonic, s_min_epsilon=0.0
    )
    (p_auto, x_auto), jac_auto = dmap.map([r])
    (r_inv,), jac_inv = dmap.map_inverse([p_auto, x_auto])

    print_info(ww_llvv)
    print("Δp max    ", (p_auto - p_hand).abs().max())
    print("Δx max    ", (x_auto - x_hand).abs().max())
    print("Δjac max  ", (jac_auto - jac_hand).abs().max())
    print("inv Δr    ", (r - r_inv).abs().max())
    print("inv Δjac  ", (jac_hand * jac_inv - 1.0).abs().max())
