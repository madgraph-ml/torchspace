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
leptonic = False

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

vbs = Diagram(
    incoming=[in1, in2], outgoing=[out1, out2, out3, out4], vertices=[v1, v2, v3, v4]
)
dmap = DiagramMapping(vbs, torch.tensor(13000.0**2))

(p, x), jac_fw = dmap.map([r])
print("+++++++++++++++++++")
(r_inv,), jac_inv = dmap.map_inverse([p, x])


print("Δr  ", (r - r_inv).abs().max())
print("Δjac", (jac_fw * jac_inv - 1.0).abs().max())

"""
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

ww_llvv = Diagram(
    incoming=[in1, in2],
    outgoing=[out1, out2, out3, out4],
    vertices=[v1, v2, v3, v4]
)
dmap = DiagramMapping(
    ww_llvv, torch.tensor(13000.**2), 20.**2, leptonic=leptonic, s_min_epsilon=0.
)
(p, x), jac_fw = dmap.map([r])
print("+++++++++++++++++++")
(r_inv,), jac_inv = dmap.map_inverse([p, x])


print("Δr  ", (r - r_inv).abs().max())
print("Δjac", (jac_fw * jac_inv - 1.).abs().max())
"""
