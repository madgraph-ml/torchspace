import torch

torch.set_default_dtype(torch.float64)
from torchspace.single_channel import Diagramm_llvvA, Diagramm_ww_llvv

MW = torch.tensor(80.377)
WW = torch.tensor(2.085)
MZ = torch.tensor(91.1876)
WZ = torch.tensor(2.4952)


print("====== All s-channel ======\n")

for leptonic in [False, True]:
    if leptonic:
        print("------- leptonic ----------\n")
    else:
        print("------- hadronic ----------\n")
    for mV, wV in [[MZ, WZ], [None, None]]:
        if mV is not None:
            print("~~~~~ Z Channel ~~~~~\n")
        else:
            print("~~~~~ Photon Channel ~~~~~\n")
        s = torch.tensor(500.0**2) if leptonic else torch.tensor(13000.0**2)
        k = 0 if leptonic else 2
        n = int(1000000)
        shape = (n, 8 + k)
        llmap = Diagramm_ww_llvv(s, MW, WW, mV=mV, wV=wV, leptonic=leptonic)
        r = torch.rand(shape)
        (p_ext, x1x2), det = llmap.map([r])
        print(p_ext[..., 0].abs().max())
        print(p_ext[..., 0].min())
        print("\n")


print("====== Multi-structured ======\n")

for leptonic in [True, False]:
    if leptonic:
        print("------- leptonic ----------\n")
    else:
        print("------- hadronic ----------\n")
    for mV, wV in [[MZ, WZ], [None, None]]:
        if mV is not None:
            print("~~~~~ Z Channel ~~~~~\n")
        else:
            print("~~~~~ Photon Channel ~~~~~\n")
        s = torch.tensor(500.0**2) if leptonic else torch.tensor(13000.0**2)
        k = 0 if leptonic else 2
        n = int(1000000)
        shape = (n, 11 + k)
        llmap = Diagramm_llvvA(s, MW, WW, mV=mV, wV=wV, leptonic=leptonic)
        r = torch.rand(shape)
        (p_ext, x1x2), det = llmap.map([r])
        print(p_ext[..., 0].abs().max())
        print(p_ext[..., 0].min())
        print("\n")
