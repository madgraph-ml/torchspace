from madspace.threeparticle import ThreeBodyDecayCOM
import torch
from pytest import approx

torch.set_default_dtype(torch.float64)

##=============== Definitions ==================##

E_BEAM = 500. # should be the beam energy from your madgraph run_card 
S_TOT = E_BEAM * E_BEAM
M1 = 20.
M2 = 30.
M3 = 40.


##=============== GENERATE THE MOMENTA ==================##

# Initialize phase-space generator
decay3to1 = ThreeBodyDecayCOM()

# sample momenta
n = int(1e4)
r = torch.rand((n, 5))
s = torch.full((n,), S_TOT)
m_out = torch.tensor([M1, M2, M3]).repeat(n, 1)
(p_decay, ), weight = decay3to1.map([r, s, m_out])

##=============== Checks ==================##

def mass(momentum):
    return torch.sqrt(momentum[:, 0]**2 - torch.sum(momentum[:, 1:]**2, axis=1))

print("Sampled momenta:")
print(p_decay.shape)
print(p_decay[0:2])
print(p_decay[0:2].sum(dim=1))
m1 = mass(p_decay[:, 0, :])
m2 = mass(p_decay[:, 1, :])
m3 = mass(p_decay[:, 2, :])
assert torch.allclose(m1, torch.full((n,), M1))
assert torch.allclose(m2, torch.full((n,), M2))
assert torch.allclose(m3, torch.full((n,), M3))
  # check momentum conservation