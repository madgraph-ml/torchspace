from torchspace.rambo import Mahambo
from torchspace.cuts import PhaseSpaceCuts
import torch
import yaml

##=============== Definitions ==================##

# Some definitions
# Particle PIDS
TOP = 6
HIGGS = 25
JET = 100

# Final state pids requires
# With this, the cutter knows on which particles to cut
PIDS = [TOP, -TOP, HIGGS, JET]
NPARTICLES = 4 # number of final state particles
MASSES = [173.0, 173.0, 125.0, 0.0] # Make sure these masses agree with what you generated in madgraph
E_BEAM = 6500 # should be the beam energy from your madgraph run_card 


##=============== GENERATE THE MOMENTA ==================##

# Initialize phase-space generator
rambo = Mahambo(E_BEAM, NPARTICLES, masses=MASSES)

# sample momenta
n = int(1e4)
r = torch.rand((n, 3*NPARTICLES-2))
(p_lab, x1x2), weight = rambo.map([r])

##=============== DO THE CUTS ==================##

# Get Cuts from yaml
CUT_PATH = "cuts.yaml"
with open(CUT_PATH, "r") as f1:
    CUTS = yaml.load(f1, Loader=yaml.FullLoader)

# Define the cutter
cutter = PhaseSpaceCuts(pids=PIDS, nparticles=NPARTICLES, cuts=CUTS)

# get cuts
cut_mask = cutter.cut(p_lab[:, 2:]) # only cut final_state

# get cutted momenta
p_lab_cut = p_lab[cut_mask]

# just for checking, see if there are less events
print(p_lab.shape)
print(p_lab_cut.shape)