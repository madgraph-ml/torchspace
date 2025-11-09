import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
from .functional import kinematics as obs


# Assign PIDS to particle classes
JET_PIDS = [1, 2, 3, 4, -1, -2, -3, -4, 21]
LEPTON_PIDS = [11, 13, 15, -11, -13, -15]
MISSING_PIDS = [12, 14, 16, -12, -14, -16]


class PhaseSpaceCuts(nn.Module):
    """Base class for all phase-space cuts."""

    def __init__(
        self,
        pids: list,
        nparticles: int,
        cuts: dict,
        **kwargs,
    ):
        """_summary_

        Args:
            pids (list): expects pids of outgoing particles
            nparticles (int): number of outgoing particles
            cuts (dict): Dictionary with cuts.
        """
        super().__init__()

        if len(pids) != nparticles:
            raise ValueError(f"Expected {nparticles} pids, got {len(pids)}")
        self.nparticles = nparticles

        # Do some pid arithmetic to get ids
        pids = [self.pid_to_inttype(pid) for pid in pids]
        pid_tensor = torch.tensor(pids)

        # get ids of common particles
        jet_ids = torch.argwhere(pid_tensor == 100)[:, 0]
        b_ids = torch.argwhere(pid_tensor == 101)[:, 0]
        lepton_ids = torch.argwhere(pid_tensor == 102)[:, 0]
        photon_ids = torch.argwhere(pid_tensor == 103)[:, 0]
        invisible_ids = torch.argwhere(pid_tensor == 104)[:, 0]

        # Get ids for 2-pair combinatorics
        jj_tri_id = torch.triu_indices(len(jet_ids), len(jet_ids), 1)
        jj_ids = (jet_ids[jj_tri_id[0]], jet_ids[jj_tri_id[1]])

        bb_tri_id = torch.triu_indices(len(b_ids), len(b_ids), 1)
        bb_ids = (b_ids[bb_tri_id[0]], b_ids[bb_tri_id[1]])

        ll_tri_id = torch.triu_indices(len(lepton_ids), len(lepton_ids), 1)
        ll_ids = (lepton_ids[ll_tri_id[0]], lepton_ids[ll_tri_id[1]])

        aa_tri_id = torch.triu_indices(len(photon_ids), len(photon_ids), 1)
        aa_ids = (photon_ids[aa_tri_id[0]], photon_ids[aa_tri_id[1]])

        jl_ids = (
            jet_ids.repeat_interleave(len(lepton_ids)),
            torch.tile(lepton_ids, dims=(len(jet_ids),)),
        )

        # Get the cuts
        key_list = list(cuts.keys())
        self.pt_cuts = cuts["pt"] if "pt" in key_list else None
        self.eta_cuts = cuts["eta"] if "eta" in key_list else None
        self.dR_cuts = cuts["dR"] if "dR" in key_list else None
        self.mm_cuts = cuts["mm"] if "mm" in key_list else None
        self.sqrts_cut = cuts["sqrt_shat"] if "sqrt_shat" in key_list else None

        self.key_to_ids = {
            "jets": jet_ids,
            "bquarks": b_ids,
            "photon": photon_ids,
            "lepton": lepton_ids,
            "invisible": invisible_ids,
            "jj": jj_ids,
            "bb": bb_ids,
            "aa": aa_ids,
            "ll": ll_ids,
            "jl": jl_ids,
        }

    def cut(self, p: torch.Tensor) -> torch.Tensor:
        B, N, k = p.shape
        assert k == 4, "Expected last dimension to be 4 (4-momenta)"
        assert N == self.nparticles, f"Expected {self.nparticles} particles, got {N}"

        # Start with all momenta
        mask = torch.ones(p.shape[0], device=p.device).bool()

        # Cut on shat > shat_min for full process
        if self.sqrts_cut is not None:
            mask_shat = obs.sqrt_shat(p) > self.sqrts_cut
            mask *= mask_shat

        # Do pt > pt_min cuts
        if self.pt_cuts is not None:
            for key, value in self.pt_cuts.items():
                ids = self.key_to_ids[key]
                invalid_pt = obs.pT(p[:, ids]) < value
                mask_pt = torch.any(invalid_pt, dim=1)
                mask *= ~mask_pt

        # Do |eta| < eta_max cuts
        if self.eta_cuts is not None:
            for key, value in self.eta_cuts.items():
                ids = self.key_to_ids[key]
                invalid_eta = torch.abs(obs.eta(p[:, ids])) > value
                mask_eta = torch.any(invalid_eta, dim=1)
                mask *= ~mask_eta

        # Do dR > dR_min cuts
        if self.dR_cuts is not None:
            for key, value in self.dR_cuts.items():
                ids = self.key_to_ids[key]
                invalid_dR = obs.deltaR(p[:, ids[0]], p[:, ids[1]]) < value
                mask_dR = torch.any(invalid_dR, dim=1)
                mask *= ~mask_dR

        # Do m_inv <  cuts
        if self.mm_cuts is not None:
            for key, value in self.pt_cuts.items():
                ids = self.key_to_ids[key]
                invalid_mm = obs.minv(p[:, ids[0]], p[:, ids[1]]) < value
                mask_mm = torch.any(invalid_mm, dim=1)
                mask *= ~mask_mm
        return mask

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.cut(inputs)

    def pid_to_inttype(self, pid: int) -> int:
        if pid in JET_PIDS:
            return 100
        elif pid in [-5, 5]:
            return 101
        elif pid in LEPTON_PIDS:
            return 102
        elif pid in [22]:
            return 103
        elif pid in MISSING_PIDS:
            return 104
        else:
            return pid
