""" Implement t- channel block 
    from the chilli/pepper paper
"""


from typing import Tuple, Optional
import torch
from math import pi
from torch import Tensor, sqrt, log, sinh, cos, sin, exp

from .base import PhaseSpaceMapping, TensorList
from .functional.kinematics import rapidity, pT2, lsquare, phi, mass


class tChiliBlock(PhaseSpaceMapping):
    """
    Get generic 2 -> n t-channel block as implemented in Chili [1] and Pepper [2]

        [1] Paper: https://arxiv.org/abs/2302.10449
            Code:  https://gitlab.com/spice-mc/Chili

        [2] Paper: https://arxiv.org/abs/2311.06198
            Code:  https://gitlab.com/spice-mc/pepper
    """

    def __init__(self, nparticles: int, ymax: Tensor, ptmin: Tensor):
        """
        Args:
            nparticles (int): Number of outgoing particles
            ymax (Tensor): integration limits for rapidities with shape=(b,n)
            ptmin (Tensor): integration limits for pt with shape=(b,n)
        """
        self.nout = nparticles
        self.ymax = ymax
        self.ptmin = ptmin
        dof = 3 * self.nout - 2
        dims_in = [(dof,), (), (nparticles,)]
        dims_out = [(nparticles, 4)]
        super().__init__(dims_in, dims_out)

    def map(self, inputs: TensorList, condition=None):
        """Map from random numbers to momenta

        Args:
            inputs (TensorList): list of two tensors [r, e_cm, m_out]
                r: random numbers with shape=(b,3*np - 2)
                e_cm: COM energy with shape=(b,)
                m_out: (virtual) masses of outgoing particles
                    with shape=(b,np)

        Returns:
            p_ext (Tensor): external momenta (lab frame) with shape=(b,np,4)
            x1x2 (Tensor): pdf fractions with shape=(b,2)
            det (Tensor): log det of mapping with shape=(b,)
        """
        del condition
        r, e_cm, m_out = inputs[0], inputs[1], inputs[2]
        p_nm1 = torch.zeros(r.shape[0], self.nout - 1, 4, device=r.device)
        p_n = torch.zeros(r.shape[0], 1, 4, device=r.device)
        p_in = torch.zeros(r.shape[0], 2, 4, device=r.device)

        # Get individual rands
        r_pt = r[:, : self.nout - 1]
        r_y = r[:, self.nout - 1 : 2 * self.nout - 1]
        r_phi = r[:, 2 * self.nout - 1 :]

        # --------------------------------------------------
        # Handle the first n-1 outgoing t-channel particles

        # get the pts
        ptmin = self.ptmin
        ptmax = e_cm[:, None] / 2  # generally too large, thats why we get x1x2 > 1 and NaNs
        ptc = torch.where(m_out[..., :-1] > 0.0, m_out[..., :-1], ptmin[:-1])
        delta_pt = (ptmax - ptmin)[..., :-1]
        pt_denom = 2 * ptc + ptmax * (1 - r_pt)
        pt = ptmin[:-1] + (2 * ptc * delta_pt * r_pt) / pt_denom
        det_pt = ((2 * ptc * delta_pt * (2 * ptc + ptmax)) / pt_denom**2).prod(dim=1)

        # get first n-1 rapidities and phi
        pt2 = pt**2
        ymax = log(sqrt(e_cm[:, None]**2 / 4 / pt2) + sqrt(e_cm[:, None]**2 / 4 / pt2 - 1.0))
        ymax = torch.minimum(ymax, self.ymax[:-1])
        y_nm1 = ymax * (2 * r_y[:, :-1] - 1.0)
        det_y = torch.prod(2 * ymax, dim=1)
        phi_nm1 = 2 * pi * r_phi
        det_phi = (2 * pi) ** (self.nout - 1)

        # calculate the momenta
        sinhy = sinh(y_nm1)
        coshy = sqrt(1 + sinhy**2)
        mt = sqrt(pt2 + m_out[:, :-1] ** 2)
        p_nm1[:, :, 0] = mt * coshy
        p_nm1[:, :, 1] = pt * cos(phi_nm1)
        p_nm1[:, :, 2] = pt * sin(phi_nm1)
        p_nm1[:, :, 3] = mt * sinhy

        # get the sum
        psum = p_nm1.sum(dim=1)

        # --------------------------------------------------
        # Handle initial states and last t-channel momentum

        # Build last t-channel momentum
        mj2 = lsquare(psum)
        yj = rapidity(psum)
        ptj2 = pT2(psum)
        m2 = m_out[:, -1] ** 2
        qt = sqrt(ptj2 + m2)
        mt = sqrt(ptj2 + mj2)
        yminn = -log(e_cm / qt * (1.0 - mt / e_cm * exp(-yj)))
        yminn = torch.maximum(yminn, -self.ymax[-1])  # apply potential cuts
        ymaxn = log(e_cm / qt * (1.0 - mt / e_cm * exp(yj)))
        ymax = torch.minimum(ymaxn, self.ymax[-1])  # apply potential cuts
        dely = ymaxn - yminn
        yn = yminn + r_y[:, 1] * dely
        det_yn = dely
        sinhyn = sinh(yn)
        coshyn = sqrt(1 + sinhyn**2)
        p_n[:, 0, 0] = qt * coshyn
        p_n[:, 0, 1] = -psum[:, 1]
        p_n[:, 0, 2] = -psum[:, 2]
        p_n[:, 0, 3] = qt * sinhyn

        p_out = torch.cat([p_nm1, p_n], dim=1)

        # Build incoming momenta
        psum += p_n[:, 0]
        pp = psum[:, 0] + psum[:, 3]
        pm = psum[:, 0] - psum[:, 3]
        p_in[:, 0, 0] = pp / 2
        p_in[:, 0, 3] = pp / 2
        p_in[:, 1, 0] = pm / 2
        p_in[:, 1, 3] = -pm / 2

        # Get the bjorken variables
        x1 = pp / e_cm
        x2 = pm / e_cm

        # Get all momenta
        #p_ext = torch.cat([p_in, p_out], dim=1)
        x1x2 = torch.stack([x1, x2], dim=1)
        det = det_pt * det_y * det_phi * det_yn

        # keep invalid point but set det/weight to zero (important to obtain correct integral)
        mask_x1x2 = x1x2.prod(dim=1) < 1.0  # covers both this and NaNs!
        det[~mask_x1x2] = 0.0

        return (p_in, p_out), det

    def map_inverse(self, inputs: TensorList, condition=None):
        """Inverse map from decay momenta onto random numbers

        Args:
            inputs (TensorList): list with only one tensor [p_decay]
                p_ext: decay momenta (lab frame) with shape=(b,n+2,4)
                x1x2 (Tensor): pdf fractions with shape=(b,2)

        Returns:
            r (Tensor): random numbers with shape=(b,2)
            e_cm (Tensor): squared COM energy with shape=(b,)
            m_out (Tensor): (virtual) masses of outgoing particles with shape=(b,np)
            det (Tensor): log det of mapping with shape=(b,)
        """
        del condition
        p_ext = inputs[0]
        x1x2 = inputs[1]

        # Make sure to only invert valid points
        mask_x1x2 = x1x2.prod(dim=1) < 1.0
        p_ext = p_ext[mask_x1x2]
        ptmin_masked = self.ptmin[mask_x1x2]
        ymax_masked = self.ymax[mask_x1x2]

        # get initial an final state particles
        p_in = p_ext[:, :2]
        p_out = p_ext[:, 2:]
        s = lsquare(p_in.sum(dim=1))
        e_cm = sqrt(s)
        m_out = mass(p_out)

        # Get relevant psum and energies etc
        psum_nm1 = p_out[:, :-1]  # sum over final states except last particle
        m2n = lsquare(p_out[:, -1])  # last particle mass

        # --------------------------------------------------
        # Handle last t-channel momentum

        mj2 = lsquare(psum_nm1)
        yj = rapidity(psum_nm1)
        ptj2 = pT2(psum_nm1)
        qt = sqrt(ptj2 + m2n)
        mt = sqrt(ptj2 + mj2)
        yminn = -log(e_cm / qt * (1.0 - mt / e_cm * exp(-yj)))
        yminn = torch.maximum(yminn, -ymax_masked[:, -1])  # apply potential cuts
        ymaxn = log(e_cm / qt * (1.0 - mt / e_cm * exp(yj)))
        ymaxn = torch.minimum(ymaxn, ymax_masked[:, -1])  # apply potential cuts
        dely = ymaxn - yminn

        # extract random number
        p_n = p_out[:, -1]
        yn = rapidity(p_n)
        r_yn = (yn - yminn) / dely
        det_yn = 1 / dely

        # --------------------------------------------------
        # Handle other n-1 momenta

        p_nm1 = p_out[:, :-1]
        pt2_nm1 = pT2(p_nm1)
        y_nm1 = rapidity(p_nm1)
        phi_nm1 = phi(p_nm1)

        # get random numbers for phi
        r_phi = phi_nm1 / 2 / pi
        det_phi = (2 * pi) ** (1 - self.nout)

        # get random numbers for eta
        ymax = log(sqrt(s / 4 / pt2_nm1) + sqrt(s / 4 / pt2_nm1 - 1.0))
        ymax = torch.minimum(ymax, ymax_masked[:, :-1])
        r_y = 0.5 * (y_nm1 / ymax + 1.0)
        det_y = torch.prod(1 / (2 * ymax), dim=1)

        # get random numbers for pt
        ptmin = ptmin_masked
        ptmax = e_cm / 2
        delta_pt = ptmax - ptmin
        ptc = torch.where(m_out[..., :-1] > 0.0, m_out[..., :-1], ptmin)
        pt_nm1 = sqrt(pt2_nm1)
        nom = (2 * ptc + ptmax) * (pt_nm1 - ptmin)
        denom = 2 * delta_pt * ptc + ptmax * (pt_nm1 - ptmin)
        r_pt = nom / denom
        det_pt = ((2 * delta_pt * ptc * (2 * ptc + ptmax)) / denom**2).prod(dim=1)

        # Collect everything
        r = torch.stack([r_pt, r_y, r_yn[..., None], r_phi])
        det = det_pt * det_y * det_phi * det_yn

        return (r, e_cm, m_out), det

    def density(self, inputs: TensorList, condition=None, inverse=False):
        del condition
        if inverse:
            _, det = self.map_inverse(inputs)
            return det

        _, det = self.map(inputs)
        return det
