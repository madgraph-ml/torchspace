"""Kinematic functions needed for phase-space mappings"""

from math import pi

import torch
from torch import Tensor, atan2, cos, cosh, log, sin, sinh, sqrt

DTYE = torch.get_default_dtype()
EPS = 1e-12 if DTYE == torch.float64 else 1e-6


def kaellen(x: Tensor, y: Tensor, z: Tensor) -> Tensor:
    """Definition of the standard kaellen function [1]

    [1] https://en.wikipedia.org/wiki/Källén_function

    Args:
        x (Tensor): input 1
        y (Tensor): input 2
        z (Tensor): input 3

    Returns:
        Tensor: Kaellen function
    """
    return (x - y - z) ** 2 - 4 * y * z


def bk_g(x: Tensor, y: Tensor, z: Tensor, u: Tensor, v: Tensor, w: Tensor) -> Tensor:
    """Definition of the Byckling–Kajantie G-function as defined in Eq. (A5) in [1]

    [1] E. Byckling and K. Kajantie,
    ``Reductions of the phase-space integral in terms of simpler processes,''
    Phys. Rev. 187 (1969), doi:10.1103/PhysRev.187.2008.

    Args:
        x (Tensor): input 1
        y (Tensor): input 2
        z (Tensor): input 3
        u (Tensor): input 4
        v (Tensor): input 5
        w (Tensor): input 6

    Returns:
        Tensor: G function (related to Gram determinant of three 4-vectors, see (A4) in [1])
    """
    return (
        x * x * y
        + x * y * y
        + z * z * u
        + z * u * u
        + v * v * w
        + v * w * w
        + x * z * w
        + x * u * v
        + y * z * v
        + y * u * w
        - x * y * (z + u + v + w)
        - z * u * (x + y + v + w)
        - v * w * (x + y + z + u)
    )


def rotate_zy(p: Tensor, phi: Tensor, costheta: Tensor) -> Tensor:
    """Performs rotation around y- and z-axis:

        p -> p' = R_z(phi).R_y(theta).p

    with the explizit matrice following the conventions in [1]
    to achieve proper spherical coordinates [2]:

    R_z = (  1       0         0      0  )
          (  0   cos(phi)  -sin(phi)  0  )
          (  0   sin(phi)   cos(phi)  0  )
          (  0       0         0      1  )

    R_y = (  1       0       0      0       )
          (  0   cos(theta)  0  sin(theta)  )
          (  0       0       1      0       )
          (  0  -sin(theta)  0  cos(theta)  )

    For a 3D vector v = (0, 0, |v|)^T this results in the general spherical
    coordinate vector

        v -> v' = (  |v|*sin(theta)*cos(phi)  )
                  (  |v|*sin(theta)*sin(phi)  )
                  (  |v|*cos(theta)           )

    [1] https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    [2] https://en.wikipedia.org/wiki/Spherical_coordinate_system

    Args:
        p (Tensor): 4-momentum to rotate with shape=(b,...,4)
        phi (Tensor): rotation angle phi shape=(b,...)
        costheta (torch.tensor): cosine of rotation angle theta shape=(b,...)

    Returns:
        p' (Tensor): Rotated vector
    """
    sintheta = sqrt(1 - costheta**2)
    p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]

    # Define the rotation
    q0 = p0
    q1 = p1 * costheta * cos(phi) + p3 * sintheta * cos(phi) - p2 * sin(phi)
    q2 = p1 * costheta * sin(phi) + p3 * sintheta * sin(phi) + p2 * cos(phi)
    q3 = p3 * costheta - p1 * sintheta

    return torch.stack((q0, q1, q2, q3), dim=-1)


def inv_rotate_zy(p: Tensor, phi: Tensor, costheta: Tensor) -> Tensor:
    """Performs inverse rotation around y- and z-axis:

        p' -> p = R_y(-theta).R_z(-phi).p

    Args:
        p (Tensor): rotated 4-momentum inverse with shape=(b,...,4)
        phi (Tensor): rotation angle phi shape=(b,...)
        costheta (torch.tensor): cosine of rotation angle theta shape=(b,...)

    Returns:
        p' (Tensor): Rotated vector
    """
    sintheta = sqrt(1 - costheta**2)
    p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]

    # Define the rotation
    q0 = p0
    q1 = p1 * costheta * cos(phi) + p2 * costheta * sin(phi) - p3 * sintheta
    q2 = p2 * cos(phi) - p1 * sin(phi)
    q3 = p3 * costheta + p1 * sintheta * cos(phi) + p2 * sintheta * sin(phi)

    return torch.stack((q0, q1, q2, q3), dim=-1)


def rotate_z(p: Tensor, phi: Tensor) -> Tensor:
    """Performs rotation around z-axis:

        p -> p' = R_z(phi).p

    Special case of rotate_zy with theta=0.

    Args:
        p (Tensor): 4-momentum to rotate with shape=(b,...,4)
        phi (Tensor): rotation angle phi shape=(b,...)

    Returns:
        p' (Tensor): Rotated vector
    """
    p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    # Define the rotation
    q0 = p0
    q1 = p1 * cos(phi) - p2 * sin(phi)
    q2 = p1 * sin(phi) + p2 * cos(phi)
    q3 = p3

    return torch.stack((q0, q1, q2, q3), dim=-1)


def inv_rotate_z(p: Tensor, phi: Tensor) -> Tensor:
    """Performs inverserotation around z-axis:

        p' -> p = R_z(-phi).p'

    Special case of rotate_zy with theta=0.

    Args:
        p (Tensor): 4-momentum to rotate with shape=(b,...,4)
        phi (Tensor): rotation angle phi shape=(b,...)

    Returns:
        p' (Tensor): Rotated vector
    """
    p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]

    # Define the rotation
    q0 = p0
    q1 = p1 * cos(phi) + p2 * sin(phi)
    q2 = p2 * cos(phi) - p1 * sin(phi)
    q3 = p3

    return torch.stack((q0, q1, q2, q3), dim=-1)


def rotxxx(p: Tensor, q: Tensor) -> Tensor:
    """
    This function performs the spacial rotation of a four-momentum.
    The momentum p is assumed to be given in the frame where the spacial
    component of q points the positive z-axis.

    Args:
        p (Tensor): Four-momentum to rotate. Spatial part is given in the frame where q points along +z
            with shape (b,..., 4)
        q (Tensor): Four-momentum whose spatial direction defines the target frame (only q[...,1:3] used).
            with shape (b,..., 4)

    Returns:
        prot (Tensor): Rotated four-momentum p in the frame where q has its given spatial direction
            with shape (b,..., 4)
    """
    # Copy time component (pure rotation → p0 unchanged)
    p0 = p[..., 0]
    p1, p2, p3 = p[..., 1], p[..., 2], p[..., 3]

    q1, q2, q3 = q[..., 1], q[..., 2], q[..., 3]
    qt2 = pT2(q)
    qt = sqrt(qt2.clamp(min=EPS))
    qq = sqrt(pmag2(q).clamp(min=EPS))

    # General-case formulas (qt2 > 0)
    # prot(1) = q1*q3/qq/qt*p1 - q2/qt*p2 + q1/qq*p3
    # prot(2) = q2*q3/qq/qt*p1 + q1/qt*p2 + q2/qq*p3
    # prot(3) =      -qt/qq*p1            + q3/qq*p3
    prot1_gen = (q1 * q3) / (qq * qt) * p1 - (q2 / qt) * p2 + (q1 / qq) * p3
    prot2_gen = (q2 * q3) / (qq * qt) * p1 + (q1 / qt) * p2 + (q2 / qq) * p3
    prot3_gen = -(qt / qq) * p1 + (q3 / qq) * p3

    # Special case qt2 == 0  (q along ±z or zero) → Fortran does:
    #   if q3 == 0: prot(1:3) = p(1:3)
    #   else:       prot(1:3) = sign(q3) * p(1:3)
    mask_qt0 = qt2 < EPS
    psgn = torch.where(q3 >= -EPS, p1.new_tensor(1.0), p1.new_tensor(-1.0))

    prot1 = torch.where(mask_qt0, psgn * p1, prot1_gen)
    prot2 = torch.where(mask_qt0, psgn * p2, prot2_gen)
    prot3 = torch.where(mask_qt0, psgn * p3, prot3_gen)

    return torch.stack((p0, prot1, prot2, prot3), dim=-1)


def rotxxx_inv(prot: Tensor, q: Tensor) -> Tensor:
    """
    Same as rotxxx, but inverse. That is, first doing
    rotxxx(p,q) and then rotxxx_inv(prot,q) should give you
    back the original p.

    Args:
        prot (Tensor): Rotated four-momentum with shape (b,..., 4)
        q (Tensor): Reference four-momentum (only spatial part used), shape (b,...,4).

    Returns:
        p (Tensor): Four-momentum in the q-aligned (+z) frame, shape (b,...,4).
    """
    # Copy time component (pure rotation → p0 unchanged)
    p0 = prot[..., 0]
    v1, v2, v3 = prot[..., 1], prot[..., 2], prot[..., 3]

    q1, q2, q3 = q[..., 1], q[..., 2], q[..., 3]
    qt2 = pT2(q)
    qt = sqrt(qt2.clamp(min=EPS))
    qq = sqrt(pmag2(q).clamp(min=EPS))

    # R^T(q) applied to (v1,v2,v3):
    # [p1] = [ q1*q3/(qq*qt)   q2*q3/(qq*qt)   -qt/qq ] [v1]
    # [p2]   [    -q2/qt          q1/qt          0    ]  [v2]
    # [p3]   [     q1/qq          q2/qq        q3/qq  ] [v3]
    p1_gen = (q1 * q3) / (qq * qt) * v1 + (q2 * q3) / (qq * qt) * v2 - (qt / qq) * v3
    p2_gen = -(q2 / qt) * v1 + (q1 / qt) * v2
    p3_gen = (q1 / qq) * v1 + (q2 / qq) * v2 + (q3 / qq) * v3

    # Special case qt2 == 0  (q along ±z or zero) → Fortran does:
    #   if q3 == 0: prot(1:3) = p(1:3)
    #   else:       prot(1:3) = sign(q3) * p(1:3)
    mask_qt0 = qt2 < EPS
    psgn = torch.where(q3 >= -EPS, v1.new_tensor(1.0), v1.new_tensor(-1.0))

    p1 = torch.where(mask_qt0, psgn * v1, p1_gen)
    p2 = torch.where(mask_qt0, psgn * v2, p2_gen)
    p3 = torch.where(mask_qt0, psgn * v3, p3_gen)

    return torch.stack((p0, p1, p2, p3), dim=-1)


def rapidity(p: Tensor) -> Tensor:
    """Gives the rapidity of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    Es = p[..., 0]
    Pz = p[..., 3]

    y = 0.5 * log((Es + Pz) / (Es - Pz))
    return torch.where(Es < EPS, 99.0, y)


def eta(p: Tensor) -> Tensor:
    """Gives the pseudo-rapidity (eta) of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    Ps = sqrt(esquare(vec3(p)))
    Pz = p[..., 3]

    eta = 0.5 * log((Ps + Pz) / (Ps - Pz))
    return torch.where(Ps < EPS, 99.0, eta)


def phi(p: Tensor) -> Tensor:
    """Gives the azimuthal phi of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    phi = atan2(p[..., 2], p[..., 1])
    return phi


def pT2(p: Tensor) -> Tensor:
    """Gives the squared pT of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    pt2 = p[..., 1] ** 2 + p[..., 2] ** 2
    return pt2


def pT(p: Tensor) -> Tensor:
    """Gives the pT of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    return sqrt(pT2(p))


def vec3(p: Tensor) -> Tensor:
    """Gives the 3-vector of 4-mometum

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: 3-momentum with shape=(b,...,3)
    """
    return p[..., 1:]


def delta_rap(p: Tensor, q: Tensor) -> Tensor:
    """Gives Delta-rapidity between two (sets) of 4-momenta p and q

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)
        q (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: delta_y with shape=(b,...)
    """
    dy = rapidity(p) - rapidity(q)
    return torch.abs(dy)


def delta_eta(p: Tensor, q: Tensor) -> Tensor:
    """Gives Delta-eta between two (sets) of 4-momenta p and q

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)
        q (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: delta_y with shape=(b,...)
    """
    deta = eta(p) - eta(q)
    return torch.abs(deta)


def delta_phi(p: Tensor, q: Tensor) -> Tensor:
    """Gives Delta-rapidity between two (sets) of 4-momenta p and q

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)
        q (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: 3-momentum with shape=(b,...,3)
    """
    dphi = torch.abs(phi(p) - phi(q))
    dphi = torch.where(dphi > pi, 2 * pi - dphi, dphi)
    return dphi


def deltaR(p: Tensor, q: Tensor) -> Tensor:
    """Gives DeltaR between two (sets) of 4-momenta p and q

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)
        q (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: 3-momentum with shape=(b,...,3)
    """
    dy = delta_eta(p, q)
    dphi = delta_phi(p, q)
    return sqrt(dy**2 + dphi**2)


def pmag2(p: Tensor) -> Tensor:
    """Gives the squared three-momentum |p_vec|^2

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    pmag2 = esquare(p[..., 1:])
    return pmag2


def pmag(p: Tensor) -> Tensor:
    """Gives the absolute three-momentum |p_vec|

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    return sqrt(pmag2(p))


def sqrt_shat(p: Tensor) -> Tensor:
    """Gives the center-of-mass energy

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    psum = p.sum(dim=1)
    return mass(psum)


def shat(p: Tensor) -> Tensor:
    """Gives the squared center-of-mass energy

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    psum = p.sum(dim=1)
    return lsquare(psum)


def costheta(p: Tensor) -> Tensor:
    """Gives the costheta angle of a particle

    Args:
        p (Tensor): momentum 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    return p[..., 3] / pmag(p)


def minv(p1: Tensor, p2: Tensor) -> Tensor:
    """Gives invariant mass of two momenta

    Args:
        p1 (Tensor): momentum 4-vector with shape shape=(b,4)
        p2 (Tensor): momentum 4-vector with shape shape=(b,4)

    Returns:
        Tensor: minv shape=(b,)
    """
    p1p2 = p1 + p2
    return mass(p1p2)


def mass(a: Tensor) -> Tensor:
    """Gives the mass of a particle

    Args:
        a (Tensor): 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: mass with shape=(b,...)
    """
    return sqrt(torch.clip(lsquare(a), min=0.0))


def lsquare(a: Tensor) -> Tensor:
    """Gives the lorentz invariant a^2 using
    the Mikowski metric (1.0, -1.0, -1.0, -1.0)

    Args:
        a (Tensor): 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: Lorentzscalar with shape=(b,...)
    """
    a2 = a.square()
    s = a2[..., 0] - a2[..., 1] - a2[..., 2] - a2[..., 3]
    return s


def ldot(a: Tensor, b: Tensor) -> Tensor:
    """Gives the Lorentz inner product ab using
    the Mikowski metric (1.0, -1.0, -1.0, -1.0)

    Args:
        a (Tensor): 4-vector with shape shape=(b,...,4)
        b (Tensor): 4-vector with shape shape=(b,...,4)

    Returns:
        Tensor: Lorentzscalar with shape=(b,...)
    """
    ab = a * b
    return ab[..., 0] - ab[..., 1] - ab[..., 2] - ab[..., 3]


def esquare(a: Tensor) -> Tensor:
    """Gives the euclidean square a^2 using
    the Euclidean metric

    Args:
        a (Tensor): 4-vector with shape=(b,...,4)

    Returns:
        Tensor: Square with shape=(b,...)
    """
    return torch.einsum("...d,...d->...", a, a)


def edot(a: Tensor, b: Tensor) -> Tensor:
    """Gives the euclidean inner product ab using
    the Euclidean metric

    Args:
        a (Tensor): 4-vector with shape=(b,...,4)
        b (Tensor): 4-vector with shape=(b,...,4)

    Returns:
        Tensor: Lorentzscalar with shape=(b,...)
    """
    return torch.einsum("...d,...d->...", a, b)


def boost(k: Tensor, p_boost: Tensor, inverse: bool = False) -> Tensor:
    """
    Boost k into the frame of p_boost in argument.
    This means that the following command, for any vector k=(E, px, py, pz)
    gives:

        k  -> k' = boost(k, k, inverse=True) = (M,0,0,0)
        k' -> k  = boost(k', k) = (E, px, py, pz)

    Args:
        k (Tensor): input vector with shape=(b,n,4)/(b,4)
        p_boost (Tensor): boosting vector with shape=(b,1,4)/(b,4)
        inverse (bool): if boost is performed inverse or forward

    Returns:
        k' (Tensor): boosted vector with shape=(b,n,4)/(b,4)
    """
    # Change sign if inverse boost is performed
    sign = -1.0 if inverse else 1.0

    # Perform the boost
    # This is in fact a numerical more stable implementation then often used
    rsq = torch.clip(mass(p_boost), min=EPS)
    k0 = (k[..., 0] * p_boost[..., 0] + sign * edot(k[..., 1:], p_boost[..., 1:])) / rsq
    c1 = (k[..., 0] + k0) / (rsq + p_boost[..., 0])
    k1 = k[..., 1] + sign * c1 * p_boost[..., 1]
    k2 = k[..., 2] + sign * c1 * p_boost[..., 2]
    k3 = k[..., 3] + sign * c1 * p_boost[..., 3]

    return torch.stack((k0, k1, k2, k3), dim=-1)


def boost_beam(
    q: Tensor,
    rapidity: Tensor,
    inverse: bool = False,
) -> Tensor:
    """Boosts q along the beam axis with given rapidity

    Args:
        q (Tensor): input vector with shape=(b,n,4)/(b,4)
        rapidity (Tensor): boosting parameter with shape=(b,1)/(b,)
        inverse (bool, optional): inverse boost. Defaults to False.

    Returns:
        q' (Tensor): boosted vector with shape=(b,n,4)
    """
    sign = -1.0 if inverse else 1.0

    pi0 = q[..., 0] * cosh(rapidity) + sign * q[..., 3] * sinh(rapidity)
    pix = q[..., 1]
    piy = q[..., 2]
    piz = q[..., 3] * cosh(rapidity) + sign * q[..., 0] * sinh(rapidity)

    return torch.stack((pi0, pix, piy, piz), dim=-1)
