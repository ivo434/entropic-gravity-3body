"""Spherical-grid integrators for entropic-force integrals.

All integrals are of the form ∫ d³r · f(r) with f built from Lorentzian
factors 1/(|r−x_i|² + a²).  We use a radial map u = r/(r+R0) (midpoint rule
in u, θ, φ) centered at the centroid of the source points, which resolves
both the near-field peaks and the 1/r^n tails.
"""

import numpy as np


def integrate3d(f, center, R0, nr=220, nth=96, nph=96):
    """Integrate f(X, Y, Z) over R³ on a spherical grid centered at `center`.

    Radial map: u = r/(r+R0), u ∈ (0,1) midpoints → r = R0·u/(1−u),
    dr = R0/(1−u)² du.  Midpoint rule in all three variables.
    """
    u = (np.arange(nr) + 0.5) / nr
    r = R0 * u / (1 - u)
    dr = R0 / (1 - u) ** 2 * (1 / nr)
    th = (np.arange(nth) + 0.5) * np.pi / nth
    dth = np.pi / nth
    ph = (np.arange(nph) + 0.5) * 2 * np.pi / nph
    dph = 2 * np.pi / nph
    Rg, Tg, Pg = np.meshgrid(r, th, ph, indexing="ij")
    X = Rg * np.sin(Tg) * np.cos(Pg) + center[0]
    Y = Rg * np.sin(Tg) * np.sin(Pg) + center[1]
    Z = Rg * np.cos(Tg) + center[2]
    W = (Rg**2 * np.sin(Tg)) * dr[:, None, None] * dth * dph
    return np.sum(f(X, Y, Z) * W)


def lorentz(X, Y, Z, pos, a):
    """Regulated Lorentzian factor 1/(|r − pos|² + a²)."""
    return 1.0 / ((X - pos[0]) ** 2 + (Y - pos[1]) ** 2 + (Z - pos[2]) ** 2 + a**2)


def I2(pos_i, pos_j, a, nr=220, nth=96, nph=96, R0=None):
    """I₂ = ∫ d³r / [(|r−x_i|²+a²)(|r−x_j|²+a²)].

    Exact: π³·s(d)/d with s(d) = 1 − (2/π)·arctan(2a/d).
    """
    pos_i = np.asarray(pos_i, dtype=float)
    pos_j = np.asarray(pos_j, dtype=float)
    d = np.linalg.norm(pos_i - pos_j)
    if R0 is None:
        R0 = 2 * d
    center = 0.5 * (pos_i + pos_j)
    f = lambda X, Y, Z: lorentz(X, Y, Z, pos_i, a) * lorentz(X, Y, Z, pos_j, a)
    return integrate3d(f, center, R0, nr, nth, nph)


def I2_exact(d, a):
    """Closed form of I₂: π³·s(d)/d, s(d) = 1 − (2/π)·arctan(2a/d)."""
    return np.pi**3 * (1 - (2 / np.pi) * np.arctan(2 * a / d)) / d


def I3(pos1, pos2, pos3, a, nr=220, nth=96, nph=96, R0=None):
    """I₃ = ∫ d³r · Π_{i=1..3} 1/(|r−x_i|²+a²)."""
    pos = [np.asarray(p, dtype=float) for p in (pos1, pos2, pos3)]
    center = sum(pos) / 3.0
    if R0 is None:
        dmax = max(np.linalg.norm(pos[i] - pos[j]) for i in range(3) for j in range(i + 1, 3))
        R0 = 2 * dmax
    f = lambda X, Y, Z: (
        lorentz(X, Y, Z, pos[0], a) * lorentz(X, Y, Z, pos[1], a) * lorentz(X, Y, Z, pos[2], a)
    )
    return integrate3d(f, center, R0, nr, nth, nph)


def J(pos_i, pos_j, a, nr=220, nth=96, nph=96, R0=None):
    """J = ∫ d³r / [(|r−x_i|²+a²)²·(|r−x_j|²+a²)].

    Squared factor on site i (the ℓ_i² leg of the anomalous two-body term).
    """
    pos_i = np.asarray(pos_i, dtype=float)
    pos_j = np.asarray(pos_j, dtype=float)
    d = np.linalg.norm(pos_i - pos_j)
    if R0 is None:
        R0 = 2 * d
    center = 0.5 * (pos_i + pos_j)
    f = lambda X, Y, Z: lorentz(X, Y, Z, pos_i, a) ** 2 * lorentz(X, Y, Z, pos_j, a)
    return integrate3d(f, center, R0, nr, nth, nph)
