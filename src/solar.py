"""Sun-Earth-Moon perturbations from the ω³ term, and the LLR bound.

All ω³ effects at solar-system separations use the a→0 closed form
I₃ = π³/(d₁₂·d₁₃·d₂₃) integrated over the mass distributions:

  V₃(SEM)  = A · M_S·M_E·M_M / (d_ES·d_EM·d_MS)          (pure 3-body)
  W_iij(d) = (A/2) · mᵢ²·mⱼ · Xᵢ / d²                     (anomalous 2-body)

with A ≡ (1−2σ*)·G_N·L²/T and Xᵢ = ⟨1/r₁₂⟩ over body i:
  Xᵢ = 6/(5Rᵢ)   uniform sphere of radius Rᵢ  (extended, defensible)
  Xᵢ = 1/(π·a)   point mass regulated by the lattice spacing (spec baseline;
                 cutoff-dominated — the ω-expansion is invalid at r ~ a from
                 a point mass, so this massively overestimates the effect)

Both follow from the same object: W_iij = 3·c₃·(π³L⁶/a³)·mᵢ²mⱼ·⟨1/r₁₂⟩ᵢ/d²,
and for coincident points the regulated ⟨1/r₁₂⟩ is J·d²/π³ = 1/(π·a).

Newton identification: L⁴ = G_N·T·a³/(σ*(1−σ*)·π³)  →  A ∝ √(a³/T),
so every ω³ observable scales as √(a³/T)·(1−2σ*)/√(σ*(1−σ*)).
"""

import numpy as np

from .model import (G_NEWTON, M_SUN, M_EARTH, M_MOON, D_EM, D_ES, METER,
                    L4_from_GN, SIGMA_ENTROPIC)

R_SUN = 6.96e8 * METER
R_EARTH = 6.371e6 * METER
R_MOON = 1.7374e6 * METER


def prefactor_A(sigma, T, a):
    """A = (1−2σ*)·G_N·L²/T with L⁴ fixed by the Newton identification."""
    L2 = np.sqrt(L4_from_GN(sigma, T, a))
    return (1 - 2 * sigma) * G_NEWTON * L2 / T


def geometry(phi):
    """Earth at origin, Sun at (D_ES,0,0), Moon at D_EM·(cosφ, sinφ, 0).

    φ = Sun-Earth-Moon angle (lunar phase). Returns positions (GeV⁻¹).
    """
    xE = np.array([0.0, 0.0, 0.0])
    xS = np.array([D_ES, 0.0, 0.0])
    xM = D_EM * np.array([np.cos(phi), np.sin(phi), 0.0])
    return xS, xE, xM


def _unit(v):
    n = np.linalg.norm(v)
    return v / n, n


def accel_pair(x1, x2, m1, m2, C):
    """Accelerations of bodies 1 and 2 from a pair potential W = C/d².

    Returns (a1, a2); repulsive for C > 0.
    """
    u12, d = _unit(x1 - x2)
    f = 2 * C / d**3
    return (f / m1) * u12, -(f / m2) * u12


def accel_V3(xS, xE, xM, mS, mE, mM, C3):
    """Accelerations of Moon and Earth from V₃ = C₃/(d_ES·d_EM·d_MS)."""
    uME, dEM = _unit(xM - xE)
    uMS, dMS = _unit(xM - xS)
    uES, dES = _unit(xE - xS)
    aM = (C3 / mM) * (uME / (dEM**2 * dES * dMS) + uMS / (dMS**2 * dEM * dES))
    aE = (C3 / mE) * (-uME / (dEM**2 * dES * dMS) + uES / (dES**2 * dEM * dMS))
    return aM, aE


def perturbation(phi, sigma, T, a, extended=True):
    """δa_rel = δa_M − δa_E from all ω³ terms, at lunar phase φ.

    Returns dict with the vector, its radial (along Earth→Moon) component,
    the Newtonian relative acceleration, and per-component breakdown.
    """
    xS, xE, xM = geometry(phi)
    A = prefactor_A(sigma, T, a)

    if extended:
        XS, XE, XM = 6 / (5 * R_SUN), 6 / (5 * R_EARTH), 6 / (5 * R_MOON)
    else:
        XS = XE = XM = 1 / (np.pi * a)

    # pair coefficients C = (A/2)·(mᵢ²mⱼXᵢ + mⱼ²mᵢXⱼ)
    C_SE = (A / 2) * (M_SUN**2 * M_EARTH * XS + M_EARTH**2 * M_SUN * XE)
    C_SM = (A / 2) * (M_SUN**2 * M_MOON * XS + M_MOON**2 * M_SUN * XM)
    C_EM = (A / 2) * (M_EARTH**2 * M_MOON * XE + M_MOON**2 * M_EARTH * XM)
    C3 = A * M_SUN * M_EARTH * M_MOON

    aM_SM, _ = accel_pair(xM, xS, M_MOON, M_SUN, C_SM)
    aE_SE, _ = accel_pair(xE, xS, M_EARTH, M_SUN, C_SE)
    aM_EM, aE_EM = accel_pair(xM, xE, M_MOON, M_EARTH, C_EM)
    aM_3, aE_3 = accel_V3(xS, xE, xM, M_SUN, M_EARTH, M_MOON, C3)

    parts = {
        "SSM (Sun²·Moon pair)": aM_SM,
        "SSE (Sun²·Earth pair)": -aE_SE,
        "EM pair (E²M + M²E)": aM_EM - aE_EM,
        "V3 (pure 3-body)": aM_3 - aE_3,
    }
    total = sum(parts.values())
    u_EM, _ = _unit(xM - xE)
    aN = G_NEWTON * (M_EARTH + M_MOON) / D_EM**2
    return {
        "vec": total,
        "radial": float(np.dot(total, u_EM)),
        "aN": aN,
        "parts": parts,
    }


def eps_modulation(sigma, T, a, extended=True, nphi=360):
    """Mean and modulation amplitude of ε(φ) = δa_radial/a_N over a month."""
    phis = np.linspace(0, 2 * np.pi, nphi, endpoint=False)
    eps = np.array(
        [perturbation(p, sigma, T, a, extended)["radial"] for p in phis]
    )
    aN = G_NEWTON * (M_EARTH + M_MOON) / D_EM**2
    eps = eps / aN
    F = np.fft.rfft(eps) / nphi
    return {
        "mean": float(np.mean(eps)),
        "amp": float((eps.max() - eps.min()) / 2),
        "fourier_c1": float(2 * np.abs(F[1])),  # monthly (synodic) harmonic
        "fourier_c2": float(2 * np.abs(F[2])),  # fortnightly — dominates (tidal)
        "eps": eps,
        "phis": phis,
    }
