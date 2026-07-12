"""Non-perturbative (Möbius-subtracted) interaction free energies.

Full free energy per site g(ω) = −T·ln(1+e^{(μ−ω)/T}); the interacting
pieces are extracted by Möbius/inclusion-exclusion subtraction, which
cancels all self-energies and is FINITE (unlike the perturbative ω³
two-body term J, which diverges as 1/a):

  A₂(d)  = ∫d³r/a³ [ g(ω₁+ω₂) − g(ω₁) − g(ω₂) + g(0) ]
  A₃     = ∫d³r/a³ [ g(Σω) − Σ_pairs g(ωᵢ+ωⱼ) + Σᵢ g(ωᵢ) − g(0) ]

Saturation: within r* = √(ℓ/T) of a point mass, ω ≫ T and g'(ω) → 0
(Pauli-blocked sites). The anomalous two-body correction beyond Newton is
governed by the 1D "saturation moment"

  S(ℓ) = 4π ∫₀^∞ dr r² [ σ(ω(r)) − σ* + (σ*(1−σ*)/T)·ω(r) ],
  ω(r) = ℓ/(r²+a²),  σ(ω) = 1/(1+e^{(ω−μ)/T}),

which is finite at both ends, and for widely separated masses (linear
response in the partner's field, w = ℓ_partner/d² ≪ T):

  A₂(d) = −[σ*(1−σ*)/T]·(ℓ₁ℓ₂/a³)·I₂(d,a)  +  [ℓ₂S(ℓ₁)+ℓ₁S(ℓ₂)]/(a³d²)

Perturbative limit (ℓ/a² ≪ T): S → 3c₃·ℓ²·π²/a, reproducing the J-term.
Saturated limit (r* ≫ a): S ~ O(σ-dependent)·r*³, so the effective UV cutoff
of the anomalous term is r*, not a.
"""

import numpy as np

from .integrators import integrate3d


def g_free(omega, T, mu):
    """g(ω) = −T·ln(1+e^{(μ−ω)/T}), stable for μ<0, ω≥0 (arg always <0)."""
    return -T * np.log1p(np.exp((mu - omega) / T))


def sigma_of(omega, T, mu):
    """Occupation σ(ω) = 1/(1+e^{(ω−μ)/T}), overflow-safe (expit)."""
    from scipy.special import expit

    return expit(-(omega - mu) / T)


def _omega(X, Y, Z, pos, ell, a):
    return ell / ((X - pos[0]) ** 2 + (Y - pos[1]) ** 2 + (Z - pos[2]) ** 2 + a**2)


def A2_full(pos1, pos2, ell1, ell2, T, mu, a, nr=330, nth=144, nph=144,
            R0=None, r_max_over_d=150.0):
    """Möbius-subtracted two-body free energy, full g, 3D grid.

    The subtracted integrand K ~ 1/r⁴ in the far zone while per-point
    float64 cancellation noise is constant (~eps·|g(0)|); noise×r³ would
    swamp the signal if the radial map ran to infinity. We therefore
    truncate at r_max = r_max_over_d·d (noise there is still negligible)
    and add the analytic perturbative tail −(σ*(1−σ*)/T)·ℓ₁ℓ₂·4π/r_max,
    exact to O(d²/r_max²) ≈ 4e-5.
    """
    pos1 = np.asarray(pos1, float)
    pos2 = np.asarray(pos2, float)
    d = np.linalg.norm(pos1 - pos2)
    if R0 is None:
        R0 = 2 * d
    center = 0.5 * (pos1 + pos2)
    r_max = r_max_over_d * d

    def integrand(X, Y, Z):
        R2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        w1 = _omega(X, Y, Z, pos1, ell1, a)
        w2 = _omega(X, Y, Z, pos2, ell2, a)
        g0 = g_free(np.zeros_like(w1), T, mu)
        K = g_free(w1 + w2, T, mu) - g_free(w1, T, mu) - g_free(w2, T, mu) + g0
        return np.where(R2 < r_max**2, K, 0.0)

    core = integrate3d(integrand, center, R0, nr, nth, nph)
    s = 1.0 / (1.0 + np.exp(-mu / T))
    tail = -(s * (1 - s) / T) * ell1 * ell2 * 4 * np.pi / r_max
    return (core + tail) / a**3


def A3_full(positions, ells, T, mu, a, nr=330, nth=144, nph=144, R0=None,
            r_max_over_d=25.0):
    """Möbius-subtracted three-body free energy, full g, 3D grid.

    Far tail ~ 1/r⁶ (triple product): truncating at 25·d_max leaves a
    relative tail below 1e-4 while keeping roundoff-noise×volume negligible
    (same mechanism as in A2_full).
    """
    pos = [np.asarray(p, float) for p in positions]
    center = sum(pos) / 3.0
    dmax = max(np.linalg.norm(pos[i] - pos[j]) for i in range(3) for j in range(i + 1, 3))
    if R0 is None:
        R0 = 2 * dmax
    r_max = r_max_over_d * dmax

    def integrand(X, Y, Z):
        R2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        w = [_omega(X, Y, Z, pos[i], ells[i], a) for i in range(3)]
        g0 = g_free(np.zeros_like(w[0]), T, mu)
        val = g_free(w[0] + w[1] + w[2], T, mu)
        for i in range(3):
            for j in range(i + 1, 3):
                val -= g_free(w[i] + w[j], T, mu)
        for i in range(3):
            val += g_free(w[i], T, mu)
        return np.where(R2 < r_max**2, val - g0, 0.0)

    return integrate3d(integrand, center, R0, nr, nth, nph) / a**3


def _S_integrand(w, T, mu):
    """σ(ω) − σ* + σ*(1−σ*)ω/T, evaluated stably.

    For ω/T < 0.05 the direct form suffers catastrophic cancellation (the
    result is O(ω²/T²) while the terms are O(1)); switch to the Taylor
    series (sympy-derived):
      c₂ω² + c₃ω³ + c₄ω⁴,  cₙ in terms of s = σ*:
      c₂ = s(1−s)(1−2s)/(2T²)
      c₃ = −s(1−s)(6s²−6s+1)/(6T³)
      c₄ = s(1−s)(1−2s)(12s²−12s+1)/(24T⁴)
    Relative series error at the threshold ~ (ω/T)³ ≈ 1e-4.
    """
    s = 1.0 / (1.0 + np.exp(-mu / T))
    x = w / T
    exact = sigma_of(w, T, mu) - s + s * (1 - s) * x
    c2 = s * (1 - s) * (1 - 2 * s) / 2
    c3 = -s * (1 - s) * (6 * s**2 - 6 * s + 1) / 6
    c4 = s * (1 - s) * (1 - 2 * s) * (12 * s**2 - 12 * s + 1) / 24
    series = c2 * x**2 + c3 * x**3 + c4 * x**4
    return np.where(x < 0.05, series, exact)


def S_moment(ell, T, mu, a, n=6000):
    """Saturation moment S(ℓ) by 1D log-grid quadrature (midpoint in ln r).

    S = 4π∫dr r²[σ(ω)−σ* + σ*(1−σ*)ω/T]. Integrand ~ −σ* + σ*(1−σ*)ω/T for
    r ≪ r* (saturated), ~ 3c₃ω² for ω ≪ T (perturbative tail ~ r⁻⁴·r²).
    """
    rstar = np.sqrt(ell / T)
    r_lo = min(a, rstar) * 1e-4
    r_hi = max(a, rstar) * 1e6
    u = np.linspace(np.log(r_lo), np.log(r_hi), n + 1)
    um = 0.5 * (u[1:] + u[:-1])
    du = np.diff(u)
    r = np.exp(um)
    w = ell / (r**2 + a**2)
    return 4 * np.pi * np.sum(_S_integrand(w, T, mu) * r**3 * du)


def S_perturbative(ell, T, mu, a):
    """Perturbative limit of S (valid for ℓ/a² ≪ T): 3c₃·ℓ²·π²/a."""
    s = 1.0 / (1.0 + np.exp(-mu / T))
    c3 = s * (1 - s) * (1 - 2 * s) / (6 * T**2)
    return 3 * c3 * ell**2 * np.pi**2 / a


def A2_anomalous(d, ell1, ell2, T, mu, a):
    """Anomalous part of A₂ beyond Newton: [ℓ₂S(ℓ₁)+ℓ₁S(ℓ₂)]/(a³d²).

    Valid for d ≫ r*, a (linear response in the partner field).
    """
    return (ell2 * S_moment(ell1, T, mu, a) + ell1 * S_moment(ell2, T, mu, a)) / (
        a**3 * d**2
    )
