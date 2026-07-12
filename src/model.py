"""Physical model: local entropic-gravity qubit lattice (Carney et al. 2025).

Conventions (audited from first principles, see writeup/notes_signs.md):

  g(ω) = g(0) + σ*·ω − [σ*(1−σ*)/(2T)]·ω²
              + [σ*(1−σ*)(1−2σ*)/(6T²)]·ω³ + O(ω⁴)

  ω_α = Σ_i ℓ_i / (|r_α − x_i|² + a²),   ℓ_i = m_i L²

Continuum limit Σ_α → ∫ d³r / a³:

  V₂(d)  = −[σ*(1−σ*)/T] · (ℓ_i ℓ_j / a³) · I₂(d)                (Newton)
  V₃     = +[σ*(1−σ*)(1−2σ*)/T²] · (ℓ₁ℓ₂ℓ₃ / a³) · I₃           (pure 3-body)
  δV₂(d) = +[σ*(1−σ*)(1−2σ*)/(2T²)] · (ℓ_i²ℓ_j + ℓ_iℓ_j²)/a³ · J(d)

(The 3·(ℓ_i²ℓ_j·J_ij + ℓ_iℓ_j²·J_ji) cross terms of the cube carry the
1/6 prefactor → 1/2 overall; by symmetry J_ij = J_ji ≡ J(d) for the
two-body case, giving the (ℓ_i²ℓ_j + ℓ_iℓ_j²) combination above.)

Newton identification (paper Eq. 33):
  G_N = σ*(1−σ*)·π³·L⁴ / (T·a³)

All internal units: natural, ħ = c = k_B = 1, energies in GeV.
"""

import numpy as np

# ---------------------------------------------------------------- constants
KG = 5.610e26        # 1 kg in GeV
METER = 5.068e15     # 1 m in GeV⁻¹
SECOND = 1.519e24    # 1 s in GeV⁻¹
KELVIN = 8.617e-14   # 1 K in GeV

G_NEWTON = 6.709e-39  # GeV⁻²

M_SUN = 1.989e30 * KG
M_EARTH = 5.972e24 * KG
M_MOON = 7.342e22 * KG
D_EM = 3.844e8 * METER
D_ES = 1.496e11 * METER

A_MIN = 1e-13 * METER          # anomalous-heating bound on lattice spacing
T_300K = 300 * KELVIN

# Purely entropic point of the paper: μ = −2.39936·T
MU_OVER_T_ENTROPIC = -2.39936
SIGMA_ENTROPIC = 1.0 / (np.exp(-MU_OVER_T_ENTROPIC) + 1.0)  # 0.0832215


def sigma_star(mu_over_T):
    """Mean polarization σ* = 1/(e^{−μ/T} + 1)."""
    return 1.0 / (np.exp(-mu_over_T) + 1.0)


def c2(sigma, T):
    """Quadratic coefficient of g(ω): −σ*(1−σ*)/(2T)."""
    return -sigma * (1 - sigma) / (2 * T)


def c3(sigma, T):
    """Cubic coefficient of g(ω): +σ*(1−σ*)(1−2σ*)/(6T²)."""
    return sigma * (1 - sigma) * (1 - 2 * sigma) / (6 * T**2)


def L4_from_GN(sigma, T, a):
    """Invert the Newton identification: L⁴ = G_N·T·a³/(σ*(1−σ*)·π³)."""
    return G_NEWTON * T * a**3 / (sigma * (1 - sigma) * np.pi**3)


def V2(d, ell_i, ell_j, sigma, T, a):
    """Two-body (Newtonian) potential from the ω² term, closed form for I₂."""
    from .integrators import I2_exact

    return -(sigma * (1 - sigma) / T) * (ell_i * ell_j / a**3) * I2_exact(d, a)
