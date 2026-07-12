"""Closed forms for the integrals, derived analytically.

Derivations (see writeup/three_body_entropic.md for details):

1. Generalized two-center integral, distinct regulators a, b (3D Fourier,
   FT[1/(r²+a²)] = 2π²·e^{−a|k|}/|k|):

     I₂(d; a, b) = ∫ d³r / [(|r−x₁|²+a²)(|r−x₂|²+b²)]
                 = (2π²/d)·arctan(d/(a+b))

   For a = b this reduces to π³·s(d)/d with s(d) = 1 − (2/π)arctan(2a/d).

2. J from differentiation w.r.t. the regulator of the first factor:

     J(d, a) = −∂I₂(d; a, b)/∂(a²) |_{b=a} = π² / [a·(d² + 4a²)]

   Linearly divergent as a → 0 (the ∫d³ρ/ρ⁴ singularity at the squared
   center); the lattice spacing a is the physical regulator.
   Leading behavior: J ≈ π²/(a·d²).

3. Three-center integral at a = 0 via the star-triangle (uniqueness)
   relation in D dimensions: for Σδᵢ = D,

     ∫ d^D x₀ Π 1/|x₀−xᵢ|^{2δᵢ}
       = π^{D/2} · Π[Γ(D/2−δᵢ)/Γ(δᵢ)] · Π |x_ij|^{−2(D/2−δ_k)}

   With D = 3, δ₁ = δ₂ = δ₃ = 1 the uniqueness condition holds exactly
   (Σδ = 3 = D), Γ(1/2) = √π, and

     I₃(x₁, x₂, x₃; a=0) = π³ / (d₁₂ · d₁₃ · d₂₃)

   In particular I₃·d³ = π³ for the equilateral triangle — the conjecture.
"""

import numpy as np


def I2_closed(d, a, b=None):
    """I₂(d; a, b) = (2π²/d)·arctan(d/(a+b)). b defaults to a."""
    if b is None:
        b = a
    return (2 * np.pi**2 / d) * np.arctan(d / (a + b))


def J_closed(d, a):
    """J(d, a) = π²/(a·(d² + 4a²)). Exact."""
    return np.pi**2 / (a * (d**2 + 4 * a**2))


def I3_star_triangle(pos1, pos2, pos3):
    """I₃ at a = 0: π³/(d₁₂·d₁₃·d₂₃), from the star-triangle relation."""
    p1, p2, p3 = (np.asarray(p, dtype=float) for p in (pos1, pos2, pos3))
    d12 = np.linalg.norm(p1 - p2)
    d13 = np.linalg.norm(p1 - p3)
    d23 = np.linalg.norm(p2 - p3)
    return np.pi**3 / (d12 * d13 * d23)


# ---------------------------------------------------------------------------
# Feynman-parametric representations (exact, fast 2D/1D quadrature).
#
# 1/(A₁A₂A₃) = 2·∫_simplex dλ / (Σλᵢ Aᵢ)³ and Σλᵢ Aᵢ = |r−X|² + Δ with
# Δ = λ₁λ₂d₁₂² + λ₁λ₃d₁₃² + λ₂λ₃d₂₃² + a²; ∫d³r/(r²+Δ)³ = (π²/8)·Δ^{−3/2}
# gives
#   I₃(a) = (π²/2)·∫_simplex dλ₁dλ₂ [λ₁λ₂d₁₂² + λ₁λ₃d₁₃² + λ₂λ₃d₂₃² + a²]^{−3/2}
#
# Same with 1/(A²B) = ∫₀¹ 2λ dλ/(λA+(1−λ)B)³:
#   J(d,a) = (π²/2)·∫₀¹ λ dλ [λ(1−λ)d² + a²]^{−3/2}  = π²/(a(d²+4a²))  (exact)


def I3_feynman(d12, d13, d23, a=0.0, epsrel=1e-11):
    """I₃ via 2D Feynman-parameter quadrature. Exact regulator dependence."""
    from scipy.integrate import dblquad

    def integrand(y, x):
        z = 1.0 - x - y
        delta = x * y * d12**2 + x * z * d13**2 + y * z * d23**2 + a**2
        return delta ** (-1.5)

    val, err = dblquad(integrand, 0, 1, 0, lambda x: 1 - x,
                       epsabs=0, epsrel=epsrel)
    return (np.pi**2 / 2) * val
