# Sign audit: this repo's convention vs. the paper

**Paper:** Carney, Karydas, Scharnhorst, Singh, Taylor, PRX 15, 031038 (2025), arXiv:2502.17575.

## First-principles derivation (our convention)

The free energy per site of a qubit with frequency ω at temperature T and
chemical potential μ is

```
g(ω) = −T·ln(1 + e^{(μ−ω)/T})
```

Expanding in ω (verified symbolically with sympy in `tests/test_expansion.py`,
exact tolerance):

```
g(ω) = g(0) + σ*·ω − [σ*(1−σ*)/(2T)]·ω² + [σ*(1−σ*)(1−2σ*)/(6T²)]·ω³ + O(ω⁴)
```

with σ* = 1/(e^{−μ/T} + 1). The coefficients follow from σ'(ω) = −σ(1−σ)/T with
σ(ω) = 1/(1 + e^{(ω−μ)/T}); there is no sign freedom.

## Consequence: the quadratic coefficient is NEGATIVE

With ω_α = Σ_i ℓ_i/(|r_α−x_i|²+a²) and the continuum limit Σ_α → ∫d³r/a³, the
cross term 2ℓ_iℓ_j of the square gives

```
V₂(d) = −[σ*(1−σ*)/T]·(ℓ_iℓ_j/a³)·I₂(d),   I₂(d) = π³·s(d)/d ≥ 0
```

V₂ < 0 and decreasing in magnitude with d, so the force F = −∂V₂/∂d is
**attractive**. Verified numerically in
`tests/test_regression.py::test_sign_audit_force_attractive`
(centered derivative, F < 0). With the identification
G_N = σ*(1−σ*)·π³·L⁴/(T·a³) (Eq. 33 of the paper) one recovers exactly
V₂ = −G_N·m_i·m_j·s(d)/d, that is, attractive Newton with the short-range
correction s(d) = 1 − (2/π)·arctan(2a/d)
(test `test_V2_matches_newton_scaling`, closed-form agreement to 1e-12).

## The apparent Eq. (27) vs Eq. (30) discrepancy in the paper

- Eq. (27) of the paper writes the free-energy expansion with a **negative**
  quadratic coefficient, consistent with our first-principles derivation and
  with the paper's own phrase "controlled by the negative sign in front of
  the Σω² term".
- Eq. (30) writes the term with a sign that, taken literally, would give a
  **positive** contribution ∝ +ω² (a repulsive force once G_N > 0 is
  identified), in tension with the G_N identification in Eq. (33).

Our independent derivation (above) confirms that the correct sign is that of
Eq. (27): **negative quadratic coefficient in the free energy**. Eq. (30)
appears to carry a sign typo (or an unstated intermediate convention). This
entire repo uses the convention derived from first principles:

| term | coefficient | sign |
|---|---|---|
| ω¹ | σ* | + (constant, generates no force) |
| ω² | −σ*(1−σ*)/(2T) | − gives attractive Newton |
| ω³ | +σ*(1−σ*)(1−2σ*)/(6T²) | + for σ* < 1/2, so V₃ > 0 (repulsive) at the entropic point |

Note: at σ* = 1/2 (μ = 0) the cubic term vanishes identically; the paper's
"purely entropic point" (μ = −2.39936·T, σ* = 0.0832215) has
(1−2σ*) = 0.8335570 > 0.
