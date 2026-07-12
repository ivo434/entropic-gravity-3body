# A three-body force and an LLR constraint from the ω³ term of the local entropic-gravity model

**Scope.** We work entirely within the "local model" of Carney, Karydas,
Scharnhorst, Singh & Taylor, *On the quantum mechanics of entropic forces*,
PRX **15**, 031038 (2025), arXiv:2502.17575 (see also the follow-up
arXiv:2603.26075). The paper obtains Newtonian gravity from the ω² term of the
thermal free energy of a qubit lattice and truncates there. We compute the
first neglected term (ω³) and show it produces two observable signatures: a
long-range **three-body force** with an exact closed form, and an **anomalous
two-body correction** scaling as m²m′. Lunar laser ranging (LLR) then puts a
lower bound on the mediator temperature T as a function of the lattice
spacing a.

## 1. Setup and conventions

A cubic lattice (spacing a) of qubits at temperature T and chemical potential
μ; a mass mᵢ at xᵢ shifts the qubit frequency at site r by
ωᵢ(r) = ℓᵢ/(|r−xᵢ|²+a²), ℓᵢ = mᵢL². The free energy per site
g(ω) = −T·ln(1+e^{(μ−ω)/T}) expands (sympy-verified, `tests/test_expansion.py`) as

g(ω) = g(0) + σ*ω − [σ*(1−σ*)/2T]·ω² + [σ*(1−σ*)(1−2σ*)/6T²]·ω³ + O(ω⁴),

σ* = 1/(e^{−μ/T}+1). In the continuum limit Σ_sites → ∫d³r/a³ the ω² cross
term reproduces Newton: V₂(d) = −G_N·mᵢmⱼ·s(d)/d with
s(d) = 1−(2/π)arctan(2a/d) and the identification (paper Eq. 33)
**G_N = σ*(1−σ*)π³L⁴/(Ta³)**. Our sign conventions are derived from first
principles and audited numerically (attractive force; `writeup/notes_signos.md`
documents an apparent sign inconsistency between the paper's Eqs. 27 and 30 —
Eq. 30's sign is the correct one). Throughout, the "purely entropic point" is
μ = −2.39936·T, σ* = 0.0832215, (1−2σ*) = 0.8335570. At μ = 0 (σ* = ½) the
cubic term vanishes identically.

## 2. The ω³ term

Expanding ω³ = (Σᵢωᵢ)³, the fully-mixed piece gives a **pure three-body
potential** and the ℓᵢ²ℓⱼ pieces an **anomalous two-body correction**:

- V₃ = [σ*(1−σ*)(1−2σ*)/T²]·(ℓ₁ℓ₂ℓ₃/a³)·I₃,  I₃ = ∫d³r Πᵢ 1/(|r−xᵢ|²+a²)
- δV₂ = [σ*(1−σ*)(1−2σ*)/2T²]·(ℓᵢ²ℓⱼ+ℓᵢℓⱼ²)/a³·J(d),
  J = ∫d³r (|r−xᵢ|²+a²)^{−2}(|r−xⱼ|²+a²)^{−1}

### 2.1 Closed form for I₃ (exact)

Fourier methods give the generalized two-center integral
I₂(d;a,b) = (2π²/d)·arctan(d/(a+b)). For I₃ at a = 0, conformal inversion
centered on x₃ (r = x₃+R/R²) maps the three-denominator integral exactly onto
I₂, using |x−y|² → |X−Y|²/(X²Y²) and d³r = d³R/R⁶. The result:

**I₃(x₁,x₂,x₃; a=0) = π³/(d₁₂·d₁₃·d₂₃).**

This is the D = 3, δ₁ = δ₂ = δ₃ = 1 "uniqueness" case (Σδᵢ = D) of the
star-triangle relation of conformal field theory [Symanzik (1972), *On
calculations in conformal invariant field theories*; D'Eramo, Parisi &
Peliti (1971)]; the inversion derivation above is self-contained. For the
equilateral triangle
I₃·d³ = π³ = 31.00628 exactly, settling the conjecture from the preliminary
exploration. Verification: an exact Feynman-parametric quadrature
(I₃ = (π²/2)∫_simplex dλ [λ₁λ₂d₁₂²+λ₁λ₃d₁₃²+λ₂λ₃d₂₃²+a²]^{−3/2}) agrees with
the closed form to 1e-13 at four scalene geometries, with the 3D lattice-grid
integrator at the 1e-5 level, and the a→0 limit approaches π³ smoothly with a
leading correction I₃d³ ≈ π³(1−c·a/d), c ≈ 1.911 ≈ 6/π (numerical; 2/π per
regulated denominator, matching s(d)'s 4/π for two).

Thus the entropic model generates a three-body potential

**V₃ = (1−2σ*)·(G_N·m₁m₂m₃·L²/T)/(d₁₂·d₁₃·d₂₃),**

repulsive for σ* < ½, scaling as 1/d³ at overall separation d. Contrast:
in linearized quantum gravity the leading quantum three-body correction found
by Hu & Yu (arXiv:2201.06200) falls as r_A⁻⁵r_B⁻⁵r_C⁻⁵ — parametrically far
steeper. A 1/(d₁₂d₁₃d₂₃) three-body force of gravitational-ish strength is a
distinctive, genuinely long-range signature of the entropic mechanism.
For equal masses on an equilateral triangle the exact net-force ratio is

|F₃|/|F_Newton| = (1−2σ*)·(ω_d/T),  ω_d ≡ mL²/d²,

verified numerically to all digits (`notebooks/task3_llr.py`).

### 2.2 Validity of the cubic truncation: the anomalous two-body term

J has the exact closed form (sympy-verified via J = −∂I₂(d;a,b)/∂a²|_{b=a}):

**J(d,a) = π²/[a·(d²+4a²)]** → π²/(a·d²) for d ≫ a.

**J is linearly divergent as a → 0**: it is dominated by lattice sites within
~a of the mass (a 10a-ball split puts >85% of J inside, versus O(a/d) of I₃ —
verified in `tests/test_addendum.py`), exactly where ω/T ≫ 1 for a point mass
and the ω-expansion is invalid — there g(ω) saturates (Fermi). **The
perturbative m²m′ term is therefore an artifact of the truncation, not a
model prediction.** The three-body term, by contrast, is infrared-dominated
and trustworthy.

The correct treatment is non-perturbative. The Möbius-subtracted interaction
free energies with the full g(ω),

A₂(d) = ∫d³r/a³ [g(ω₁+ω₂)−g(ω₁)−g(ω₂)+g(0)],  (and its 3-mass analogue A₃)

are finite (self-energies cancel in the subtraction; implemented and gated at
1% against the perturbative closed forms in a ω ≪ T toy regime). For widely
separated point masses, linear response in the partner field gives

A₂(d) = −G_N·m₁m₂·s(d)/d + [ℓ₂S(ℓ₁)+ℓ₁S(ℓ₂)]/(a³d²),
**S(ℓ) = 4π∫dr r² [σ(ω(r)) − σ* + σ*(1−σ*)ω(r)/T]**,  ω(r) = ℓ/(r²+a²).

S is finite: sites within the saturation radius **r* = √(ℓ/T)** are
Pauli-blocked (σ(ω)→0) and the effective UV cutoff of the anomalous term is
r*, not a. In the perturbative limit (r* ≪ a) S → 3c₃ℓ²π²/a, reproducing the
J-term; in deep saturation S ∝ r*³. At solar benchmark parameters the
suppression is dramatic: S/S_pert ≈ 5e-21 for the Sun (a_eff ≈ 2e7 m ~ r*
instead of a = 1e-13 m).

The residual anomalous term still scales as mᵢmⱼ(mᵢ+mⱼ)-like with a 1/d²
tail, and — for physical extended sources, which never saturate
(r* < R_body for Sun, Earth and Moon at all relevant T) — takes the
perturbative extended-body form

δV₂^(ij) = W_iij + W_jji,  **W_iij = ½(1−2σ*)·(G_N·mᵢ²mⱼ·L²/T)·⟨1/r₁₂⟩ᵢ/d²,**

with ⟨1/r₁₂⟩ᵢ the mean inverse separation of mass elements in body i
(= 6/5Rᵢ for a uniform sphere). Either way the term probes source structure
(⟨1/r₁₂⟩ or r*) — a violation of the effacement of internal structure that
Newtonian gravity enjoys.

## 3. Sun-Earth-Moon and the LLR constraint

At solar-system separations a/d ~ 1e-22, so all a→0 closed forms are exact
(the closed form for I₃(φ) is cross-checked against an independent
1D-reduced Feynman quadrature to 1e-12 at the real, extremely hierarchical
solar geometry). We compute the perturbation of the Moon's acceleration
relative to Earth from (i) V₃(S,E,M) and (ii) the pair terms, as a function
of the lunar phase φ. All the physics beyond Newton enters through
**Λ ≡ (1−2σ*)·L²/T**; LLR bounds Λ, and T_min follows at fixed (a, σ*).
Modulation amplitudes of ε(φ) = δa_radial/a_N at the benchmark (entropic σ*,
T = 300 K, a = 1e-13 m — the paper's anomalous-heating floor), by treatment
of the anomalous term:

| treatment | ε modulation |
|---|---|
| point masses, perturbative (cutoff a) — truncation artifact | 1.0e15 |
| point masses, non-perturbative Möbius (cutoff r*) | 5.5e-6 |
| extended uniform spheres, perturbative (physical bodies) | 5.7e-7 |
| V₃ only — structure-independent floor | 1.2e-11 |

The ℓ_Sun² leg dominates the anomalous term, but the Sun-Moon and Sun-Earth
pieces nearly cancel in the relative acceleration; the observable residual is
tidal-like, dominated by the **fortnightly (cos 2φ) harmonic** (monthly is
~18× smaller in the extended case). The mean (non-modulating) offset is ~10×
the modulation and is absorbed in orbit fitting.

Since the anomalous term depends on source structure while V₃ does not, we
quote a layered bound, ε ∝ f(σ*)·√(a³/T) with f(σ) = (1−2σ)/√(σ(1−σ))
(scaling verified numerically to 3e-4):

- **V₃ only (assumption-free):** T_min(a₀) = 4.4 K at δ = 1e-10, 436 K at
  δ = 1e-11 — the latter already grazing a room-temperature mediator at
  a₀ = 1e-13 m; equivalently Λ < 8.3e-10 GeV⁻³·(δ/1e-10).
- **Extended bodies (realistic):** **T_min(a) = 9.8e9 K · (a/1e-13 m)³ ·
  [f(σ*)/3.018]² · (1e-10/δ)²** — a room-temperature mediator at the
  entropic point is excluded by ~7 orders of magnitude in ε; consistency
  requires T ≳ 1e10 K (≈ 1 MeV) at the anomalous-heating floor.
- The saturated point-mass evaluation (model-literal, cutoff r* = √(ℓ/T))
  lies a further ~10× above the extended-body one.

The exclusion is self-consistent: on the boundary ω/T ≪ 1 everywhere,
including inside the Sun (ω_center/T ∝ √(a³/T) is constant along T ∝ a³
lines and ≈ 3e-6 on the extended δ=1e-10 boundary). See
`figures/llr_constraint.png` (both boundaries in the (T,a) plane, entropic
point marked, expansion-invalid region shaded separately) and
`figures/eps_phase.png`.

## 4. Honest limitations

1. **Non-relativistic, static sources**; no claim about relativistic or
   cosmological regimes.
2. **Adiabaticity assumed, not derived**: the free-energy treatment requires
   the qubit bath to thermalize fast compared to orbital motion
   (γ_th ≫ 2.7e-6 Hz). We do not model the thermalization dynamics.
3. **Higher orders**: the near-zone breakdown of the cubic term is handled
   non-perturbatively (Möbius subtraction, saturation moment S), but the
   linear-response step (w = ℓ_partner/d² ≪ T) and the ω⁴+ inter-body tail
   are still truncated; on the exclusion boundary ω/T ≪ 1 in the infrared
   region keeps these controlled.
4. **Uniform-sphere structure factors**: ⟨1/r₁₂⟩ = 6/5R is a conservative
   simplification for the extended-body bound (the centrally condensed Sun
   has larger ⟨1/r₁₂⟩, strengthening it); realistic profiles change O(1)
   factors. The V₃-only bound has no structure dependence at all.
5. LLR sensitivity is applied as a fractional-acceleration criterion on the
   modulating component (δ ≈ 5e-11–1e-10 from ~2 cm residuals); a full
   orbit-fit analysis (partials against the standard LLR solution parameters)
   would sharpen but not qualitatively change the bound.

## References

Carney, Karydas, Scharnhorst, Singh & Taylor, PRX 15, 031038 (2025),
arXiv:2502.17575 (and follow-up arXiv:2603.26075). Hu & Yu, arXiv:2201.06200.
Symanzik, "On calculations in conformal invariant field theories" (1972).
D'Eramo, Parisi & Peliti (1971).

## Reproduction

All numbers above: `pytest tests/` (33 tests: symbolic expansion, integrator
regression, closed forms, sign audit, addendum gates incl. the 1% Möbius
consistency gate) and `notebooks/task{1,2,2b,3}_*.py`.
