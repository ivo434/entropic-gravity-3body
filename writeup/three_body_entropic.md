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

(This is the D=3 case of the star-triangle/uniqueness relation, Σδᵢ = D;
the derivation above is self-contained.) For the equilateral triangle
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

### 2.2 The anomalous two-body term and its UV sensitivity

J has the exact closed form (sympy-verified via J = −∂I₂(d;a,b)/∂a²|_{b=a}):

**J(d,a) = π²/[a·(d²+4a²)]** → π²/(a·d²) for d ≫ a.

Two consequences. First, the correction scales as mᵢmⱼ(mᵢ+mⱼ)·L² — the
anomalous m²m′ mass dependence — with a 1/d² tail:
δV₂/V₂ = −[(1−2σ*)/2πT]·(ℓᵢ+ℓⱼ)/(a·d). Second, and importantly, **J is
linearly divergent as a → 0**: it is dominated by lattice sites within ~a of
the mass, exactly where ω/T ≫ 1 for a point mass and the ω-expansion is
invalid. The ω³ truncation is therefore UV-sensitive in its two-body sector
(the ω² Newton sector is not: s(d) → 1). For physical, extended sources the
divergence is cut off by the body itself: writing the cubic term for a mass
density and using the closed form of I₃,

δV₂^(ij) = W_iij + W_jji,  **W_iij = ½(1−2σ*)·(G_N·mᵢ²mⱼ·L²/T)·⟨1/r₁₂⟩ᵢ/d²,**

where ⟨1/r₁₂⟩ᵢ is the mean inverse separation of mass elements within body i
(= 6/5Rᵢ for a uniform sphere; the point-mass-with-cutoff value is 1/πa).
The term probes the internal structure of the source — a violation of the
effacement of internal structure that Newtonian gravity enjoys.

## 3. Sun-Earth-Moon and the LLR constraint

At solar-system separations a/d ~ 1e-22, so all a→0 closed forms are exact.
We compute the perturbation of the Moon's acceleration relative to Earth from
(i) V₃(S,E,M) and (ii) all six W_iij pair terms, as a function of the lunar
phase φ, using uniform-sphere ⟨1/r₁₂⟩ (conservative: the real, centrally
condensed Sun has larger ⟨1/r₁₂⟩, strengthening the bound). At the benchmark
(entropic σ*, T = 300 K, a = 1e-13 m — the paper's anomalous-heating floor):

| component | |δa|/a_N at φ=90° |
|---|---|
| Sun²-Moon pair (W_SSM) | 1.08e-4 |
| Sun²-Earth pair (W_SSE) | 1.08e-4 |
| Earth-Moon pairs | 6.7e-6 |
| pure three-body V₃ | 1.9e-7 |

The ℓ_Sun² leg dominates, as expected, but the two solar terms nearly cancel
in the relative acceleration; the observable residual is tidal-like:
ε(φ) has mean 6.6e-6 and a **fortnightly (cos 2φ) modulation of amplitude
5.6e-7** (monthly harmonic 3.2e-8). The point-mass (X = 1/πa) evaluation gives
ε ~ 1e15 — not a prediction but a demonstration that the truncated expansion
applied to point sources is meaningless in the near zone; extended bodies are
the physically defensible treatment.

Every ω³ observable scales as ε ∝ f(σ*)·√(a³/T), f(σ) = (1−2σ)/√(σ(1−σ))
(via L⁴ = G_N·T·a³/(σ*(1−σ*)π³); verified numerically to 3e-4). Imposing the
LLR fractional sensitivity δ on the modulation amplitude:

**T_min(a) = 9.8e9 K · (a/1e-13 m)³ · [f(σ*)/3.018]² · (1e-10/δ)²**

(δ = 1e-10 conservative; δ = 1e-11 gives 9.8e11 K). At the anomalous-heating
floor a = 1e-13 m, a room-temperature mediator at the purely entropic point is
excluded by ~7 orders of magnitude in ε; consistency requires T ≳ 1e10 K
(≈ 1 MeV). The exclusion is self-consistent: on the boundary ω/T ≪ 1
everywhere, including inside the Sun (ω_center/T ∝ √(a³/T) is constant along
T ∝ a³ lines and ≈ 3e-6 on the δ=1e-10 boundary). See
`figures/llr_constraint.png` (excluded region in the (T,a) plane, entropic
point marked, expansion-invalid region shaded separately) and
`figures/eps_phase.png`.

## 4. Honest limitations

1. **Non-relativistic, static sources**; no claim about relativistic or
   cosmological regimes.
2. **Adiabaticity assumed, not derived**: the free-energy treatment requires
   the qubit bath to thermalize fast compared to orbital motion
   (γ_th ≫ 2.7e-6 Hz). We do not model the thermalization dynamics.
3. **The ω³ term is itself truncated**: ω⁴ and higher terms are more
   UV-sensitive still (∫d³ρ/ρ⁶ etc.); our extended-body treatment controls
   ω³, and on the exclusion boundary ω/T ≪ 1 keeps the hierarchy, but a
   resummation near the sources would be needed for T far below the bound.
4. **Uniform-sphere structure factors**: ⟨1/r₁₂⟩ = 6/5R is a conservative
   simplification; realistic density profiles change O(1) factors only.
5. LLR sensitivity is applied as a fractional-acceleration criterion on the
   modulating component (δ ≈ 5e-11–1e-10 from ~2 cm residuals); a full
   orbit-fit analysis (partials against the standard LLR solution parameters)
   would sharpen but not qualitatively change the bound.

## Reproduction

All numbers above: `pytest tests/` (23 tests: symbolic expansion, integrator
regression, closed forms, sign audit) and `notebooks/task{1,2,3}_*.py`.
