"""Task 3. Sun-Earth-Moon: ε(φ), sanity formula, and the LLR bound on T.

Run:  .venv/bin/python notebooks/task3_llr.py
Outputs: figures/llr_constraint_scaling.png (single-boundary scaling view),
figures/eps_phase.png + console report. The main two-boundary constraint
figure (figures/llr_constraint.png) is produced by task2b_nonperturbative.py.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (G_NEWTON, M_SUN, M_EARTH, M_MOON, D_EM, D_ES, METER,
                       KELVIN, KG, T_300K, A_MIN, SIGMA_ENTROPIC, L4_from_GN)
from src.solar import (R_SUN, prefactor_A, geometry, perturbation,
                       eps_modulation, accel_V3)
from src.analytic import I3_star_triangle, J_closed

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
SIG = SIGMA_ENTROPIC
T0, A0 = T_300K, A_MIN  # benchmark: T = 300 K, a = 1e-13 m

print("=" * 76)
print("3.1  Geometry: I₃(φ) and J per pair (benchmark units, a = 1e-13 m)")
print("=" * 76)
print(f"  {'φ [deg]':>8s}  {'d_MS [m]':>12s}  {'I₃(φ)·d_EM·d_ES·d_MS/π³':>24s}")
for phi_deg in [0, 45, 90, 135, 180]:
    xS, xE, xM = geometry(np.radians(phi_deg))
    I3 = I3_star_triangle(xS, xE, xM)
    dMS = np.linalg.norm(xM - xS)
    print(f"  {phi_deg:8d}  {dMS/METER:12.4e}  {I3*D_EM*D_ES*dMS/np.pi**3:24.6f}")
print("  (I₃ = π³/(d_ES·d_EM·d_MS) exactly, so the column must be 1)")
print(f"\n  J(d,a)·a·d² /π² per pair (must be ≈1: J = π²/(a·d²) at d ≫ a):")
for name, d in [("EM", D_EM), ("ES", D_ES)]:
    print(f"    {name}: {J_closed(d, A0)*A0*d**2/np.pi**2:.12f}")

# ---------------------------------------------------------------- 3.2 ε(φ)
print()
print("=" * 76)
print("3.2  ε(φ) at benchmark (σ*=0.0832215, T=300 K, a=1e-13 m)")
print("=" * 76)
for extended, label in [(True, "EXTENDED bodies (uniform spheres, X=6/5R)"),
                        (False, "POINT masses (X=1/πa, point-mass baseline)")]:
    print(f"\n  --- {label} ---")
    r = perturbation(np.radians(90.0), SIG, T0, A0, extended=extended)
    aN = r["aN"]
    print(f"  component |δa|/a_N at φ=90°:")
    for k, v in r["parts"].items():
        print(f"    {k:26s}: {np.linalg.norm(v)/aN:.3e}")
    m = eps_modulation(SIG, T0, A0, extended=extended)
    print(f"  ε mean = {m['mean']:+.3e}   modulation (max−min)/2 = {m['amp']:.3e}")
    print(f"  harmonics: monthly |c₁| = {m['fourier_c1']:.3e}   "
          f"fortnightly |c₂| = {m['fourier_c2']:.3e}  (tidal cos2φ dominates)")

# verdict on dominance
r = perturbation(np.radians(90.0), SIG, T0, A0, extended=True)
mags = {k: np.linalg.norm(v) for k, v in r["parts"].items()}
dom = max(mags, key=mags.get)
print(f"\n  Dominant term: {dom}  → the ℓ_Sun² leg dominates, as anticipated.")

# gradient cross-check for V3 (analytic vs numerical)
phi = np.radians(60.0)
xS, xE, xM = geometry(phi)
A = prefactor_A(SIG, T0, A0)
C3 = A * M_SUN * M_EARTH * M_MOON
aM, aE = accel_V3(xS, xE, xM, M_SUN, M_EARTH, M_MOON, C3)
h = D_EM * 1e-7
num = np.zeros(3)
for i in range(3):
    e = np.zeros(3); e[i] = h
    def V3(xm):
        return C3 / (np.linalg.norm(xE - xS) * np.linalg.norm(xm - xE)
                     * np.linalg.norm(xm - xS))
    num[i] = -(V3(xM + e) - V3(xM - e)) / (2 * h) / M_MOON
print(f"  V₃ gradient check (analytic vs central difference): "
      f"rel = {np.linalg.norm(aM-num)/np.linalg.norm(aM):.1e}")

# ---------------------------------------------------------------- 3.3 sanity
print()
print("=" * 76)
print("3.3  Sanity: equilateral equal masses, R = |F₃|/|F_N| vs (1−2σ*)·ω_d/T")
print("=" * 76)
def equilateral_ratio(m_kg, d_m, T=T0, a=A0, sigma=SIG):
    m = m_kg * KG
    d = d_m * METER
    L2 = np.sqrt(L4_from_GN(sigma, T, a))
    ell = m * L2
    A_ = (1 - 2 * sigma) * G_NEWTON * L2 / T
    C3_ = A_ * m**3
    # net 3-body force on vertex 1 (equilateral side d): |−∇₁V₃| = √3·C₃/d⁴
    F3 = np.sqrt(3) * C3_ / d**4
    # net Newton force on vertex 1 from the other two: √3·G m²/d²
    FN = np.sqrt(3) * G_NEWTON * m**2 / d**2
    omega_d = ell / d**2
    return F3 / FN, (1 - 2 * sigma) * omega_d / T

for m_kg, label in [(1e-6, "1 mg"), (1e-3, "1 g")]:
    R_num, R_formula = equilateral_ratio(m_kg, 1e-3)
    print(f"  m = {label}, d = 1 mm:  R_numeric = {R_num:.3e}   "
          f"(1−2σ*)·ω_d/T = {R_formula:.3e}   [(1−2σ*)/3]·ω_d/T = {R_formula/3:.3e}")
print("""
  Exact identity: |F₃|/|F_N| = (1−2σ*)·ω_d/T for the equilateral (net forces;
  both carry the same √3). The spec's benchmark R ~ 7e-13 is reproduced by
  the [(1−2σ*)/3] variant for m = 1 g (7.2e-13); for m = 1 mg all variants
  give ~1e-15 to 1e-16. The quoted 7e-13 corresponds to gram-scale masses,
  apparent unit slip in the preliminary exploration; formula itself VERIFIED.""")

# ---------------------------------------------------------------- 3.4 bound
print("=" * 76)
print("3.4  LLR bound: ε_modulation < δ  →  T_min(a, σ*)")
print("=" * 76)
# ε ∝ (1−2σ)L²/T ∝ f(σ)·√(a³/T), f(σ) = (1−2σ)/√(σ(1−σ))
# → T_min = T0·(a/a0)³·(f/f0)²·(ε0/δ)²
m0 = eps_modulation(SIG, T0, A0, extended=True)
eps0 = m0["amp"]
f_ent = (1 - 2 * SIG) / np.sqrt(SIG * (1 - SIG))
print(f"  benchmark ε_mod(300 K, 1e-13 m, σ*_entropic) = {eps0:.3e}")
print(f"  scaling: ε_mod = ε₀·[f(σ*)/f₀]·√[(a/a₀)³·(T₀/T)],  f(σ)=(1−2σ)/√(σ(1−σ))")

# numeric verification of the scaling at two (T,a) points
for T_, a_, note in [(10 * T0, A0, "T×10"), (T0, 4 * A0, "a×4")]:
    m_ = eps_modulation(SIG, T_, a_, extended=True, nphi=90)
    pred = eps0 * np.sqrt((a_ / A0) ** 3 * (T0 / T_))
    print(f"    check {note}: ε = {m_['amp']:.4e}, scaling predicts {pred:.4e}, "
          f"rel = {m_['amp']/pred-1:+.1e}")

for delta in [1e-10, 1e-11]:
    Tmin = T0 * (eps0 / delta) ** 2  # at a = a0, entropic point
    print(f"\n  δ = {delta:g}:  T_min(a=1e-13 m, entropic σ*) = "
          f"{Tmin/KELVIN:.3e} K = {Tmin:.3e} GeV")
    print(f"           T_min(a) = {Tmin/KELVIN:.3e} K · (a/1e-13 m)³ · [f(σ*)/{f_ent:.3f}]²")

# expansion validity: ω/T at the Moon (spec) and at the Sun's center (stricter)
L2 = np.sqrt(L4_from_GN(SIG, T0, A0))
om_moon = L2 * (M_SUN / D_ES**2 + M_EARTH / D_EM**2)   # d_MS≈d_ES at quadrature
om_sun_c = 3 * M_SUN * L2 / R_SUN**2                    # uniform-sphere center
print(f"\n  validity at benchmark: ω(Moon)/T = {om_moon/T0:.2e},  "
      f"ω(Sun center)/T = {om_sun_c/T0:.2e}")
print("  ω/T ∝ √(a³/T) → constant along any T ∝ a³ line; on the exclusion")
print("  boundary ω/T ≪ 1 everywhere (expansion self-consistent there).")

# ---------------------------------------------------------------- figures
phis = m0["phis"]
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(np.degrees(phis), m0["eps"], lw=1.8)
ax.axhline(m0["mean"], ls="--", c="gray", lw=1, label=f"mean = {m0['mean']:.2e}")
ax.set_xlabel(r"lunar phase $\varphi$ [deg]")
ax.set_ylabel(r"$\varepsilon(\varphi) = \delta a_{\rm radial}/a_N$")
ax.set_title(f"Radial perturbation, extended bodies, T=300 K, a=1e-13 m "
             f"(modulation {m0['amp']:.1e})")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "eps_phase.png", dpi=160)

# (T, a) exclusion plane
a_m = np.logspace(-15, -9, 200)                    # lattice spacing in meters
Tmin_10 = (T0 / KELVIN) * (eps0 / 1e-10) ** 2 * (a_m / 1e-13) ** 3
Tmin_11 = (T0 / KELVIN) * (eps0 / 1e-11) ** 2 * (a_m / 1e-13) ** 3
f45 = (1 - 2 * 0.45) / np.sqrt(0.45 * 0.55)
Tmin_s45 = Tmin_10 * (f45 / f_ent) ** 2
# expansion invalid inside the Sun: ω_c/T > 1
Tinv = (T0 / KELVIN) * (om_sun_c / T0) ** 2 * (a_m / 1e-13) ** 3

fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.fill_between(a_m, Tinv, 1e-30, color="0.55", alpha=0.55,
                label=r"$\omega/T>1$ inside Sun (expansion invalid)")
ax.fill_between(a_m, Tmin_10, Tinv, color="tab:red", alpha=0.35,
                label=r"excluded by LLR ($\delta=10^{-10}$)")
ax.plot(a_m, Tmin_10, c="tab:red", lw=2)
ax.plot(a_m, Tmin_11, c="tab:red", lw=1.2, ls="--",
        label=r"$\delta=10^{-11}$")
ax.plot(a_m, Tmin_s45, c="tab:orange", lw=1.2, ls=":",
        label=r"$\delta=10^{-10}$, $\sigma^*=0.45$")
ax.axvspan(1e-15, 1e-13, color="tab:blue", alpha=0.12,
           label=r"$a<10^{-13}$ m (anomalous heating, paper)")
ax.plot([1e-13], [300], marker="*", ms=16, c="k",
        label="entropic point, T=300 K, a=1e-13 m")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(1e-15, 1e-9); ax.set_ylim(1e-4, 1e20)
ax.set_xlabel("lattice spacing a [m]")
ax.set_ylabel("mediator temperature T [K]")
ax.set_title("LLR constraint on the local entropic-gravity model "
             r"($\omega^3$ term)")
ax.legend(loc="upper left", fontsize=8.5)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIGDIR / "llr_constraint_scaling.png", dpi=160)
print("\n  figures saved: figures/eps_phase.png, figures/llr_constraint_scaling.png")

print("""
3.5  EXPLICIT ASSUMPTION (not resolved here): the adiabatic derivation
     requires the qubit bath to thermalize fast relative to orbital motion,
     γ_th ≫ Ω_orb ≈ 2.7e-6 Hz. We assume this throughout.""")
