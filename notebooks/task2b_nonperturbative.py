"""Addendum 1 — non-perturbative anomalous correction and the 2.3 decision.

Run:  .venv/bin/python notebooks/task2b_nonperturbative.py
Outputs: figures/llr_constraint.png (updated) + console decision report.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import (M_SUN, M_EARTH, M_MOON, D_EM, D_ES, METER, KELVIN,
                       T_300K, A_MIN, SIGMA_ENTROPIC, L4_from_GN, G_NEWTON)
from src.solar import (R_SUN, R_EARTH, R_MOON, eps_modulation, geometry)
from src.nonperturbative import S_moment, S_perturbative
from src.analytic import I3_feynman_1d, I3_star_triangle

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
SIG, T0, A0 = SIGMA_ENTROPIC, T_300K, A_MIN
MU0 = T0 * np.log(SIG / (1 - SIG))
L2 = np.sqrt(L4_from_GN(SIG, T0, A0))

# ------------------------------------------------- A: solar I₃(φ) cross-check
print("=" * 76)
print("A.  I₃(φ) closed form vs Feynman quadrature, real solar geometry")
print("=" * 76)
for phi_deg in [0, 90, 180]:
    xS, xE, xM = geometry(np.radians(phi_deg))
    scale = D_ES
    d12 = np.linalg.norm(xS - xE) / scale
    d13 = np.linalg.norm(xS - xM) / scale
    d23 = np.linalg.norm(xE - xM) / scale
    num = I3_feynman_1d(d12, d13, d23) / scale**3
    closed = I3_star_triangle(xS, xE, xM)
    print(f"  φ={phi_deg:3d}°:  quadrature/closed = {num/closed:.12f}")

# ------------------------------------------------- B: saturation moments
print()
print("=" * 76)
print("B.  Saturation radii and moments (point masses, benchmark params)")
print("=" * 76)
print(f"  {'body':>6s} {'r* = √(ℓ/T) [m]':>18s} {'R_body [m]':>12s} "
      f"{'r*/R':>8s} {'S/S_pert':>12s} {'a_eff = π²/... [m]':>18s}")
for name, m, R in [("Sun", M_SUN, R_SUN), ("Earth", M_EARTH, R_EARTH),
                   ("Moon", M_MOON, R_MOON)]:
    ell = m * L2
    rstar = np.sqrt(ell / T0)
    S = S_moment(ell, T0, MU0, A0)
    Sp = S_perturbative(ell, T0, MU0, A0)
    # effective cutoff: S ≡ 3c₃ℓ²π²/a_eff → a_eff = a·S_pert/S
    a_eff = A0 * Sp / S
    print(f"  {name:>6s} {rstar/METER:18.4e} {R/METER:12.4e} "
          f"{rstar/R:8.2e} {S/Sp:12.4e} {a_eff/METER:18.4e}")
print("""
  → Saturation suppresses the point-mass anomalous term by a_eff ≈ a·S_pert/S
    ~ r*-scale instead of a. Note r* < R_body for all three bodies: REAL
    (extended) bodies never reach saturation — their perturbative treatment
    with X = 6/(5R) stands. The saturated numbers below are the model-literal
    answer for true point masses.""")

# ------------------------------------------------- C: decision table 2.3
print("=" * 76)
print("C.  Decision table: ε modulation at benchmark by treatment")
print("=" * 76)
rows = []
for mode, only, label in [
    ("point", None, "point masses, perturbative (cutoff a)  [NOT a prediction]"),
    ("saturated", None, "point masses, NON-perturbative (Möbius)"),
    ("extended", None, "extended uniform spheres, perturbative"),
    ("extended", ["V3"], "V₃ only (structure-independent floor)"),
]:
    m = eps_modulation(SIG, T0, A0, mode=mode, only=only, nphi=180)
    rows.append((label, m))
    print(f"  {label:58s} amp = {m['amp']:.3e}")

eps_sat = rows[1][1]["amp"]
eps_ext = rows[2][1]["amp"]
eps_v3 = rows[3][1]["amp"]
print(f"""
  DECISION (documented): the saturated point-mass anomalous term
  ({eps_sat:.1e}) is NOT ≪ V₃ ({eps_v3:.1e}) — it exceeds it — so the
  anomalous term cannot be dropped in general. BUT its size depends on the
  source model (r* vs R_body vs a). We therefore quote a LAYERED bound:
    (i)  headline / bulletproof: V₃ ONLY — no dependence on source structure
         or on the near-zone treatment; pure closed-form physics.
    (ii) realistic: extended bodies (physical Sun/Earth/Moon, which never
         saturate since r* < R) — anomalous term included, ~3x stronger.
  The point-mass numbers are reported as model-internal only.""")

# ------------------------------------------------- D: layered bounds + figure
print("=" * 76)
print("D.  Bounds: Λ ≡ (1−2σ*)L²/T and T_min(a)")
print("=" * 76)
Lam0 = (1 - 2 * SIG) * L2 / T0
f_ent = (1 - 2 * SIG) / np.sqrt(SIG * (1 - SIG))
for eps0, tag in [(eps_v3, "V₃-only"), (eps_ext, "extended")]:
    for delta in [1e-10, 1e-11]:
        # ε ∝ Λ (all terms linear in Λ at fixed geometry/structure)
        Lam_max = Lam0 * delta / eps0
        Tmin = T0 * (eps0 / delta) ** 2
        print(f"  {tag:9s} δ={delta:g}:  Λ_max = {Lam_max:.3e} GeV⁻³ "
              f"(benchmark Λ = {Lam0:.3e});  T_min(a₀) = {Tmin/KELVIN:.3e} K")
    print(f"           T_min(a) = T_min(a₀)·(a/1e-13 m)³·[f(σ*)/{f_ent:.3f}]²")

# updated exclusion figure: two boundaries (V₃-only solid, extended dashed)
a_m = np.logspace(-15, -9, 200)
om_sun_c = 3 * M_SUN * L2 / R_SUN**2
Tv3 = (T0 / KELVIN) * (eps_v3 / 1e-10) ** 2 * (a_m / 1e-13) ** 3
Text = (T0 / KELVIN) * (eps_ext / 1e-10) ** 2 * (a_m / 1e-13) ** 3
Tinv = (T0 / KELVIN) * (om_sun_c / T0) ** 2 * (a_m / 1e-13) ** 3

fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.fill_between(a_m, Tinv, 1e-30, color="0.55", alpha=0.55,
                label=r"$\omega/T>1$ inside Sun (expansion invalid)")
ax.fill_between(a_m, Tv3, Tinv, color="tab:red", alpha=0.35,
                label=r"excluded by LLR, $V_3$ only ($\delta=10^{-10}$)")
ax.plot(a_m, Tv3, c="tab:red", lw=2)
ax.plot(a_m, Text, c="tab:red", lw=1.4, ls="--",
        label=r"+ anomalous term, extended bodies")
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
fig.savefig(FIGDIR / "llr_constraint.png", dpi=160)
print("\n  figure updated: figures/llr_constraint.png")
