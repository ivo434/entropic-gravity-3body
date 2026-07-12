"""Task 2 — anomalous two-body correction: J(d), power law, δV₂/V₂.

Run:  .venv/bin/python notebooks/task2_j.py
Outputs: figures/j_powerlaw.png + console tables.

KEY DEVIATION FROM THE SPEC (verified, exact):
  J does NOT converge without regulator. Closed form (sympy-verified):
      J(d, a) = π²/(a·(d² + 4a²))
  → linearly divergent as a→0 (the ∫d³ρ/ρ⁴ singularity at the squared
  center). The lattice spacing a is the physical regulator.
  Consequence: δV₂/V₂ is ENHANCED by d/a over the naive ω_d/T estimate.
"""

import sys
from pathlib import Path

import numpy as np
import mpmath as mp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrators import integrate3d
from src.analytic import J_closed, I2_closed

FIGDIR = Path(__file__).resolve().parents[1] / "figures"
FIGDIR.mkdir(exist_ok=True)

A = 1.0  # work in units of the lattice spacing; J depends only on d/a (×1/a³)

# --------------------------------------------------- 2.1 numeric J over d/a
print("=" * 74)
print("2.1  J(d) for d/a ∈ [10, 1e4] — mpmath Feynman quadrature (30 digits)")
print("     vs closed form π²/(a(d²+4a²));  3D-grid spot checks at d/a ≤ 100")
print("=" * 74)
mp.mp.dps = 30

def J_feynman_mp(d, a):
    return mp.pi**2 / 2 * mp.quad(
        lambda lam: lam / (lam * (1 - lam) * d**2 + a**2) ** mp.mpf(1.5), [0, 1]
    )

ds = np.logspace(1, 4, 13)
Js = []
print(f"  {'d/a':>10s}  {'J (Feynman)':>16s}  {'J closed':>16s}  {'rel':>10s}")
for d in ds:
    jf = float(J_feynman_mp(mp.mpf(d), 1))
    jc = J_closed(d, A)
    Js.append(jf)
    print(f"  {d:10.4g}  {jf:16.8e}  {jc:16.8e}  {jf/jc-1:+10.1e}")

# 3D-grid spot checks (grid centered on the squared center to resolve the peak)
for d in [10.0, 100.0]:
    f = lambda X, Y, Z: 1 / (X**2 + Y**2 + Z**2 + A**2) ** 2 / (
        (X - d) ** 2 + Y**2 + Z**2 + A**2
    )
    num = integrate3d(f, [0, 0, 0], 2 * A, nr=600, nth=128, nph=128)
    print(f"  3D grid, d/a={d:g}: {num:.6e}  vs closed {J_closed(d, A):.6e}  "
          f"rel={num/J_closed(d, A)-1:+.1e}")

# --------------------------------------------------- 2.2 power law
logd, logJ = np.log(ds), np.log(Js)
slope, intercept = np.polyfit(logd, logJ, 1)
print(f"\n2.2  power-law fit over the full decade range: J ~ d^{slope:.4f}")
print(f"     amplitude e^b = {np.exp(intercept):.6f}  (π²/a = {np.pi**2:.6f} in a=1 units)")
print("     → J = π²/(a·d²) at d ≫ a; exact closed form J = π²/(a(d²+4a²))")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(ds, Js, "o", label="numeric (Feynman quadrature)")
dd = np.logspace(1, 4, 200)
ax.loglog(dd, J_closed(dd, A), "-", label=r"$\pi^2/(a(d^2+4a^2))$ (closed form)")
ax.set_xlabel("d / a")
ax.set_ylabel(r"$J \cdot a^3$  (units of $a$)")
ax.set_title(r"Anomalous two-body integral: $J(d,a)=\pi^2/(a(d^2+4a^2))$")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIGDIR / "j_powerlaw.png", dpi=160)
print("     figure saved: figures/j_powerlaw.png")

# --------------------------------------------------- 2.3 δV₂/V₂
print()
print("=" * 74)
print("2.3  δV₂/V₂ — analytic ratio and scaling check")
print("=" * 74)
print("""
  δV₂ = +[σ*(1−σ*)(1−2σ*)/(2T²)]·(ℓᵢ²ℓⱼ + ℓᵢℓⱼ²)/a³·J(d)
  V₂  = −[σ*(1−σ*)/T]·(ℓᵢℓⱼ/a³)·I₂(d)

  δV₂/V₂ = −[(1−2σ*)/(2T)]·(ℓᵢ+ℓⱼ)·J(d)/I₂(d)
         = −[(1−2σ*)/(2πT)]·(ℓᵢ+ℓⱼ)/(a·d)          (d ≫ a, exact forms)
         = −[(1−2σ*)/(2π)]·[(ω_d,i+ω_d,j)/T]·(d/a)  with ω_d ≡ ℓ/d²

  Mass scaling ∝ (mᵢ+mⱼ)·L² CONFIRMED, but the distance scaling is
  1/(a·d), NOT 1/d²: the naive ω_d/T estimate is enhanced by d/a ≫ 1.
""")
# numeric verification of the ratio at a random point
sigma, T = 0.0832215, 1.0
li, lj = 2.0, 5.0
d = 300.0
lhs = -( (1 - 2 * sigma) / (2 * T) ) * (li + lj) * J_closed(d, A) / I2_closed(d, A)
rhs = -((1 - 2 * sigma) / (2 * np.pi * T)) * (li + lj) / (A * d)
print(f"  check at d/a=300, ℓᵢ=2, ℓⱼ=5: exact ratio = {lhs:.8e},")
print(f"    asymptotic formula = {rhs:.8e},  rel. diff = {lhs/rhs-1:+.1e}")
