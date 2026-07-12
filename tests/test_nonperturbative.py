"""Star-triangle limit and non-perturbative (Moebius) consistency gates.

A. I₃·(r₁₂r₁₃r₂₃)/π³ → 1 as a → 0, three geometries; correction structure
   1 − (2a/π)·Σᵢ 1/dᵢ (one factor 2/π per regulated denominator).
B. Gate 2.2: Möbius-subtracted A₂/A₃ with FULL g reproduce the perturbative
   closed forms at 1% in a regime with ω ≪ T everywhere.
   S_moment reproduces its perturbative limit 3c₃ℓ²π²/a.
C. Ball-split: the 10a-balls around the masses contribute O(a/d) to I₃
   (three-body term is infrared-dominated → trustworthy);
   for J they contribute O(1) (UV-dominated → not trustworthy).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrators import integrate3d, lorentz, I3
from src.analytic import I3_feynman, I3_feynman_1d, I2_closed, J_closed
from src.nonperturbative import (A2_full, A3_full, S_moment, S_perturbative,
                                 g_free)

MU_OVER_T = -2.39936  # entropic point


def geometries():
    eq = [(0, 0, 0), (1, 0, 0), (0.5, np.sqrt(3) / 2, 0)]
    # isosceles, apex 30°, legs 1 → base 2·sin15°
    base = 2 * np.sin(np.radians(15))
    iso = [(0, 0, 0), (base, 0, 0), (base / 2, np.sqrt(1 - base**2 / 4), 0)]
    # scalene sides (1, 1.5, 2), the verification benchmark
    x = (1 + 4 - 2.25) / 4  # place: d12=2, d13=1, d23=1.5
    sc = [(0, 0, 0), (2, 0, 0), (x, np.sqrt(1 - x**2), 0)]
    return {"equilateral": eq, "isosceles30": iso, "scalene(1,1.5,2)": sc}


@pytest.mark.parametrize("name,pos", list(geometries().items()))
def test_I3_a_to_0_three_geometries(name, pos):
    p = [np.asarray(q, float) for q in pos]
    d12 = np.linalg.norm(p[0] - p[1])
    d13 = np.linalg.norm(p[0] - p[2])
    d23 = np.linalg.norm(p[1] - p[2])
    pred = np.pi**3 / (d12 * d13 * d23)
    sum_inv = 1 / d12 + 1 / d13 + 1 / d23
    prev_dev = None
    for a in (0.008, 0.002):
        ratio = I3_feynman(d12, d13, d23, a) / pred
        dev = 1 - ratio
        # approaches 1 from below, deviation O(a/d): matches (2a/π)Σ1/dᵢ to ~15%
        assert 0 < dev < 3 * a * sum_inv
        assert abs(dev / ((2 * a / np.pi) * sum_inv) - 1) < 0.15, (name, a, dev)
        if prev_dev is not None:
            assert abs(prev_dev / dev - 4) < 0.6  # linear in a: factor-4 drop
        prev_dev = dev


def test_addendum_scalene_regression():
    # independent 3D-grid check: (1,1.5,2), a=0.008 gave I₃/pred = 0.9897.
    # Exact Feynman quadrature gives 0.98766; the 2e-3 gap is that grid's
    # discretization error. Structural prediction 1−(2a/π)Σ1/dᵢ = 0.98897
    # brackets both.
    ratio = I3_feynman(1, 1.5, 2, 0.008) / (np.pi**3 / (1 * 1.5 * 2))
    assert abs(ratio - 0.9897) < 4e-3
    assert abs(ratio - (1 - (2 * 0.008 / np.pi) * (1 + 1 / 1.5 + 1 / 2))) < 1.5e-3


# ------------------------------------------------------------- gate 2.2
# Toy regime: a/d = 0.2 keeps the near-mass ω(a) = ℓ/a² ≪ T while leaving
# enough signal above float64 cancellation noise. References use the exact
# finite-a forms (I₂ closed form; I₃ Feynman quadrature), whose a→0 limits
# were validated in Task 1.
T_, A_, D_ = 1.0, 0.2, 1.0
MU_ = MU_OVER_T * T_
SIG_ = 1.0 / (1.0 + np.exp(-MU_OVER_T))
ELL_ = 0.03 * A_**2 * T_  # ω(0) = ℓ/a² = 0.03·T


def test_gate22_A2_matches_perturbative():
    a2 = A2_full([0, 0, 0], [D_, 0, 0], ELL_, ELL_, T_, MU_, A_,
                 nr=440, nth=192, nph=192)
    pred = -(SIG_ * (1 - SIG_) / T_) * (ELL_**2 / A_**3) * I2_closed(D_, A_)
    assert abs(a2 / pred - 1) < 0.01, (a2, pred)


def test_gate22_A3_matches_perturbative():
    pos = [np.array(p, float) for p in geometries()["equilateral"]]
    a3 = A3_full(pos, [ELL_] * 3, T_, MU_, A_, nr=440, nth=192, nph=192)
    c3 = SIG_ * (1 - SIG_) * (1 - 2 * SIG_) / (6 * T_**2)
    pred = 6 * c3 * (ELL_**3 / A_**3) * I3_feynman(1, 1, 1, A_)
    assert abs(a3 / pred - 1) < 0.01, (a3, pred)


def test_S_moment_perturbative_limit():
    # ℓ/a² = 1e-3·T → deep perturbative; S → 3c₃ℓ²π²/a
    ell = 1e-3 * A_**2 * T_
    s = S_moment(ell, T_, MU_, A_)
    assert abs(s / S_perturbative(ell, T_, MU_, A_) - 1) < 1e-3


def test_S_moment_saturated_scaling():
    # deep saturation r* ≫ a: S must scale as r*³ = (ℓ/T)^{3/2}
    # (ratio of S at ℓ and 4ℓ → 8)
    ell = 1e6 * A_**2 * T_
    s1 = S_moment(ell, T_, MU_, A_)
    s2 = S_moment(4 * ell, T_, MU_, A_)
    assert s1 > 0
    assert abs(s2 / s1 / 8 - 1) < 0.05


# ------------------------------------------------------------- ball split
def _masked_I3(pos, a, inside, nr=440, nth=192, nph=192):
    p = [np.asarray(q, float) for q in pos]
    center = sum(p) / 3.0
    dmax = max(np.linalg.norm(p[i] - p[j]) for i in range(3) for j in range(i + 1, 3))

    def f(X, Y, Z):
        val = (lorentz(X, Y, Z, p[0], a) * lorentz(X, Y, Z, p[1], a)
               * lorentz(X, Y, Z, p[2], a))
        r2min = np.minimum.reduce(
            [(X - q[0]) ** 2 + (Y - q[1]) ** 2 + (Z - q[2]) ** 2 for q in p]
        )
        mask = r2min < (10 * a) ** 2
        return val * np.where(mask == inside, 1.0, 0.0)

    return integrate3d(f, center, 2 * dmax, nr, nth, nph)


def test_I3_infrared_dominated_but_J_is_not():
    a, d = 0.005, 1.0
    pos = geometries()["equilateral"]
    balls = _masked_I3(pos, a, inside=True)
    total = I3(*[np.asarray(q, float) for q in pos], a, nr=440, nth=192, nph=192)
    frac = balls / total
    # 10a-balls contribute O(a/d) to I₃ → the three-body term is trustworthy
    assert frac < 20 * a / d, f"ball fraction {frac}"
    # same split for J: the r<10a ball around the squared center holds
    # nearly everything (1 − 1/(1+100) ≈ 99%): UV-dominated
    # analytic: ∫₀^R r²/(r²+a²)² dr = (1/2)[−R/(R²+a²) + arctan(R/a)/a]·4π
    R = 10 * a
    ball_J = 4 * np.pi * 0.5 * (-R / (R**2 + a**2) + np.arctan(R / a) / a)
    frac_J = ball_J * (1 / d**2) / J_closed(d, a)
    assert frac_J > 0.85, f"J ball fraction {frac_J}"


def test_I3_1d_reduction_solar_hierarchy():
    # 1D-reduced Feynman quadrature must match the star-triangle closed form
    # on extreme (solar-like) hierarchies where 2D dblquad fails.
    for sides in [(1, 1, 1), (1, 1, 1e-2), (1.0, 0.99871, 2.5695e-3)]:
        v = I3_feynman_1d(*sides)
        st = np.pi**3 / np.prod(sides)
        assert abs(v / st - 1) < 1e-10, sides
