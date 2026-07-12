"""Regression for the closed forms derived in Task 1/2:

  I₂(d; a, b) = (2π²/d)·arctan(d/(a+b))          (Fourier)
  J(d, a)     = π²/(a·(d²+4a²))                  (∂/∂a² of I₂)
  I₃(a=0)     = π³/(d₁₂·d₁₃·d₂₃)                 (conformal inversion)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrators import J, integrate3d
from src.analytic import I2_closed, J_closed, I3_feynman, I3_star_triangle


def test_I2_general_reduces_to_paper_form():
    # a = b must give π³·s(d)/d with s = 1 − (2/π)arctan(2a/d)
    d, a = 2.0, 0.05
    s = 1 - (2 / np.pi) * np.arctan(2 * a / d)
    assert abs(I2_closed(d, a) / (np.pi**3 * s / d) - 1) < 1e-14


def test_I2_general_closed_form_sympy():
    # Fourier route: I₂(d;a,b) = (4π⁴/(2π)³)·(4π/d)·∫₀^∞ dk e^{−(a+b)k} sin(kd)/k
    # and the radial integral is arctan(d/(a+b)), fully symbolic.
    k = sp.Symbol("k", positive=True)
    d, a, b = sp.symbols("d a b", positive=True)
    # Differentiate under the integral sign in d (the integrand vanishes at
    # d = 0): ∂_d [e^{−ck}·sin(kd)/k] = e^{−ck}·cos(kd), whose k-integral is
    # c/(c²+d²); re-integrating over d from 0 gives arctan(d/c), c = a+b.
    # (Antiderivative + limit avoids sympy's exp_polar artifacts on the
    # improper integral.)
    c, t = sp.symbols("c t", positive=True)
    anti = sp.integrate(sp.exp(-c * k) * sp.cos(k * t), k)
    ddint = sp.simplify(sp.limit(anti, k, sp.oo) - anti.subs(k, 0))
    assert sp.simplify(ddint - c / (c**2 + t**2)) == 0
    radial = sp.integrate(ddint, (t, 0, d)).subs(c, a + b)
    assert sp.simplify(radial - sp.atan(d / (a + b))) == 0
    I2_sym = (4 * sp.pi**4 / (2 * sp.pi) ** 3) * (4 * sp.pi / d) * radial
    assert sp.simplify(I2_sym - 2 * sp.pi**2 / d * sp.atan(d / (a + b))) == 0


def test_J_closed_form_sympy():
    # J = −∂I₂(d;a,b)/∂(a²)|_{b=a} = −(1/2a)·∂_a I₂|_{b=a}, symbolic and exact.
    d, a, b = sp.symbols("d a b", positive=True)
    I2_sym = 2 * sp.pi**2 / d * sp.atan(d / (a + b))
    J_sym = sp.simplify((-sp.diff(I2_sym, a) / (2 * a)).subs(b, a))
    expected = sp.pi**2 / (a * (d**2 + 4 * a**2))
    assert sp.simplify(J_sym - expected) == 0


def test_J_feynman_representation_mpmath():
    # 30-digit quadrature of J = (π²/2)∫₀¹ λdλ [λ(1−λ)d²+a²]^{−3/2} vs closed form
    import mpmath as mp

    mp.mp.dps = 30
    for d, a in [(1, mp.mpf(1) / 20), (3, mp.mpf(1) / 7)]:
        val = mp.pi**2 / 2 * mp.quad(
            lambda lam: lam / (lam * (1 - lam) * d**2 + a**2) ** mp.mpf(1.5), [0, 1]
        )
        expected = mp.pi**2 / (a * (d**2 + 4 * a**2))
        assert abs(val / expected - 1) < mp.mpf(10) ** (-25)


def test_J_numeric_vs_closed():
    d, a = 1.0, 0.05
    num = J([0, 0, 0], [d, 0, 0], a, nr=440, nth=144, nph=144)
    assert abs(num / J_closed(d, a) - 1) < 1e-4


def test_J_diverges_as_inverse_a():
    # J·a·(d²+4a²) is constant = π², so J does NOT converge as a → 0
    for a in [0.1, 0.01, 0.001]:
        assert abs(J_closed(1.0, a) * a * (1 + 4 * a**2) / np.pi**2 - 1) < 1e-14


@pytest.mark.parametrize("sides", [(1, 1, 1), (1, 2, 2.5), (3, 4, 5), (1, 1, 1.9)])
def test_I3_star_triangle_vs_feynman(sides):
    d12, d13, d23 = sides
    f = I3_feynman(d12, d13, d23, 0.0)
    st = np.pi**3 / (d12 * d13 * d23)
    assert abs(f / st - 1) < 1e-9


def test_I3_equilateral_pi3():
    # The conjecture, now theorem: I₃·d³ = π³ for the equilateral at a=0
    assert abs(I3_feynman(1, 1, 1, 0.0) / np.pi**3 - 1) < 1e-10
