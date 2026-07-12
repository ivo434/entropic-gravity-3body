"""Gates 0.3 / 0.4, numeric regression against preliminary exploration, and
sign audit of the two-body force.

Regression values (a = 0.05, baseline grid nr=220, nth=96, nph=96):
  I₂: d=1 → 29.0398, d=2 → 15.0088, d=4 → 7.6194   (vs closed form)
  I₃ equilateral side d: d=1 → 2.7955e1, d=2 → 3.6875e0,
                         d=4 → 4.7277e-1, d=8 → 5.9783e-2
Tolerance: 0.5%.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.integrators import I2, I2_exact, I3
from src.model import V2

A = 0.05
TOL = 5e-3


@pytest.mark.parametrize(
    "d,expected",
    [(1.0, 29.0398), (2.0, 15.0088), (4.0, 7.6194)],
)
def test_I2_regression(d, expected):
    val = I2([0, 0, 0], [d, 0, 0], A)
    assert abs(val / expected - 1) < TOL, f"I2(d={d}) = {val}, expected {expected}"
    # and against the closed form
    assert abs(val / I2_exact(d, A) - 1) < TOL


def equilateral(d):
    return (
        np.array([0.0, 0.0, 0.0]),
        np.array([d, 0.0, 0.0]),
        np.array([d / 2, d * np.sqrt(3) / 2, 0.0]),
    )


# d=8 (a/d = 6.25e-3) has narrow Lorentzian peaks relative to R0 = 2d and
# needs a finer grid: at the baseline grid the discretization error is 0.62%,
# converging monotonically to the regression value (5.975e-2 at 2× grid,
# 5.981e-2 at 3x, within 0.05% of 5.9783e-2). Verified via Richardson study.
@pytest.mark.parametrize(
    "d,expected,grid",
    [
        (1.0, 2.7955e1, (220, 96, 96)),
        (2.0, 3.6875e0, (220, 96, 96)),
        (4.0, 4.7277e-1, (220, 96, 96)),
        (8.0, 5.9783e-2, (440, 192, 192)),
    ],
)
def test_I3_regression(d, expected, grid):
    p1, p2, p3 = equilateral(d)
    nr, nth, nph = grid
    val = I3(p1, p2, p3, A, nr=nr, nth=nth, nph=nph)
    assert abs(val / expected - 1) < TOL, f"I3(d={d}) = {val}, expected {expected}"


def test_sign_audit_force_attractive():
    """F = −∂V₂/∂d must be attractive (F < 0, pointing toward smaller d)
    with our convention: negative quadratic coefficient in the free energy.
    """
    sigma, T, a, ell = 0.0832215, 1.0, 0.05, 1.0
    d = 2.0
    h = 1e-4
    dV = (V2(d + h, ell, ell, sigma, T, a) - V2(d - h, ell, ell, sigma, T, a)) / (2 * h)
    F = -dV
    # V₂ ~ −C/d (C>0): dV/dd > 0, so F = −dV/dd < 0 → attractive
    assert V2(d, ell, ell, sigma, T, a) < 0, "V2 must be negative (bound state)"
    assert F < 0, f"Force must be attractive, got F = {F}"


def test_V2_matches_newton_scaling():
    """At large d/a, V₂ → −G_N·m_i·m_j/d with G_N = σ(1−σ)π³L⁴/(T a³)."""
    sigma, T, a = 0.0832215, 1.0, 0.05
    L = 1.0
    m_i, m_j = 2.0, 3.0
    ell_i, ell_j = m_i * L**2, m_j * L**2
    d = 50.0  # d/a = 1000
    GN = sigma * (1 - sigma) * np.pi**3 * L**4 / (T * a**3)
    v2 = V2(d, ell_i, ell_j, sigma, T, a)
    # Exact relation: V₂ = −G_N·m_i·m_j·s(d)/d with s(d) = 1 − (2/π)arctan(2a/d)
    s = 1 - (2 / np.pi) * np.arctan(2 * a / d)
    newton = -GN * m_i * m_j / d
    assert abs(v2 / (newton * s) - 1) < 1e-12   # closed form, exact
    assert abs(v2 / newton - 1) < 2e-3          # → Newton as a/d → 0
