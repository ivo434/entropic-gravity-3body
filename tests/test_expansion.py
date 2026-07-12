"""Gate 0.2, symbolic regression: expansion of g(ω) to O(ω³).

g(ω) = −T·ln(1 + e^{(μ−ω)/T}),  σ* = 1/(e^{−μ/T} + 1)

Expected (exact symbolic match required):
  g(ω) = g(0) + σ*·ω − [σ*(1−σ*)/(2T)]·ω² + [σ*(1−σ*)(1−2σ*)/(6T²)]·ω³ + O(ω⁴)
"""

import sympy as sp


def test_g_expansion_coefficients():
    w, T, mu = sp.symbols("omega T mu", positive=True, real=True)
    mu = sp.Symbol("mu", real=True)

    g = -T * sp.log(1 + sp.exp((mu - w) / T))
    sigma = 1 / (sp.exp(-mu / T) + 1)

    series = sp.series(g, w, 0, 4).removeO()

    c0 = series.coeff(w, 0)
    c1 = series.coeff(w, 1)
    c2 = series.coeff(w, 2)
    c3 = series.coeff(w, 3)

    expected0 = -T * sp.log(1 + sp.exp(mu / T))
    expected1 = sigma
    expected2 = -sigma * (1 - sigma) / (2 * T)
    expected3 = sigma * (1 - sigma) * (1 - 2 * sigma) / (6 * T**2)

    assert sp.simplify(c0 - expected0) == 0, f"g(0) mismatch: {c0}"
    assert sp.simplify(c1 - expected1) == 0, f"linear coeff mismatch: {c1}"
    assert sp.simplify(c2 - expected2) == 0, f"quadratic coeff mismatch: {c2}"
    assert sp.simplify(c3 - expected3) == 0, f"cubic coeff mismatch: {c3}"


def test_cubic_vanishes_at_half_filling():
    # At μ = 0, σ* = 1/2 and the cubic term must vanish identically.
    sigma = sp.Rational(1, 2)
    T = sp.Symbol("T", positive=True)
    assert sp.simplify(sigma * (1 - sigma) * (1 - 2 * sigma) / (6 * T**2)) == 0


def test_entropic_point_values():
    # μ = −2.39936·T → σ* = 0.0832215, (1−2σ*) = 0.8335570
    import numpy as np

    sigma = 1 / (np.exp(2.39936) + 1)
    assert abs(sigma - 0.0832215) < 1e-6
    assert abs((1 - 2 * sigma) - 0.8335570) < 1e-6
