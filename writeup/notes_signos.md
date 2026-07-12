# Auditoría de signos — convención de este repo vs. paper

**Paper:** Carney, Karydas, Scharnhorst, Singh, Taylor, PRX 15, 031038 (2025), arXiv:2502.17575.

## Derivación de primeros principios (nuestra convención)

La energía libre por sitio de un qubit de frecuencia ω a temperatura T y potencial
químico μ es

```
g(ω) = −T·ln(1 + e^{(μ−ω)/T})
```

Expandiendo en ω (verificado simbólicamente con sympy en `tests/test_expansion.py`,
tolerancia exacta):

```
g(ω) = g(0) + σ*·ω − [σ*(1−σ*)/(2T)]·ω² + [σ*(1−σ*)(1−2σ*)/(6T²)]·ω³ + O(ω⁴)
```

con σ* = 1/(e^{−μ/T} + 1). Los coeficientes se siguen de σ'(ω) = −σ(1−σ)/T con
σ(ω) = 1/(1 + e^{(ω−μ)/T}); no hay libertad de signo.

## Consecuencia: el coeficiente cuadrático es NEGATIVO

Con ω_α = Σ_i ℓ_i/(|r_α−x_i|²+a²) y el límite continuo Σ_α → ∫d³r/a³, el término
cruzado 2ℓ_iℓ_j del cuadrado da

```
V₂(d) = −[σ*(1−σ*)/T]·(ℓ_iℓ_j/a³)·I₂(d),   I₂(d) = π³·s(d)/d ≥ 0
```

V₂ < 0 y decreciente en magnitud con d → la fuerza F = −∂V₂/∂d es **atractiva**.
Verificado numéricamente en `tests/test_regression.py::test_sign_audit_force_attractive`
(derivada centrada, F < 0). Con la identificación G_N = σ*(1−σ*)·π³·L⁴/(T·a³)
(Eq. 33 del paper) se recupera exactamente V₂ = −G_N·m_i·m_j·s(d)/d, es decir
Newton atractivo con corrección de corto alcance s(d) = 1 − (2/π)·arctan(2a/d)
(test `test_V2_matches_newton_scaling`, coincidencia con la forma cerrada a 1e-12).

## La discrepancia aparente Eq. (27) vs Eq. (30) del paper

- La Eq. (27) del paper escribe la expansión de la energía libre con el
  coeficiente cuadrático **negativo**, consistente con nuestra derivación de
  primeros principios y con la frase del propio paper "controlled by the
  negative sign in front of the Σω² term".
- La Eq. (30) escribe el término con un signo que, tomado literalmente,
  daría una contribución **positiva** ∝ +ω² (fuerza repulsiva al identificar
  G_N > 0), en tensión con la identificación de G_N en la Eq. (33).

Nuestra derivación independiente (arriba) confirma que el signo correcto es el
de la Eq. (27): **coeficiente cuadrático negativo en la energía libre**. La
Eq. (30) parece tener una errata de signo (o una convención intermedia no
explicitada). Todo este repo usa la convención derivada de primeros principios:

| término | coeficiente | signo |
|---|---|---|
| ω¹ | σ* | + (constante, no genera fuerza) |
| ω² | −σ*(1−σ*)/(2T) | − → Newton atractivo |
| ω³ | +σ*(1−σ*)(1−2σ*)/(6T²) | + para σ* < 1/2 → V₃ > 0 (repulsivo) en el punto entrópico |

Nota: en σ* = 1/2 (μ = 0) el término cúbico se anula idénticamente; el punto
"puramente entrópico" del paper (μ = −2.39936·T, σ* = 0.0832215) tiene
(1−2σ*) = 0.8335570 > 0.
