# SEMBRA_CC — Charge-Controlled

**Surrogate for Elastomer Membrane Behavior under Radial loading and Actuation**

A trained neural surrogate that predicts equilibrium and instability behavior of a circular dielectric elastomer membrane under combined mechanical pressure and charge-controlled electrical actuation. Replaces a slow numerical-continuation solver (~10–20 s per query) with a millisecond-latency surrogate that retains physical accuracy.

Companion package for the voltage-controlled scenario: [SEMBRA](https://github.com/retroranger04/sembra-framework/tree/main/SEMBRA).

---

## What it does

Given three nondimensional inputs:

| Symbol | Meaning | Typical range |
|--------|---------|--------------|
| α      | Ogden hyperelastic exponent (material nonlinearity) | 1.2 – 2.0 |
| p      | Nondimensional inflation pressure | 0 – 2.5 |
| αp     | Nondimensional electrostatic loading parameter | 0 – 0.4 |

…the model returns:

- **Equilibrium state**: pole stretch λ₀, pole height η₀, Hessian determinant det(H)
- **Stability verdict** (electromechanical instability via Hessian sign)
- **Four failure-mode diagnostics**: electromechanical instability, wrinkling, peak electric field, peak stresses

Inference latency is ~1–2 ms per query.

Charge-controlled actuation has fundamentally different stability characteristics from voltage-controlled actuation — the instability boundary is nearly independent of electrical loading, giving inherent stabilization against electromechanical instability. The trade-off is earlier onset of wrinkling at lower αp. See the source paper for full discussion.

---

## Quick start

```bash
git clone https://github.com/retroranger04/sembra-framework.git
cd SEMBRA_CC
pip install -r requirements.txt
```

Requires Python 3.11 or higher.

Then open `notebooks/sembra_cc.ipynb` in Jupyter and run all cells. The prediction interface is in the last cell — change the three input values (`alpha`, `alpha_p`, `p`) at the top of that cell, re-run the cell, and the surrogate will print the equilibrium state, stability verdict, and failure-mode diagnostics, along with a visualization of the deformed membrane.

For a programmatic interface:

```python
from sembra_cc.wrapper import predict

result = predict(alpha=1.6, alpha_p=0.20, p=1.38)
print(result.stable, result.lambda_0, result.eta_0, result.det_H)
```

`result` is a `PredictionResult` dataclass with the echoed inputs, the equilibrium state, a stability boolean, a plain-language `message`, and a `diagnostics` dict (`p_crit`, `lambda_limit`, residuals).

---

## What the outputs mean

- **λ₀** — pole stretch. Values > 1 mean the membrane is inflated. Larger λ₀ means more inflation.
- **η₀** — pole height (nondimensionalized by undeformed radius R). Larger means a taller dome.
- **det(H)** — sign of the Hessian determinant of the total potential energy. **Positive = stable equilibrium**, negative = unstable.
- **Wrinkling** — detects vanishing principal stress (loss of in-plane tension). Reports whether wrinkling is predicted and where. Charge-controlled actuation tends to wrinkle at lower αp than voltage-controlled.
- **Peak electric field** — reports the maximum normalized electric field Ē in the membrane. For charge-controlled actuation, the maximum occurs at the rim (Ē_max = 1/√((3 − 2λ₀)² + 4η₀²)).
- **Peak stresses** — reports the maximum in-plane Cauchy stresses (t₁, t₂) for the user to compare against the elastomer's rupture strength σ_crit and dielectric breakdown threshold E_crit.

---

## Reproducing paper figures

The notebook visualizes, for your chosen `alpha`, the location of your query in the (αp, p) design plane against the predicted stability boundary p_crit(αp) — the stability map from the source paper. To reproduce it, simply run all cells.

---

## Technical details

- **Surrogate network** (`sembra_cc/network.py`): a small MLP, 3 → 128 → 128 → 128 → 3 with SiLU activations, mapping (α, αp, λ₀) → (p, η₀, det(H)). Trained with a data-MSE loss plus a physics-residual penalty (the equilibrium first-order conditions ∂Π/∂λ₀ = ∂Π/∂η₀ = 0) evaluated on both data points and random collocation points.
- **Symbolic core** (`sembra_cc/symbolic.py`): the total potential energy Π for the charge-controlled scenario is assembled in SymPy from the Ogden elastic energy, the electrostatic co-energy, and the pressure work, then differentiated to give the equilibrium residuals and the analytical 2×2 Hessian whose determinant is the stability indicator. The reported `det(H)` always comes from this analytical Hessian.
- **Stability envelope** (`sembra_cc/envelope.py`): two Gaussian-process regressors over (α, αp) predict the critical pressure p_crit and the limit-point stretch λ_limit, used to route a query into its stable / grey-zone / past-limit branch.
- **Kinematics**: a two-parameter Ritz ansatz λ₂(r) = r² + λ₀(1 − r²), η(r) = η₀(1 − r²), with incompressibility λ₁λ₂λ₃ = 1, reduces the field problem to the two amplitudes (λ₀, η₀).
- **Data**: the surrogate is trained on a dataset of equilibrium solutions generated via numerical continuation; held-out slices are used for validation in `sembra_cc/validation.py`. See the note on the training dataset under *Requirements & limitations*.

---

## Citation

If you use this surrogate in academic work, please cite:

> Citation: paper in preparation; the citation will be added here upon publication.

---

## Project structure

```
.
├── sembra_cc/              # the Python package
├── notebooks/
│   └── sembra_cc.ipynb     # main user interface — start here
├── trained_models/         # trained surrogate weights and envelope models
├── tests/                  # test suite (run: pytest tests/)
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Requirements & limitations

- Python ≥ 3.11
- See `requirements.txt` for the full dependency list. Run the tests with `pytest tests/` (37 tests).
- The surrogate was trained on parameter ranges α ∈ [1.2, 2.0], αp ∈ [0, 0.4], p ∈ [0, p_critical(α, αp)]. Extrapolation beyond these ranges is not recommended.
- The model is charge-controlled. For voltage-controlled actuation, see the [SEMBRA](https://github.com/retroranger04/sembra-framework/tree/main/SEMBRA) package.

**Note on the training dataset.** The trained surrogate weights are included in `trained_models/` and are sufficient for inference. The underlying dataset of equilibrium solutions, generated via numerical continuation in Mathematica, is not redistributed with this package. Researchers wishing to retrain, validate against ground truth, or extend the model are welcome to contact the authors.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for full text.
