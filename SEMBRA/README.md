# SEMBRA — Voltage-Controlled

**Surrogate for Elastomer Membrane Behavior under Radial loading and Actuation**

A trained neural surrogate that predicts equilibrium and instability behavior of a circular dielectric elastomer membrane under combined mechanical pressure and voltage-controlled electrical actuation. Replaces a slow numerical-continuation solver (~10–20 s per query) with a millisecond-latency surrogate that retains physical accuracy.

Companion package for the charge-controlled scenario: [SEMBRA_CC](https://github.com/retroranger04/sembra-framework/tree/main/SEMBRA_CC).

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

---

## Quick start

```bash
git clone https://github.com/retroranger04/sembra-framework.git
cd SEMBRA
pip install -r requirements.txt
```

Requires Python 3.11 or higher.

Then open `notebooks/sembra.ipynb` in Jupyter and run all cells. The prediction interface is in the third cell — change the three input values at the top of that cell, re-run the cell, and the surrogate will print the equilibrium state, stability verdict, and failure-mode diagnostics, along with a visualization of the deformed membrane.

For programmatic use:

```python
from sembra.wrapper import predict

result = predict(alpha=1.6, alpha_p=0.10, p=0.5)
print(result.stable, result.lambda_0, result.eta_0, result.det_H, result.message)
```

---

## What the outputs mean

- **λ₀** — pole stretch. Values > 1 mean the membrane is inflated. Larger λ₀ means more inflation.
- **η₀** — pole height (nondimensionalized by undeformed radius R). Larger means a taller dome.
- **det(H)** — sign of the Hessian determinant of the total potential energy. **Positive = stable equilibrium**, negative = unstable.
- **Wrinkling** — detects vanishing principal stress (loss of in-plane tension). Reports whether wrinkling is predicted and where.
- **Peak electric field** — reports the maximum normalized electric field Ē in the membrane. For voltage-controlled actuation, the maximum occurs at the pole (Ē_max = λ₀²).
- **Peak stresses** — reports the maximum in-plane Cauchy stresses (t₁, t₂) for the user to compare against the elastomer's rupture strength σ_crit and dielectric breakdown threshold E_crit.

---

## Reproducing paper figures

The notebook includes a parametric sweep section that reproduces the stability map shown in the source paper. To reproduce figures, simply run all cells.

---

## Technical details

- The surrogate is a composition of three trained components, assembled in `sembra.wrapper.predict`: (1) Gaussian-process **envelope models** for the critical pressure `p_crit` and limit-point stretch `lambda_limit`, (2) a **neural surrogate network** mapping `(α, αp, λ₀)` to the equilibrium, and (3) an **analytical Hessian** (`det H`) derived symbolically from the total potential energy for the exact stability verdict.
- The total potential energy, its equilibrium conditions, and the Hessian are derived symbolically (SymPy) and integrated over a fixed 1001-point radial grid with composite-trapezoidal weights.
- Training is **deterministic** (`seed = 42`): the symbolic cache, envelope GPs, and network reproduce identically across runs.
- Trained artifacts ship in `trained_models/` (`network.pt`, `pcrit_envelope.pkl`, `lambda_limit_envelope.pkl`, `symbolic_cache.pkl`); no training is required to run the model.

---

## Citation

If you use this surrogate in academic work, please cite:

> Citation: paper in preparation; the citation will be added here upon publication.

---

## Project structure

```
.
├── sembra/                 # the Python package
├── notebooks/
│   └── sembra.ipynb        # main user interface — start here
├── trained_models/         # trained surrogate weights and envelope models
├── tests/                  # inference smoke test (run: pytest tests/)
├── README.md
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

---

## Requirements & limitations

- Python ≥ 3.11
- See `requirements.txt` for the full pinned dependency list (or `pip install .` to use `pyproject.toml`)
- The surrogate was trained on parameter ranges α ∈ [1.2, 2.0], αp ∈ [0, 0.4], p ∈ [0, p_critical(α, αp)]. Extrapolation beyond these ranges is not recommended.
- The model is voltage-controlled. For charge-controlled actuation, see the [SEMBRA_CC](https://github.com/retroranger04/sembra-framework/tree/main/SEMBRA_CC) package.
- The test suite is a single self-contained inference smoke test (`tests/test_inference.py`); it exercises the public `predict` API against the shipped trained weights and requires no dataset.

**Note on the training dataset.** The trained surrogate weights are included in `trained_models/` and are sufficient for inference. The underlying dataset of equilibrium solutions, generated via numerical continuation in Mathematica, is not redistributed with this package. Researchers wishing to retrain, validate against ground truth, or extend the model are welcome to contact the authors.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for full text.
