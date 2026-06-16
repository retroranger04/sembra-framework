# SEMBRA Framework

**Surrogate for Elastomer Membrane Behavior under Radial loading and Actuation** — a surrogate framework for dielectric elastomer membrane instability.

This repository accompanies the paper *On the Instability of Inflated 
Dielectric Elastomer Membranes: Energy-Based Analysis and a Surrogate 
Modeling Framework* by Kumar, Mathur, and DasGupta. It provides two 
neural-surrogate packages that predict the equilibrium state and 
stability of an inflated circular dielectric elastomer membrane under 
combined mechanical pressure and electrical actuation.

## Packages

- **[SEMBRA](./SEMBRA/)** — voltage-controlled actuation. Use this when 
  the membrane is connected to a constant voltage source.
- **[SEMBRA_CC](./SEMBRA_CC/)** — charge-controlled actuation. Use this 
  when the membrane is initially charged and then disconnected from 
  the power supply.

Both packages share an identical architecture and methodology, differing 
only in the underlying electrostatic energy formulation. Each is 
self-contained and can be installed and used independently.

## Quick start

Choose the package corresponding to your loading scenario:

- For voltage-controlled analysis, see [SEMBRA's README](./SEMBRA/README.md).
- For charge-controlled analysis, see [SEMBRA_CC's README](./SEMBRA_CC/README.md).

Each package provides a `predict(alpha, alpha_p, p)` function and a 
Jupyter notebook for interactive use, along with full inference and 
validation pipelines.

## Requirements

- Python 3.11 or higher

See the individual package READMEs for dependency details.

## Citation

Citation: paper in preparation; the citation will be added here upon 
publication.

## License

MIT — see [SEMBRA/LICENSE](./SEMBRA/LICENSE) and 
[SEMBRA_CC/LICENSE](./SEMBRA_CC/LICENSE).
